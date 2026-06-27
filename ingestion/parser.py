"""
Parser — extracts clean text from any document format.

All parsers return List[{"page": int, "text": str}].

Content is extracted in reading order — the exact order it visually appears
on the page. Text, tables, and images are interleaved so the LLM receives
full context ("as shown in the chart above" stays next to the chart).

Reading-order strategy per format:
  PDF    — elements sorted by y-coordinate; multi-column layout detected and
            handled (left column top-to-bottom, then right column);
            scanned pages (empty text layer) rendered as full-page image to Vision
  DOCX   — body XML iterated in document order; inline images handled inside
            their parent paragraph; headers/footers extracted
  HTML   — DOM tree walked in traversal order; one pass, no separation
  Excel  — drawing XML parsed for image AND native chart positions;
            content inserted at anchor row in the markdown table output
  Others — inherently single-type content; reading order not applicable
"""

import base64
import email as email_lib
import logging
import os
import re
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from email import policy
from typing import Any, Dict, List, Tuple

import fitz  # PyMuPDF
import pandas as pd
from bs4 import BeautifulSoup, NavigableString, Tag
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ── Shared helpers ────────────────────────────────────────────────────────────


def _extract_image_text(image_bytes: bytes, media_type: str = "image/png") -> str:
    """
    Send image bytes to Claude Vision, return extracted text.

    Claude understands layout — tables in images, charts, multi-column text —
    where traditional OCR (Tesseract) fails on complex financial documents.
    """
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set — required for image parsing")

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64.standard_b64encode(image_bytes).decode(),
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "Extract all text from this image exactly as it appears. "
                        "If there are tables, preserve their structure in markdown format. "
                        "If there are charts or graphs, describe the data they show numerically. "
                        "Return only the extracted content, no commentary."
                    ),
                },
            ],
        }],
    )
    return response.content[0].text


def _rows_to_markdown(rows: List[List[str]]) -> str:
    """Convert a 2D list of cell strings into a markdown table."""
    if not rows:
        return ""
    header = "| " + " | ".join(str(c) for c in rows[0]) + " |"
    sep = "| " + " | ".join(["---"] * len(rows[0])) + " |"
    body = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows[1:]]
    return "\n".join([header, sep] + body)


def _media_type(ext: str) -> str:
    return "image/jpeg" if ext.lower() in ("jpg", "jpeg") else f"image/{ext.lower()}"


def _bbox_overlaps(b1: Tuple, b2: Tuple) -> bool:
    """True if two bounding boxes (x0, y0, x1, y1) overlap."""
    return not (b1[2] <= b2[0] or b1[0] >= b2[2] or b1[3] <= b2[1] or b1[1] >= b2[3])


def _reading_order(elements: List[Dict], page_width: float) -> List[Dict]:
    """
    Return elements in correct visual reading order.

    Single column: sort all by y-coordinate.
    Two column: detected when >= 25% of text blocks fall in each half of the
    page width. Left column blocks (sorted by y) output first, then full-width
    elements (tables/images spanning > 60% of page width, sorted by y), then
    right column blocks (sorted by y).

    ponytail: two-column detection uses page midpoint as boundary — sufficient
    for standard two-column financial reports; irregular layouts may not sort
    perfectly. Upgrade to x-coordinate clustering if needed.
    """
    if not elements:
        return []

    mid = page_width / 2

    def cx(e: Dict) -> float:
        return (e.get("x0", 0) + e.get("x1", e.get("x0", 0))) / 2

    text_els = [e for e in elements if e.get("etype") == "text"]
    non_text = [e for e in elements if e.get("etype") != "text"]

    if not text_els:
        return sorted(elements, key=lambda e: e["y"])

    left_text = [e for e in text_els if cx(e) < mid]
    right_text = [e for e in text_els if cx(e) >= mid]
    total = len(text_els)

    is_two_col = total >= 4 and len(left_text) / total >= 0.25 and len(right_text) / total >= 0.25

    if not is_two_col:
        return sorted(elements, key=lambda e: e["y"])

    # Separate non-text into full-width vs column-specific
    full_w = page_width * 0.6
    full_width_els = [e for e in non_text if (e.get("x1", 0) - e.get("x0", 0)) >= full_w]
    left_non = [e for e in non_text if (e.get("x1", 0) - e.get("x0", 0)) < full_w and cx(e) < mid]
    right_non = [e for e in non_text if (e.get("x1", 0) - e.get("x0", 0)) < full_w and cx(e) >= mid]

    return (
        sorted(left_text + left_non, key=lambda e: e["y"]) +
        sorted(full_width_els, key=lambda e: e["y"]) +
        sorted(right_text + right_non, key=lambda e: e["y"])
    )


# ── PDF ───────────────────────────────────────────────────────────────────────


def parse_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    PDF — reading-order extraction with four passes resolved into one sorted list.

    Per page:
    1. Collect text blocks (excluding regions covered by tables) with x, y positions.
    2. Collect tables with their bounding box positions.
    3. Collect images (> 50×50px) with positions → sent to Claude Vision.
    4. Sort all elements by reading order (handles single and two-column layouts).
    5. If page is empty after all passes → render full page as image → Claude Vision.
       This handles scanned PDFs where pages are images with no text layer.
    """
    doc = fitz.open(file_path)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        elements: List[Dict] = []

        tables = list(page.find_tables())
        table_bboxes = [t.bbox for t in tables]

        # Text blocks — skip regions that belong to tables (avoid duplication)
        for block in page.get_text("blocks"):
            x0, y0, x1, y1, text, _, block_type = block
            if block_type != 0 or not text.strip():
                continue
            if any(_bbox_overlaps((x0, y0, x1, y1), tb) for tb in table_bboxes):
                continue
            elements.append({"etype": "text", "y": y0, "x0": x0, "x1": x1, "content": text.strip()})

        # Tables
        for table in tables:
            rows = [[str(cell or "").strip() for cell in row] for row in table.extract()]
            md = _rows_to_markdown(rows)
            if md:
                b = table.bbox
                elements.append({"etype": "table", "y": b[1], "x0": b[0], "x1": b[2], "content": f"[Table]\n{md}"})

        # Images (skip decorative elements smaller than 50×50px)
        for img_info in page.get_image_info():
            if img_info.get("width", 0) < 50 or img_info.get("height", 0) < 50:
                continue
            xref = img_info["xref"]
            b = img_info["bbox"]
            try:
                img_data = doc.extract_image(xref)
                img_text = _extract_image_text(img_data["image"], _media_type(img_data["ext"]))
                if img_text.strip():
                    elements.append({"etype": "image", "y": b[1], "x0": b[0], "x1": b[2], "content": f"[Image]\n{img_text}"})
            except Exception as e:
                logger.warning(f"PDF page {page_num} image extraction failed: {e}")

        # Scanned page fallback: if nothing was extracted, the page is likely a
        # full-page scan with no text layer. Render the page as a bitmap and
        # send it to Claude Vision to recover all content.
        if not elements:
            try:
                pix = page.get_pixmap(dpi=150)
                img_text = _extract_image_text(pix.tobytes("png"), "image/png")
                if img_text.strip():
                    elements.append({"etype": "image", "y": 0, "x0": 0, "x1": page.rect.width, "content": f"[Scanned page]\n{img_text}"})
            except Exception as e:
                logger.warning(f"PDF page {page_num} scanned-page render failed: {e}")

        ordered = _reading_order(elements, page.rect.width)
        if ordered:
            pages.append({"page": page_num, "text": "\n\n".join(e["content"] for e in ordered)})

    doc.close()
    return pages


# ── DOCX ──────────────────────────────────────────────────────────────────────


def _docx_header_footer_text(hf_obj) -> str:
    """Extract plain text from a DOCX header or footer object."""
    if hf_obj.is_linked_to_previous:
        return ""
    lines = [p.text.strip() for p in hf_obj.paragraphs if p.text.strip()]
    return "\n".join(lines)


def parse_docx(file_path: str) -> List[Dict[str, Any]]:
    """
    DOCX — reading-order extraction via body XML iteration.

    Extracts in document order:
    - Headers (from first section — financial docs rarely have per-section headers)
    - Body: paragraphs and tables interleaved as they appear in the XML
    - Inline images extracted inside the paragraph they belong to, in order
    - Footers (from first section)

    DOCX has no stored page numbers. Everything returns as page 1.
    """
    doc = Document(file_path)
    parts: List[str] = []

    # Headers — prepend before body content
    if doc.sections:
        header_text = _docx_header_footer_text(doc.sections[0].header)
        if header_text:
            parts.append(f"[Header]\n{header_text}")

    # Body: iterate XML in document order
    for element in doc.element.body:
        tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

        if tag == "p":
            para = DocxParagraph(element, doc)
            if para.text.strip():
                parts.append(para.text.strip())

            # Inline images live inside paragraphs — handle here to preserve order
            for drawing in element.findall(".//" + qn("w:drawing")):
                for blip in drawing.findall(".//" + qn("a:blip")):
                    r_id = blip.get(qn("r:embed"))
                    if r_id and r_id in doc.part.rels:
                        rel = doc.part.rels[r_id]
                        if "image" in rel.reltype:
                            try:
                                ext = rel.target_ref.split(".")[-1].lower()
                                img_text = _extract_image_text(rel.target_part.blob, _media_type(ext))
                                if img_text.strip():
                                    parts.append(f"[Image]\n{img_text}")
                            except Exception as e:
                                logger.warning(f"DOCX image extraction failed: {e}")

        elif tag == "tbl":
            table = DocxTable(element, doc)
            rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
            md = _rows_to_markdown(rows)
            if md:
                parts.append(f"[Table]\n{md}")

    # Footers — append after body content
    if doc.sections:
        footer_text = _docx_header_footer_text(doc.sections[0].footer)
        if footer_text:
            parts.append(f"[Footer]\n{footer_text}")

    return [{"page": 1, "text": "\n\n".join(parts)}] if parts else []


# ── Excel ─────────────────────────────────────────────────────────────────────


def _parse_chart_xml(chart_bytes: bytes) -> str:
    """
    Extract data from an Excel native chart XML (xl/charts/chart*.xml).

    Excel caches chart data inside the chart XML on save. We extract:
    - Chart title
    - Chart type (bar, line, pie, etc.)
    - Each data series: name, category labels, and values → markdown table

    This makes native chart objects visible to the LLM — previously they were
    completely invisible to any parsing approach since they are not image files.
    """
    ns = {
        "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    }

    root = ET.fromstring(chart_bytes)
    parts: List[str] = []

    # Chart title
    title_parts = (
        [t.text.strip() for t in root.findall(".//c:title//a:t", ns) if t.text] +
        [t.text.strip() for t in root.findall(".//c:title//c:v", ns) if t.text]
    )
    if title_parts:
        parts.append(f"Chart: {' '.join(title_parts)}")

    # Detect chart type for context
    chart_type_tags = ["barChart", "lineChart", "pieChart", "scatterChart",
                       "areaChart", "doughnutChart", "radarChart", "bubbleChart"]
    for ct in chart_type_tags:
        if root.find(f".//c:{ct}", ns) is not None:
            if not title_parts:
                parts.append(f"{ct.replace('Chart', ' Chart').title()}")
            break

    # Extract each series
    for ser in root.findall(".//c:ser", ns):
        series_name = ""
        for v in ser.findall(".//c:tx//c:v", ns):
            if v.text:
                series_name = v.text.strip()
                break
        for t in ser.findall(".//c:tx//a:t", ns):
            if t.text:
                series_name = t.text.strip()
                break

        categories = [
            pt.find("c:v", ns).text.strip()
            for pt in ser.findall(".//c:cat//c:pt", ns)
            if pt.find("c:v", ns) is not None and pt.find("c:v", ns).text
        ]
        values = [
            pt.find("c:v", ns).text.strip()
            for pt in (ser.findall(".//c:val//c:pt", ns) + ser.findall(".//c:yVal//c:pt", ns))
            if pt.find("c:v", ns) is not None and pt.find("c:v", ns).text
        ]

        if categories and values:
            label = f"Series: {series_name}" if series_name else "Data"
            rows = [["Category", "Value"]] + [[c, v] for c, v in zip(categories, values)]
            parts.append(f"{label}\n{_rows_to_markdown(rows)}")
        elif values:
            label = f"Series: {series_name}" if series_name else "Values"
            parts.append(f"{label}: {', '.join(values)}")

    return "\n\n".join(parts)


def _excel_drawing_content(file_path: str) -> List[Tuple[int, str]]:
    """
    Extract all embedded content (images AND native charts) from xlsx with
    their row anchor positions.

    Returns: List of (row_index, text_content) sorted by row_index.

    xlsx structure explored:
    - xl/drawings/drawing*.xml           → anchor positions + rId references
    - xl/drawings/_rels/drawing*.xml.rels → rId → image or chart file
    - xl/media/image*.png etc            → image files
    - xl/charts/chart*.xml              → chart data XML
    """
    results: List[Tuple[int, str]] = []

    r_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    r_embed = f"{{{r_ns}}}embed"

    ns = {
        "xdr": "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing",
        "a":   "http://schemas.openxmlformats.org/drawingml/2006/main",
        "c":   "http://schemas.openxmlformats.org/drawingml/2006/chart",
    }

    with zipfile.ZipFile(file_path, "r") as z:
        all_files = set(z.namelist())

        for i in range(1, 20):
            drawing_path = f"xl/drawings/drawing{i}.xml"
            rels_path = f"xl/drawings/_rels/drawing{i}.xml.rels"
            if drawing_path not in all_files:
                break

            # Map rId → (file_path, "image"|"chart")
            r_id_map: Dict[str, Tuple[str, str]] = {}
            if rels_path in all_files:
                for rel in ET.fromstring(z.read(rels_path)):
                    r_id = rel.get("Id", "")
                    target = rel.get("Target", "")
                    if "../media/" in target:
                        r_id_map[r_id] = ("xl/media/" + target.split("../media/")[-1], "image")
                    elif "../charts/" in target:
                        r_id_map[r_id] = ("xl/charts/" + target.split("../charts/")[-1], "chart")

            draw_root = ET.fromstring(z.read(drawing_path))

            for anchor_tag in ("xdr:twoCellAnchor", "xdr:oneCellAnchor"):
                for anchor in draw_root.findall(anchor_tag, ns):
                    row = 0
                    from_elem = anchor.find("xdr:from", ns)
                    if from_elem is not None:
                        row_elem = from_elem.find("xdr:row", ns)
                        if row_elem is not None and row_elem.text:
                            row = int(row_elem.text)

                    # Images — referenced via a:blip
                    for blip in anchor.findall(".//a:blip", ns):
                        r_id = blip.get(r_embed, "")
                        if r_id in r_id_map and r_id_map[r_id][1] == "image":
                            img_file, _ = r_id_map[r_id]
                            if img_file in all_files:
                                try:
                                    ext = os.path.splitext(img_file)[1].lower().lstrip(".")
                                    img_text = _extract_image_text(z.read(img_file), _media_type(ext))
                                    if img_text.strip():
                                        results.append((row, f"[Image]\n{img_text}"))
                                except Exception as e:
                                    logger.warning(f"Excel image extraction failed: {e}")

                    # Charts — referenced via c:chart element
                    for chart_ref in anchor.findall(".//c:chart", ns):
                        r_id = chart_ref.get(r_embed, "")
                        if r_id in r_id_map and r_id_map[r_id][1] == "chart":
                            chart_file, _ = r_id_map[r_id]
                            if chart_file in all_files:
                                try:
                                    chart_text = _parse_chart_xml(z.read(chart_file))
                                    if chart_text.strip():
                                        results.append((row, f"[Chart]\n{chart_text}"))
                                except Exception as e:
                                    logger.warning(f"Excel chart extraction failed: {e}")

    results.sort(key=lambda x: x[0])
    return results


def parse_excel(file_path: str) -> List[Dict[str, Any]]:
    """
    Excel — cell values as markdown table with images and native charts
    inserted at their row anchor positions.

    Each sheet = one page. .xls (legacy binary) — cell values only,
    no drawing content extraction (not a ZIP archive).
    """
    xl = pd.ExcelFile(file_path)
    pages = []

    drawing_content: List[Tuple[int, str]] = []
    if file_path.lower().endswith(".xlsx"):
        try:
            drawing_content = _excel_drawing_content(file_path)
        except Exception as e:
            logger.warning(f"Excel drawing content extraction failed: {e}")

    for sheet_num, sheet_name in enumerate(xl.sheet_names, start=1):
        df = xl.parse(sheet_name)
        parts: List[str] = [f"Sheet: {sheet_name}"]

        if not df.empty:
            md_lines = df.to_markdown(index=False).split("\n")
            result_lines = md_lines[:2]  # header + separator

            for data_row_idx, line in enumerate(md_lines[2:]):
                result_lines.append(line)
                for row_anchor, embedded_text in drawing_content:
                    if row_anchor == data_row_idx:
                        result_lines.append(f"\n{embedded_text}")

            parts.append("\n".join(result_lines))

            # Content anchored beyond the last data row
            last_row = len(md_lines) - 3
            for row_anchor, embedded_text in drawing_content:
                if row_anchor > last_row:
                    parts.append(embedded_text)

        if len(parts) > 1:
            pages.append({"page": sheet_num, "text": "\n\n".join(parts)})

    return pages


# ── CSV ───────────────────────────────────────────────────────────────────────


def parse_csv(file_path: str) -> List[Dict[str, Any]]:
    """CSV — rows as markdown table. Images are impossible in this format."""
    df = pd.read_csv(file_path)
    return [{"page": 1, "text": df.to_markdown(index=False)}]


# ── HTML ──────────────────────────────────────────────────────────────────────


def _resolve_html_image(src: str, base_dir: str) -> str:
    """Resolve any img src type (base64, relative path, remote URL) → Vision text."""
    mt_map = {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png", "gif": "image/gif", "webp": "image/webp",
    }
    try:
        if src.startswith("data:image/"):
            match = re.match(r"data:(image/\w+);base64,(.+)", src, re.DOTALL)
            if match:
                return _extract_image_text(base64.b64decode(match.group(2)), match.group(1))

        elif src.startswith(("http://", "https://")):
            with urllib.request.urlopen(src, timeout=10) as resp:
                img_bytes = resp.read()
            ext = src.split(".")[-1].lower().split("?")[0]
            return _extract_image_text(img_bytes, mt_map.get(ext, "image/png"))

        else:
            img_path = os.path.join(base_dir, src)
            if os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
                ext = os.path.splitext(src)[1].lower().lstrip(".")
                return _extract_image_text(img_bytes, mt_map.get(ext, "image/png"))

    except Exception as e:
        logger.warning(f"HTML image resolution failed for '{src}': {e}")
    return ""


def _walk_html(node: Any, base_dir: str, parts: List[str]) -> None:
    """
    Walk the HTML DOM in document order.
    Text and images collected as encountered — no separate passes.
    """
    if isinstance(node, NavigableString):
        text = str(node).strip()
        if text:
            parts.append(text)
    elif isinstance(node, Tag):
        if node.name == "img":
            src = node.get("src", "").strip()
            if src:
                img_text = _resolve_html_image(src, base_dir)
                if img_text.strip():
                    parts.append(f"[Image]\n{img_text}")
        else:
            for child in node.children:
                _walk_html(child, base_dir, parts)
            if node.name in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
                             "li", "tr", "br", "section", "article"):
                parts.append("")


def parse_html(file_path: str) -> List[Dict[str, Any]]:
    """HTML — DOM-order traversal. One walk, text and images interleaved."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    base_dir = os.path.dirname(os.path.abspath(file_path))
    parts: List[str] = []
    _walk_html(soup, base_dir, parts)

    text = "\n".join(line for line in "\n".join(parts).splitlines() if line.strip())
    return [{"page": 1, "text": text}] if text else []


# ── Standalone image ──────────────────────────────────────────────────────────


def parse_image(file_path: str) -> List[Dict[str, Any]]:
    """Standalone image file — sent directly to Claude Vision."""
    mt_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp",
    }
    ext = os.path.splitext(file_path)[1].lower()
    with open(file_path, "rb") as f:
        image_bytes = f.read()
    text = _extract_image_text(image_bytes, mt_map.get(ext, "image/png"))
    return [{"page": 1, "text": text}]


# ── Email ─────────────────────────────────────────────────────────────────────


def parse_email(file_path: str) -> List[Dict[str, Any]]:
    """
    .eml — body text + attachments parsed recursively through parse_document().
    Any attachment type is handled automatically without format-specific logic here.
    """
    with open(file_path, "rb") as f:
        msg = email_lib.message_from_binary_file(f, policy=policy.default)

    header = (
        f"From: {msg.get('from', '')}\n"
        f"Date: {msg.get('date', '')}\n"
        f"Subject: {msg.get('subject', '')}\n\n"
    )

    body_parts: List[str] = []
    attachment_pages: List[Dict[str, Any]] = []

    for part in msg.walk():
        content_type = part.get_content_type()
        disposition = str(part.get_content_disposition() or "")

        if "attachment" in disposition:
            filename = part.get_filename()
            if not filename:
                continue
            ext = os.path.splitext(filename)[1].lower()
            data = part.get_payload(decode=True)
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            try:
                for ap in parse_document(tmp_path):
                    ap["source_attachment"] = filename
                    attachment_pages.append(ap)
            except Exception as e:
                logger.warning(f"Could not parse attachment '{filename}': {e}")
            finally:
                os.unlink(tmp_path)

        elif content_type == "text/plain":
            body_parts.append(part.get_content())
        elif content_type == "text/html":
            soup = BeautifulSoup(part.get_content(), "html.parser")
            body_parts.append(soup.get_text(separator="\n"))

    body = header + "\n".join(body_parts)
    return [{"page": 1, "text": body}] + attachment_pages


# ── Router ────────────────────────────────────────────────────────────────────


def parse_document(file_path: str) -> List[Dict[str, Any]]:
    """
    Single entry point. Detects format by extension, routes to specialist.
    Adding a new format = write one function + add one line in parsers dict.
    """
    ext = os.path.splitext(file_path)[1].lower()
    parsers = {
        ".pdf":  parse_pdf,
        ".docx": parse_docx,
        ".xlsx": parse_excel,
        ".xls":  parse_excel,
        ".csv":  parse_csv,
        ".html": parse_html,
        ".htm":  parse_html,
        ".png":  parse_image,
        ".jpg":  parse_image,
        ".jpeg": parse_image,
        ".gif":  parse_image,
        ".webp": parse_image,
        ".eml":  parse_email,
    }
    parser_fn = parsers.get(ext)
    if not parser_fn:
        raise ValueError(f"Unsupported format '{ext}'. Supported: {sorted(parsers.keys())}")
    return parser_fn(file_path)
