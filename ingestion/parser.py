"""
Parser — extracts clean text from any document format.

All parsers return List[{"page": int, "text": str}].

Content is extracted in reading order — the exact order it visually appears
on the page. Text, tables, and images are interleaved so the LLM receives
full context. "As shown in the chart above" stays next to the chart it
references, not separated by several paragraphs of unrelated text.

How each format achieves reading order:
  PDF    — elements (text blocks, tables, images) sorted by vertical y-position
  DOCX   — body XML iterated in document order; inline images handled inside
            the paragraph they belong to
  HTML   — DOM tree walked in traversal order
  Excel  — cell rows in sheet order; images inserted at their anchor row
           (parsed from drawing XML inside the xlsx ZIP archive)
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


def _bbox_overlaps(b1: Tuple, b2: Tuple) -> bool:
    """True if two bounding boxes (x0,y0,x1,y1) overlap."""
    return not (b1[2] <= b2[0] or b1[0] >= b2[2] or b1[3] <= b2[1] or b1[1] >= b2[3])


def _media_type(ext: str) -> str:
    return "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"


# ── PDF ───────────────────────────────────────────────────────────────────────


def parse_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    PDF — reading-order extraction per page.

    Collects text blocks, tables, and images each with their bounding box
    y-coordinate, then sorts everything by vertical position before merging.
    Text blocks that overlap with detected table regions are excluded to avoid
    duplicating table content (PyMuPDF returns table cells as text blocks too).

    Result: content exactly as a human would read the page top to bottom.
    """
    doc = fitz.open(file_path)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        elements: List[Dict] = []

        # Detect tables first so we can exclude overlapping text blocks
        tables = list(page.find_tables())
        table_bboxes = [t.bbox for t in tables]

        # Text blocks — skip blocks whose area overlaps a table region
        for block in page.get_text("blocks"):
            x0, y0, x1, y1, text, _, block_type = block
            if block_type != 0 or not text.strip():
                continue
            block_bbox = (x0, y0, x1, y1)
            if any(_bbox_overlaps(block_bbox, tb) for tb in table_bboxes):
                continue
            elements.append({"y": y0, "content": text.strip()})

        # Tables — use their top y-coordinate for ordering
        for table in tables:
            rows = [[str(cell or "").strip() for cell in row] for row in table.extract()]
            md = _rows_to_markdown(rows)
            if md:
                elements.append({"y": table.bbox[1], "content": f"[Table]\n{md}"})

        # Images — filter out small decorative images (icons, borders < 50px)
        for img_info in page.get_image_info():
            if img_info.get("width", 0) < 50 or img_info.get("height", 0) < 50:
                continue
            xref = img_info["xref"]
            y0 = img_info["bbox"][1]
            try:
                img_data = doc.extract_image(xref)
                img_text = _extract_image_text(
                    img_data["image"], _media_type(img_data["ext"])
                )
                if img_text.strip():
                    elements.append({"y": y0, "content": f"[Image]\n{img_text}"})
            except Exception as e:
                logger.warning(f"PDF page {page_num} image extraction failed: {e}")

        # Sort by vertical position → reading order
        elements.sort(key=lambda e: e["y"])

        if elements:
            pages.append({
                "page": page_num,
                "text": "\n\n".join(e["content"] for e in elements),
            })

    doc.close()
    return pages


# ── DOCX ──────────────────────────────────────────────────────────────────────


def parse_docx(file_path: str) -> List[Dict[str, Any]]:
    """
    DOCX — reading-order extraction via XML body iteration.

    The DOCX body is a sequence of paragraph (<w:p>) and table (<w:tbl>)
    elements in document order. Images in DOCX are inline — they live inside
    paragraphs as <w:drawing> elements. By walking the body XML in order and
    handling images when we encounter the paragraph that contains them, we
    preserve the exact document flow.

    DOCX has no stored page numbers (Word calculates them at render time from
    font/margin settings). Everything returns as page 1.
    """
    doc = Document(file_path)
    parts: List[str] = []

    for element in doc.element.body:
        tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

        if tag == "p":
            para = DocxParagraph(element, doc)
            if para.text.strip():
                parts.append(para.text.strip())

            # Inline images live inside paragraphs — handle them here, in order
            for drawing in element.findall(".//" + qn("w:drawing")):
                for blip in drawing.findall(".//" + qn("a:blip")):
                    r_id = blip.get(qn("r:embed"))
                    if r_id and r_id in doc.part.rels:
                        rel = doc.part.rels[r_id]
                        if "image" in rel.reltype:
                            try:
                                ext = rel.target_ref.split(".")[-1].lower()
                                img_text = _extract_image_text(
                                    rel.target_part.blob, _media_type(ext)
                                )
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

    return [{"page": 1, "text": "\n\n".join(parts)}] if parts else []


# ── Excel ─────────────────────────────────────────────────────────────────────


def _excel_image_row_positions(file_path: str) -> List[Tuple[int, bytes, str]]:
    """
    Parse xlsx drawing XML to find each image's row anchor position.

    Returns list of (row_index, image_bytes, media_type).
    row_index is 0-based — the spreadsheet row the image is anchored to.

    xlsx is a ZIP. Image positions are in xl/drawings/drawing*.xml.
    Image files are in xl/media/. Drawing rels map rId → image file.
    """
    results: List[Tuple[int, bytes, str]] = []
    ns = {
        "xdr": "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing",
        "a":   "http://schemas.openxmlformats.org/drawingml/2006/main",
        "r":   "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }
    r_embed = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"

    with zipfile.ZipFile(file_path, "r") as z:
        all_files = set(z.namelist())

        for i in range(1, 20):
            drawing_path = f"xl/drawings/drawing{i}.xml"
            rels_path = f"xl/drawings/_rels/drawing{i}.xml.rels"
            if drawing_path not in all_files:
                break

            # Map rId → image file path inside the archive
            r_id_to_file: Dict[str, str] = {}
            if rels_path in all_files:
                rels_root = ET.fromstring(z.read(rels_path))
                for rel in rels_root:
                    r_id = rel.get("Id", "")
                    target = rel.get("Target", "")
                    if "../media/" in target:
                        img_file = "xl/media/" + target.split("../media/")[-1]
                        r_id_to_file[r_id] = img_file

            # Parse drawing XML for anchor positions
            draw_root = ET.fromstring(z.read(drawing_path))
            anchor_tags = (
                draw_root.findall(".//xdr:twoCellAnchor", ns) +
                draw_root.findall(".//xdr:oneCellAnchor", ns)
            )

            for anchor in anchor_tags:
                row = 0
                from_elem = anchor.find("xdr:from", ns)
                if from_elem is not None:
                    row_elem = from_elem.find("xdr:row", ns)
                    if row_elem is not None and row_elem.text:
                        row = int(row_elem.text)

                for blip in anchor.findall(".//a:blip", ns):
                    r_id = blip.get(r_embed, "")
                    img_file = r_id_to_file.get(r_id, "")
                    if img_file and img_file in all_files:
                        img_bytes = z.read(img_file)
                        ext = os.path.splitext(img_file)[1].lower().lstrip(".")
                        results.append((row, img_bytes, _media_type(ext)))

    return results


def parse_excel(file_path: str) -> List[Dict[str, Any]]:
    """
    Excel — cell values as markdown table with images inserted at their
    anchor row position.

    Each sheet = one page. Images are extracted from the xlsx ZIP archive,
    their row anchor parsed from drawing XML, and inserted into the table
    output immediately after the row they visually appear next to.
    .xls (legacy binary format) — cell values only, no image extraction.
    """
    xl = pd.ExcelFile(file_path)
    pages = []

    # Get image positions once (applies across sheets for now)
    # ponytail: cross-sheet image attribution requires sheet→drawing mapping
    # via xl/workbook.xml.rels — current approach loads all images from all
    # drawings and anchors them by row only, not by sheet.
    image_positions: List[Tuple[int, bytes, str]] = []
    if file_path.lower().endswith(".xlsx"):
        try:
            image_positions = _excel_image_row_positions(file_path)
            image_positions.sort(key=lambda x: x[0])
        except Exception as e:
            logger.warning(f"Excel image position parsing failed: {e}")

    for sheet_num, sheet_name in enumerate(xl.sheet_names, start=1):
        df = xl.parse(sheet_name)
        parts: List[str] = [f"Sheet: {sheet_name}"]

        if not df.empty:
            # Build rows list for interleaving
            md_lines = df.to_markdown(index=False).split("\n")
            # md_lines[0] = header row, [1] = separator, [2+] = data rows
            result_lines = md_lines[:2]

            for data_row_idx, line in enumerate(md_lines[2:]):
                result_lines.append(line)
                # Insert images anchored at this row (0-based row index)
                for row_anchor, img_bytes, media_type in image_positions:
                    if row_anchor == data_row_idx:
                        try:
                            img_text = _extract_image_text(img_bytes, media_type)
                            if img_text.strip():
                                result_lines.append(f"\n[Image at row {data_row_idx + 1}]\n{img_text}")
                        except Exception as e:
                            logger.warning(f"Excel image extraction failed: {e}")

            parts.append("\n".join(result_lines))

            # Append any images beyond the last data row
            last_row = len(md_lines) - 3
            for row_anchor, img_bytes, media_type in image_positions:
                if row_anchor > last_row:
                    try:
                        img_text = _extract_image_text(img_bytes, media_type)
                        if img_text.strip():
                            parts.append(f"[Image]\n{img_text}")
                    except Exception as e:
                        logger.warning(f"Excel image extraction failed: {e}")

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
    """Resolve any img src type (base64, relative path, remote URL) → text."""
    media_type_map = {
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
            return _extract_image_text(img_bytes, media_type_map.get(ext, "image/png"))

        else:
            img_path = os.path.join(base_dir, src)
            if os.path.exists(img_path):
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
                ext = os.path.splitext(src)[1].lower().lstrip(".")
                return _extract_image_text(img_bytes, media_type_map.get(ext, "image/png"))
    except Exception as e:
        logger.warning(f"HTML image resolution failed for '{src}': {e}")
    return ""


def _walk_html(node: Any, base_dir: str, parts: List[str]) -> None:
    """
    Recursively walk the HTML DOM in document order.

    Text nodes and <img> elements are collected as they appear — no separate
    passes. An image between two paragraphs appears between those paragraphs
    in the output, preserving the visual reading flow.
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
            # Add paragraph break after block-level elements
            if node.name in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
                             "li", "tr", "br", "section", "article"):
                parts.append("")


def parse_html(file_path: str) -> List[Dict[str, Any]]:
    """
    HTML — DOM-order traversal.

    Walks the HTML tree once, collecting text and images in the exact order
    they appear in the document. No separate passes.
    """
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
    media_type_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp",
    }
    ext = os.path.splitext(file_path)[1].lower()
    with open(file_path, "rb") as f:
        image_bytes = f.read()
    text = _extract_image_text(image_bytes, media_type_map.get(ext, "image/png"))
    return [{"page": 1, "text": text}]


# ── Email ─────────────────────────────────────────────────────────────────────


def parse_email(file_path: str) -> List[Dict[str, Any]]:
    """
    .eml file — body text + attachments parsed recursively.

    Attachments are routed through parse_document() so any attachment type
    is handled automatically without additional logic here.
    """
    with open(file_path, "rb") as f:
        msg = email_lib.message_from_binary_file(f, policy=policy.default)

    subject = msg.get("subject", "")
    sender = msg.get("from", "")
    date = msg.get("date", "")
    header = f"From: {sender}\nDate: {date}\nSubject: {subject}\n\n"

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
    Adding a new format = write one function + add one line here.
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
        raise ValueError(
            f"Unsupported format '{ext}'. Supported: {sorted(parsers.keys())}"
        )
    return parser_fn(file_path)
