"""
Parser — extracts clean text from any document format.

All parsers return List[{"page": int, "text": str}].

Content is extracted in reading order — the exact order it visually appears
on the page. Text, tables, and images are interleaved so the LLM receives
full context ("as shown in the chart above" stays next to the chart).

Reading-order strategy per format:
  PDF    — elements sorted by y-coordinate; N-column layout detected from
            x-center gaps and handled correctly for any column count; scanned
            pages rendered as full-page bitmap to Vision; duplicate image
            xrefs deduplicated per page
  DOCX   — body XML iterated in document order; inline images, text boxes,
            and embedded charts handled inside their parent paragraph;
            headers/footers extracted from all sections
  HTML   — DOM tree walked in traversal order; tables converted to markdown;
            images resolved from base64, relative path, or remote URL
  Excel  — merged cells forward-filled; drawing XML parsed for images and
            native charts; content inserted at anchor row positions
  Email  — cid: inline images in HTML body resolved from MIME parts;
            attachments parsed recursively through router
  TIFF   — multi-frame support; each frame converted to PNG via Pillow
  CSV    — delimiter and encoding auto-detected

Known limitation:
  PDF vector graphics — charts drawn as mathematical path instructions are
  invisible to text extraction and image extraction. No clean fix without
  rendering every page as a bitmap (expensive). Flagged, not fixed.
"""

import base64
import email as email_lib
import io
import logging
import os
import re
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from email import policy
from typing import Any, Dict, List, Tuple

import threading

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

# Vision client singleton — one client, reused across all image extractions.
# Avoids creating a new connection pool per image (50 images = 50 pool setups
# without this). Timeout kills hung Vision calls in 30s instead of 600s default.
_vision_client    = None
_vision_client_lock = threading.Lock()
_VISION_TIMEOUT   = 30.0


def _get_vision_client():
    global _vision_client
    if _vision_client is None:
        with _vision_client_lock:
            if _vision_client is None:
                import anthropic
                key = os.getenv("ANTHROPIC_API_KEY")
                if not key:
                    raise ValueError("ANTHROPIC_API_KEY not set — required for image parsing")
                _vision_client = anthropic.Anthropic(api_key=key, timeout=_VISION_TIMEOUT)
    return _vision_client

# DOCX namespace constants for chart and text box extraction
_DOCX_CHART_ELEM = "{http://schemas.openxmlformats.org/drawingml/2006/chart}chart"
_DOCX_R_ID_ATTR = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


# ── Shared helpers ────────────────────────────────────────────────────────────


def _extract_image_text(image_bytes: bytes, media_type: str = "image/png") -> str:
    """
    Send image bytes to Claude Vision, return extracted text.

    Claude understands layout — tables in images, charts, multi-column text —
    where traditional OCR (Tesseract) fails on complex financial documents.

    Uses the module-level singleton client (_get_vision_client) so that one
    connection pool is reused across all images in a document rather than
    creating a new one per image. Times out in _VISION_TIMEOUT seconds.
    """
    client = _get_vision_client()
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
    Return elements in correct visual reading order for N-column layouts.

    Column count is detected from gaps in the x-center distribution of text
    blocks. The gutter threshold is adaptive: max(2.5 × median gap between
    x-centers, 3% of page width). This catches tight gutters (5–6% of page
    width) that a fixed 8% threshold would miss, while the 3% floor prevents
    false splits from within-column indentation noise. Works for any column count.

    Full-width elements (section headers, wide tables spanning ≥60% of page)
    interrupt column flow: everything above them is emitted column-by-column
    first, then the full-width element, then the content below it.

    Within each column band, content is ordered top-to-bottom per column,
    columns ordered left-to-right.
    """
    if not elements:
        return []

    def cx(e: Dict) -> float:
        return (e.get("x0", 0) + e.get("x1", e.get("x0", 0))) / 2

    def el_width(e: Dict) -> float:
        return e.get("x1", e.get("x0", 0)) - e.get("x0", 0)

    text_els = [e for e in elements if e.get("etype") == "text"]
    if not text_els:
        return sorted(elements, key=lambda e: e["y"])

    # Adaptive column-gutter detection.
    # Within a column, x-centers cluster tightly. Gutters are outlier gaps.
    # Threshold = max(2.5 × median gap, 3% of page width).
    # The median factor adapts to the document's own spacing so tight gutters
    # (5–6% of page width) are caught. The 3% floor prevents false splits
    # from normal within-column indentation variation.
    centers = sorted(cx(e) for e in text_els)
    gaps = [centers[i] - centers[i - 1] for i in range(1, len(centers))]
    if len(gaps) >= 2:
        median_gap = sorted(gaps)[len(gaps) // 2]
        gap_threshold = max(page_width * 0.03, median_gap * 2.5)
    else:
        gap_threshold = page_width * 0.08  # ponytail: fallback for <2 gaps (≤2 text blocks)
    boundaries = [0.0]
    for i in range(1, len(centers)):
        if centers[i] - centers[i - 1] > gap_threshold:
            boundaries.append((centers[i - 1] + centers[i]) / 2)
    boundaries.append(page_width)

    num_cols = len(boundaries) - 1

    # Need at least 2 blocks per detected column to trust the detection
    if num_cols == 1 or len(text_els) < num_cols * 2:
        return sorted(elements, key=lambda e: e["y"])

    # Classify each element: full-width (spans ≥60% of page) or column-specific
    full_w_threshold = page_width * 0.6

    def col_of(e: Dict) -> int:
        if el_width(e) >= full_w_threshold:
            return -1  # full-width sentinel
        c = cx(e)
        for i in range(num_cols):
            if boundaries[i] <= c < boundaries[i + 1]:
                return i
        return num_cols - 1

    full_width_els = sorted([e for e in elements if col_of(e) == -1], key=lambda e: e["y"])
    column_els = [e for e in elements if col_of(e) != -1]

    cols: List[List[Dict]] = [[] for _ in range(num_cols)]
    for e in column_els:
        cols[col_of(e)].append(e)
    for col in cols:
        col.sort(key=lambda e: e["y"])

    # Emit all columns (left→right) for elements whose y falls within a band
    result: List[Dict] = []

    def emit_band(y_lo: float, y_hi: float) -> None:
        for col in cols:
            for e in col:
                if y_lo <= e["y"] < y_hi:
                    result.append(e)

    prev_y = float("-inf")
    for fw_el in full_width_els:
        emit_band(prev_y, fw_el["y"])
        result.append(fw_el)
        prev_y = fw_el["y"]

    emit_band(prev_y, float("inf"))
    return result


# ── PDF ───────────────────────────────────────────────────────────────────────


def parse_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    PDF — reading-order extraction per page.

    - Text blocks, tables, images collected with x/y positions and sorted.
    - Text blocks overlapping table regions excluded to avoid duplication.
    - Images < 50×50px skipped (decorative icons, borders).
    - Same image xref processed only once per page (deduplication).
    - Password-protected PDFs raise a clear ValueError.
    - Scanned pages (zero content extracted) rendered as full-page bitmap → Vision.
    - N-column layouts handled: column count auto-detected, any count supported.

    Known limitation: vector graphics (charts drawn as PDF path instructions)
    are invisible. No fix without rendering every page — too expensive.
    """
    doc = fitz.open(file_path)

    if doc.is_encrypted:
        doc.close()
        raise ValueError(
            f"PDF is password-protected and cannot be parsed: {file_path}. "
            "Decrypt it before ingestion."
        )

    logger.info(f"[{file_path}] parsing PDF — {doc.page_count} pages")
    pages = []

    for page_num, page in enumerate(doc, start=1):
        elements: List[Dict] = []

        tables = list(page.find_tables())
        table_bboxes = [t.bbox for t in tables]

        # Text blocks — exclude regions that belong to detected tables
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

        # Images — deduplicated by xref so the same image isn't sent to Vision twice
        seen_xrefs: set = set()
        for img_info in page.get_image_info():
            if img_info.get("width", 0) < 50 or img_info.get("height", 0) < 50:
                continue
            xref = img_info["xref"]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            b = img_info["bbox"]
            try:
                img_data = doc.extract_image(xref)
                img_text = _extract_image_text(img_data["image"], _media_type(img_data["ext"]))
                if img_text.strip():
                    elements.append({"etype": "image", "y": b[1], "x0": b[0], "x1": b[2], "content": f"[Image]\n{img_text}"})
            except Exception as e:
                logger.warning(f"[{file_path}] page {page_num} image extraction failed: {type(e).__name__}: {e}")

        # Scanned page fallback — if zero content extracted, the page has no text
        # layer. Render it as a full-page bitmap and send to Claude Vision.
        if not elements:
            try:
                pix = page.get_pixmap(dpi=150)
                img_text = _extract_image_text(pix.tobytes("png"), "image/png")
                if img_text.strip():
                    elements.append({"etype": "image", "y": 0, "x0": 0, "x1": page.rect.width, "content": f"[Scanned page]\n{img_text}"})
            except Exception as e:
                logger.warning(f"[{file_path}] page {page_num} scanned-page render failed: {type(e).__name__}: {e}")

        ordered = _reading_order(elements, page.rect.width)
        if ordered:
            pages.append({"page": page_num, "text": "\n\n".join(e["content"] for e in ordered)})

    doc.close()
    logger.info(f"[{file_path}] parsed: {len(pages)} pages with content")
    return pages


# ── DOCX ──────────────────────────────────────────────────────────────────────


def _docx_header_footer_text(hf_obj) -> str:
    """Extract plain text from a DOCX header or footer object."""
    if hf_obj.is_linked_to_previous:
        return ""
    return "\n".join(p.text.strip() for p in hf_obj.paragraphs if p.text.strip())


def _docx_extract_drawing(
    drawing_elem, doc: Document, parts: List[str], file_path: str
) -> None:
    """
    Extract all content from a DOCX drawing element — images, text boxes, charts.

    Called for every <w:drawing> found in a paragraph, preserving the order
    in which these elements appear within the paragraph.
    """
    # Images — referenced via a:blip
    for blip in drawing_elem.findall(".//" + qn("a:blip")):
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
                    logger.warning(f"[{file_path}] DOCX image extraction failed: {type(e).__name__}: {e}")

    # Text boxes — floating callout boxes, sidebars, highlighted stat boxes
    # stored as w:txbxContent inside wps:txbx inside the drawing
    for txbx in drawing_elem.findall(".//" + qn("w:txbxContent")):
        txbx_lines = []
        for p_elem in txbx.findall(".//" + qn("w:p")):
            try:
                para = DocxParagraph(p_elem, doc)
                if para.text.strip():
                    txbx_lines.append(para.text.strip())
            except Exception as e:
                logger.debug(f"[{file_path}] DOCX text box paragraph skipped: {type(e).__name__}: {e}")
        if txbx_lines:
            parts.append("[Text box]\n" + "\n".join(txbx_lines))

    # Embedded charts — Excel charts embedded in Word documents
    # referenced via c:chart element with r:id pointing to chart part
    for chart_elem in drawing_elem.findall(f".//{_DOCX_CHART_ELEM}"):
        r_id = chart_elem.get(_DOCX_R_ID_ATTR)
        if r_id and r_id in doc.part.rels:
            rel = doc.part.rels[r_id]
            if "chart" in rel.reltype:
                try:
                    chart_text = _parse_chart_xml(rel.target_part.blob)
                    if chart_text.strip():
                        parts.append(f"[Chart]\n{chart_text}")
                except Exception as e:
                    logger.warning(f"[{file_path}] DOCX chart extraction failed: {type(e).__name__}: {e}")


def parse_docx(file_path: str) -> List[Dict[str, Any]]:
    """
    DOCX — reading-order extraction via body XML iteration.

    Extracts in document order:
    - Headers (all sections)
    - Body: paragraphs and tables interleaved as they appear in the XML
      - Each paragraph: text + any inline images, text boxes, or charts
    - Footers (all sections)

    DOCX has no stored page numbers. Everything returns as page 1.
    """
    doc = Document(file_path)
    parts: List[str] = []

    # Headers — all sections (some financial docs have per-chapter headers)
    seen_headers: set = set()
    for section in doc.sections:
        text = _docx_header_footer_text(section.header)
        if text and text not in seen_headers:
            parts.append(f"[Header]\n{text}")
            seen_headers.add(text)

    # Body — iterate XML elements in document order
    for element in doc.element.body:
        tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

        if tag == "p":
            para = DocxParagraph(element, doc)
            if para.text.strip():
                parts.append(para.text.strip())

            # All drawing content (images, text boxes, charts) in this paragraph
            for drawing in element.findall(".//" + qn("w:drawing")):
                _docx_extract_drawing(drawing, doc, parts, file_path)

        elif tag == "tbl":
            table = DocxTable(element, doc)
            rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
            md = _rows_to_markdown(rows)
            if md:
                parts.append(f"[Table]\n{md}")

    # Footers — all sections, deduplicated
    seen_footers: set = set()
    for section in doc.sections:
        text = _docx_header_footer_text(section.footer)
        if text and text not in seen_footers:
            parts.append(f"[Footer]\n{text}")
            seen_footers.add(text)

    return [{"page": 1, "text": "\n\n".join(parts)}] if parts else []


# ── Excel ─────────────────────────────────────────────────────────────────────


def _parse_chart_xml(chart_bytes: bytes) -> str:
    """
    Extract data from an Excel/DOCX chart XML.

    Charts cache their data in XML on save. Extracts title, chart type,
    and each series with category labels and values as markdown tables.
    """
    ns = {
        "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    }

    root = ET.fromstring(chart_bytes)
    parts: List[str] = []

    title_parts = (
        [t.text.strip() for t in root.findall(".//c:title//a:t", ns) if t.text] +
        [t.text.strip() for t in root.findall(".//c:title//c:v", ns) if t.text]
    )
    if title_parts:
        parts.append(f"Chart: {' '.join(title_parts)}")

    chart_type_tags = ["barChart", "lineChart", "pieChart", "scatterChart",
                       "areaChart", "doughnutChart", "radarChart", "bubbleChart"]
    for ct in chart_type_tags:
        if root.find(f".//c:{ct}", ns) is not None:
            if not title_parts:
                parts.append(ct.replace("Chart", " Chart").title())
            break

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
    Extract all embedded content (images and native charts) from xlsx
    with their row anchor positions.

    Returns: List of (row_index, text_content) sorted by row_index.
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
                                    logger.warning(f"[{file_path}] Excel image extraction failed: {type(e).__name__}: {e}")

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
                                    logger.warning(f"[{file_path}] Excel chart extraction failed: {type(e).__name__}: {e}")

    results.sort(key=lambda x: x[0])
    return results


def parse_excel(file_path: str) -> List[Dict[str, Any]]:
    """
    Excel — cell values as markdown table with images and charts at anchor rows.

    Merged cells are forward-filled so NaN does not appear where a merged
    header logically covers multiple rows or columns.
    Password-protected files raise a clear ValueError.
    """
    try:
        xl = pd.ExcelFile(file_path)
    except Exception as e:
        err = str(e).lower()
        if any(k in err for k in ("password", "encrypted", "xlrd", "openpyxl")):
            raise ValueError(
                f"Excel file appears to be password-protected or encrypted: {file_path}"
            ) from e
        raise

    pages = []

    drawing_content: List[Tuple[int, str]] = []
    if file_path.lower().endswith(".xlsx"):
        try:
            drawing_content = _excel_drawing_content(file_path)
        except Exception as e:
            logger.warning(f"[{file_path}] Excel drawing content extraction failed — images and charts may be missing: {type(e).__name__}: {e}")

    for sheet_num, sheet_name in enumerate(xl.sheet_names, start=1):
        df = xl.parse(sheet_name)
        parts: List[str] = [f"Sheet: {sheet_name}"]

        if not df.empty:
            # Forward-fill merged cell areas (NaN in merged regions → cell value)
            df = df.ffill(axis=0).ffill(axis=1).fillna("")

            md_lines = df.to_markdown(index=False).split("\n")
            result_lines = md_lines[:2]

            for data_row_idx, line in enumerate(md_lines[2:]):
                result_lines.append(line)
                for row_anchor, embedded_text in drawing_content:
                    if row_anchor == data_row_idx:
                        result_lines.append(f"\n{embedded_text}")

            parts.append("\n".join(result_lines))

            last_row = len(md_lines) - 3
            for row_anchor, embedded_text in drawing_content:
                if row_anchor > last_row:
                    parts.append(embedded_text)

        if len(parts) > 1:
            pages.append({"page": sheet_num, "text": "\n\n".join(parts)})

    return pages


# ── CSV ───────────────────────────────────────────────────────────────────────


def parse_csv(file_path: str) -> List[Dict[str, Any]]:
    """
    CSV — auto-detects delimiter (comma, tab, semicolon, pipe) and encoding.
    Falls back from UTF-8 to Latin-1 on decode errors.
    """
    try:
        df = pd.read_csv(file_path, sep=None, engine="python", encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, sep=None, engine="python", encoding="latin-1")
    return [{"page": 1, "text": df.to_markdown(index=False)}]


# ── HTML ──────────────────────────────────────────────────────────────────────


def _resolve_html_image(src: str, base_dir: str) -> str:
    """Resolve img src (base64 data URI / relative path / remote URL) → Vision text."""
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
            if base_dir:
                img_path = os.path.join(base_dir, src)
                if os.path.exists(img_path):
                    with open(img_path, "rb") as f:
                        img_bytes = f.read()
                    ext = os.path.splitext(src)[1].lower().lstrip(".")
                    return _extract_image_text(img_bytes, mt_map.get(ext, "image/png"))

    except Exception as e:
        logger.warning(f"HTML image resolution failed for src='{src}': {type(e).__name__}: {e}")
    return ""


def _walk_html(node: Any, base_dir: str, parts: List[str]) -> None:
    """
    Walk the HTML DOM in document order.
    Text, images, and tables collected as encountered — no separate passes.
    Tables converted to markdown to preserve column relationships.
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

        elif node.name == "table":
            # Convert HTML table to markdown — walking as text loses structure
            rows = []
            for tr in node.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if cells:
                    rows.append(cells)
            if rows:
                md = _rows_to_markdown(rows)
                if md:
                    parts.append(f"[Table]\n{md}")

        else:
            for child in node.children:
                _walk_html(child, base_dir, parts)
            if node.name in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
                             "li", "tr", "br", "section", "article"):
                parts.append("")


def parse_html(file_path: str) -> List[Dict[str, Any]]:
    """HTML — one DOM walk, text/tables/images interleaved in document order."""
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


# ── TIFF ──────────────────────────────────────────────────────────────────────


def parse_tiff(file_path: str) -> List[Dict[str, Any]]:
    """
    TIFF — multi-page scanned document format common in financial document scanning.

    Claude Vision does not accept TIFF natively. Each frame is converted to
    PNG via Pillow and then sent to Vision. Multi-frame TIFFs (document batches
    scanned as a single file) have each frame extracted as a separate page.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "Pillow is required for TIFF parsing: pip install pillow"
        )

    pages = []
    with Image.open(file_path) as img:
        frame = 0
        while True:
            try:
                img.seek(frame)
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="PNG")
                img_text = _extract_image_text(buf.getvalue(), "image/png")
                if img_text.strip():
                    pages.append({"page": frame + 1, "text": img_text})
                frame += 1
            except EOFError:
                break

    return pages


# ── Email ─────────────────────────────────────────────────────────────────────


def _walk_html_email(
    node: Any,
    cid_images: Dict[str, Tuple[bytes, str]],
    parts: List[str],
    file_path: str = "",
) -> None:
    """
    Walk HTML email body in DOM order, resolving cid: inline image references.

    Corporate HTML emails embed images as MIME parts referenced via
    cid: scheme (<img src="cid:uniqueid@domain">). This walker matches
    those references to the collected MIME parts and sends them to Vision.

    file_path is passed for logging only — previously missing, which caused
    NameError: 'file_path' is not defined when a cid image failed to extract.
    The NameError would crash the entire email parse with a misleading message.
    """
    if isinstance(node, NavigableString):
        text = str(node).strip()
        if text:
            parts.append(text)
    elif isinstance(node, Tag):
        if node.name == "img":
            src = node.get("src", "").strip()
            if src.startswith("cid:"):
                cid = src[4:]  # strip "cid:" prefix
                if cid in cid_images:
                    img_bytes, media_type = cid_images[cid]
                    try:
                        img_text = _extract_image_text(img_bytes, media_type)
                        if img_text.strip():
                            parts.append(f"[Image]\n{img_text}")
                    except Exception as e:
                        logger.warning(f"[{file_path}] email cid image extraction failed for cid='{cid}': {type(e).__name__}: {e}")
            elif src:
                img_text = _resolve_html_image(src, "")
                if img_text.strip():
                    parts.append(f"[Image]\n{img_text}")

        elif node.name == "table":
            rows = []
            for tr in node.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if cells:
                    rows.append(cells)
            if rows:
                md = _rows_to_markdown(rows)
                if md:
                    parts.append(f"[Table]\n{md}")

        else:
            for child in node.children:
                _walk_html_email(child, cid_images, parts, file_path)
            if node.name in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
                             "li", "tr", "br", "section", "article"):
                parts.append("")


def parse_email(file_path: str) -> List[Dict[str, Any]]:
    """
    .eml — body + attachments parsed recursively through parse_document().

    cid: inline images in HTML email bodies are resolved by first collecting
    all MIME parts that have a Content-ID header, then matching cid: references
    in the HTML body to those parts during DOM traversal.
    """
    with open(file_path, "rb") as f:
        msg = email_lib.message_from_binary_file(f, policy=policy.default)

    header = (
        f"From: {msg.get('from', '')}\n"
        f"Date: {msg.get('date', '')}\n"
        f"Subject: {msg.get('subject', '')}\n\n"
    )

    # Collect all inline images by Content-ID before processing body
    cid_images: Dict[str, Tuple[bytes, str]] = {}
    for part in msg.walk():
        content_id = part.get("Content-ID", "")
        if content_id and part.get_content_maintype() == "image":
            cid = content_id.strip("<>")
            img_data = part.get_payload(decode=True)
            if img_data:
                cid_images[cid] = (img_data, part.get_content_type())

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
                logger.warning(f"[{file_path}] attachment '{filename}' failed to parse: {type(e).__name__}: {e}")
            finally:
                os.unlink(tmp_path)

        elif content_type == "text/plain":
            body_parts.append(part.get_content())

        elif content_type == "text/html":
            soup = BeautifulSoup(part.get_content(), "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            email_parts: List[str] = []
            _walk_html_email(soup, cid_images, email_parts, file_path)
            text = "\n".join(
                line for line in "\n".join(email_parts).splitlines() if line.strip()
            )
            if text:
                body_parts.append(text)

    body = header + "\n".join(body_parts)
    return [{"page": 1, "text": body}] + attachment_pages


# ── Router ────────────────────────────────────────────────────────────────────


def parse_document(file_path: str) -> List[Dict[str, Any]]:
    """
    Single entry point. Detects format by extension, routes to specialist.
    Adding a new format = write one function + add one line in parsers dict.

    Logs entry, success, and failure with full file context so any error
    in the pipeline can be traced to the exact file and stage that caused it.
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
        ".tif":  parse_tiff,
        ".tiff": parse_tiff,
        ".eml":  parse_email,
    }
    parser_fn = parsers.get(ext)
    if not parser_fn:
        raise ValueError(f"Unsupported format '{ext}'. Supported: {sorted(parsers.keys())}")

    try:
        result = parser_fn(file_path)
        if not result:
            logger.warning(
                f"[{file_path}] PARSE WARNING: parser returned 0 pages — "
                "file may be empty, corrupted, or contain only unsupported content"
            )
        else:
            logger.info(
                f"[{file_path}] PARSE SUCCESS: {len(result)} pages, "
                f"{sum(len(p['text'].split()) for p in result)} words extracted"
            )
        return result
    except Exception as e:
        logger.error(
            f"[{file_path}] PARSE FAILED at stage=parser format={ext}: "
            f"{type(e).__name__}: {e}"
        )
        raise
