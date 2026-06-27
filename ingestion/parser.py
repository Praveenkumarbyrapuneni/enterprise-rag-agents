"""
Parser — extracts clean text from any document format.

Every parser returns List[Dict] where each dict is:
    {"page": int, "text": str}

This uniform output means chunker, embedder, and Qdrant never need to know
what format the original document was. Add a new format = one function + one
line in parse_document(). Nothing else changes.

PDF and DOCX run 3 passes per page/document:
    Pass 1 — text layer
    Pass 2 — tables → markdown (structure preserved)
    Pass 3 — embedded images → Claude Vision (no context lost)
"""

import base64
import email as email_lib
import logging
import os
import tempfile
from email import policy
from typing import Any, Dict, List

import fitz  # PyMuPDF
import pandas as pd
from bs4 import BeautifulSoup
from docx import Document
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ── Shared helpers ────────────────────────────────────────────────────────────


def _extract_image_text(image_bytes: bytes, media_type: str = "image/png") -> str:
    """
    Send image bytes to Claude Vision, get back extracted text.

    Used for:
    - Standalone image files (.png, .jpg, etc.)
    - Images embedded inside PDFs
    - Images embedded inside DOCX files

    Claude understands layout — tables, charts, multi-column text — where
    traditional OCR (Tesseract) fails on complex financial documents.
    """
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set — required for image/scanned document parsing")

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
    """
    Convert a 2D list of cell strings into a markdown table.

    Why markdown? The LLM in Phase 2 reads markdown tables correctly — it
    understands column relationships. Raw CSV-style text loses that structure.
    """
    if not rows:
        return ""
    header = "| " + " | ".join(str(c) for c in rows[0]) + " |"
    sep = "| " + " | ".join(["---"] * len(rows[0])) + " |"
    body_rows = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows[1:]]
    return "\n".join([header, sep] + body_rows)


# ── Format parsers ────────────────────────────────────────────────────────────


def parse_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    PDF — 3 passes per page:
        Pass 1: text layer  (PyMuPDF get_text)
        Pass 2: tables      (PyMuPDF find_tables → markdown)
        Pass 3: images      (PyMuPDF extract_image → Claude Vision)

    Each page = one dict so citations point to exact page numbers.
    """
    doc = fitz.open(file_path)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        parts: List[str] = []

        # Pass 1 — text layer
        text = page.get_text().strip()
        if text:
            parts.append(text)

        # Pass 2 — tables
        for table in page.find_tables():
            rows = [[str(cell or "").strip() for cell in row] for row in table.extract()]
            md = _rows_to_markdown(rows)
            if md:
                parts.append(f"\n[Table]\n{md}")

        # Pass 3 — embedded images
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                img_info = doc.extract_image(xref)
                ext = img_info["ext"]
                media_type = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
                img_text = _extract_image_text(img_info["image"], media_type)
                if img_text.strip():
                    parts.append(f"\n[Image]\n{img_text}")
            except Exception as e:
                logger.warning(f"PDF page {page_num} image extraction failed: {e}")

        if parts:
            pages.append({"page": page_num, "text": "\n".join(parts)})

    doc.close()
    return pages


def parse_docx(file_path: str) -> List[Dict[str, Any]]:
    """
    DOCX — text + tables + embedded images.

    DOCX has no real page numbers (Word calculates pages at render time based
    on font/margins). Everything returns as page 1. This is the right tradeoff —
    forcing fake page numbers would be less accurate than admitting we don't have them.
    """
    doc = Document(file_path)
    parts: List[str] = []

    # Pass 1 — paragraphs
    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text.strip())

    # Pass 2 — tables
    for table in doc.tables:
        rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        md = _rows_to_markdown(rows)
        if md:
            parts.append(f"\n[Table]\n{md}")

    # Pass 3 — embedded images (stored as relationships in the DOCX zip)
    for rel in doc.part.rels.values():
        if "image" in rel.reltype:
            try:
                img_bytes = rel.target_part.blob
                ext = rel.target_ref.split(".")[-1].lower()
                media_type = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
                img_text = _extract_image_text(img_bytes, media_type)
                if img_text.strip():
                    parts.append(f"\n[Image]\n{img_text}")
            except Exception as e:
                logger.warning(f"DOCX image extraction failed: {e}")

    return [{"page": 1, "text": "\n".join(parts)}] if parts else []


def parse_excel(file_path: str) -> List[Dict[str, Any]]:
    """
    Excel — each sheet = one page, rendered as a markdown table.

    Sheets are the natural unit of separation in Excel — each sheet typically
    covers one topic (revenue, expenses, headcount). Keeping them separate
    means a query about "headcount" doesn't get polluted by revenue chunks.
    """
    xl = pd.ExcelFile(file_path)
    pages = []
    for sheet_num, sheet_name in enumerate(xl.sheet_names, start=1):
        df = xl.parse(sheet_name)
        if df.empty:
            continue
        md = df.to_markdown(index=False)
        pages.append({"page": sheet_num, "text": f"Sheet: {sheet_name}\n\n{md}"})
    return pages


def parse_csv(file_path: str) -> List[Dict[str, Any]]:
    """CSV — entire file as one markdown table."""
    df = pd.read_csv(file_path)
    return [{"page": 1, "text": df.to_markdown(index=False)}]


def parse_html(file_path: str) -> List[Dict[str, Any]]:
    """
    HTML — strip all tags, keep readable text.

    Removes nav, header, footer, scripts — noise that would pollute retrieval.
    Keeps the main content body.
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    lines = [line.strip() for line in soup.get_text(separator="\n").splitlines() if line.strip()]
    return [{"page": 1, "text": "\n".join(lines)}]


def parse_image(file_path: str) -> List[Dict[str, Any]]:
    """Standalone image file — send directly to Claude Vision."""
    ext = os.path.splitext(file_path)[1].lower()
    media_type_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp",
    }
    media_type = media_type_map.get(ext, "image/png")
    with open(file_path, "rb") as f:
        image_bytes = f.read()
    text = _extract_image_text(image_bytes, media_type)
    return [{"page": 1, "text": text}]


def parse_email(file_path: str) -> List[Dict[str, Any]]:
    """
    .eml file — extracts subject + body + attachments.

    Attachments are routed back through parse_document() recursively.
    So a .eml with a PDF attachment gives you email body on page 1,
    then every page of the PDF after that. Everything in one pipeline.
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


# ── Router — single entry point ───────────────────────────────────────────────


def parse_document(file_path: str) -> List[Dict[str, Any]]:
    """
    Detect format by extension and route to the right parser.

    This is the only function the rest of the system calls.
    Adding a new format = write one function + add one line in `parsers` dict.
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
