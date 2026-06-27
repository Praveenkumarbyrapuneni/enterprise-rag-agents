# Enterprise RAG + Multi-Agent System

A production-grade financial document intelligence platform. Users upload earnings reports, SEC filings, and company documents. The system answers complex questions with cited sources, cross-references multiple documents simultaneously, and scores its own retrieval quality using RAGAS.

---

## Ingestion Pipeline — Document Parser

### Why the Parser is the Foundation

Every component downstream — chunker, embedder, vector store, agents — receives its input from the parser. If the parser loses context at this stage, that information is gone permanently. No retrieval logic can recover data that was never extracted.

The parser is therefore the most critical correctness boundary in the system. It was built first and built to handle every document format an enterprise financial environment produces.

---

### Supported Formats

| Format | Tool | Passes | Notes |
|--------|------|--------|-------|
| PDF | PyMuPDF | 3 | Text + tables + embedded images |
| DOCX | python-docx | 3 | Text + tables + embedded images |
| Excel (.xlsx) | pandas + zipfile | 2 | Cell values + embedded images from archive |
| Excel (.xls) | pandas | 1 | Cell values only — legacy format, no image layer |
| CSV | pandas | 1 | Entire file as markdown table, no images possible |
| HTML | BeautifulSoup | 2 | Text + `<img>` tags (base64, relative path, remote URL) |
| Images (.png, .jpg, .gif, .webp) | Claude Vision | 1 | Entire file is the image |
| Email (.eml) | Python stdlib `email` | recursive | Body + all attachments parsed through router |

---

### Architecture — Reading-Order Extraction

All parsers extract content in the exact visual reading order it appears on the page. Text, tables, and images are interleaved — not collected separately and concatenated. This preserves contextual connections like "as shown in the chart above" which would break if the chart and the sentence referencing it were separated.

```
PDF — per page:
    Collect text blocks, tables, and images each with x and y coordinates.
    Multi-column detection: if ≥25% of text blocks fall in each half of the
    page width, treat as two-column layout — left column output first
    (sorted by y), full-width elements next, then right column (sorted by y).
    Single-column: all elements sorted by y.
    Text blocks overlapping table regions excluded (avoid duplication).
    Images < 50×50px skipped (decorative icons, borders).
    Scanned page fallback: if zero content extracted after all passes,
    the page is rendered as a full bitmap at 150 DPI and sent to Claude Vision.
    This recovers scanned PDFs that have no text layer at all.

DOCX — document order:
    Headers extracted from doc.sections[0].header — prepended before body.
    Body XML iterated element by element in document order.
    Each element is a paragraph (<w:p>) or a table (<w:tbl>).
    Images in DOCX are inline inside paragraphs — extracted at the paragraph
    they belong to, preserving their position relative to surrounding text.
    Footers extracted from doc.sections[0].footer — appended after body.

HTML — DOM traversal:
    Walk the DOM tree once. Text and <img> elements collected as encountered.
    <img> src resolved from: base64 data URI / relative file path / remote URL.

Excel — drawing XML parsing:
    Drawings XML (xl/drawings/drawing*.xml) parsed for both images AND native
    chart objects. Native charts (bar, line, pie, etc.) were previously invisible
    — chart XML (xl/charts/chart*.xml) is now parsed to extract title, series
    names, category labels, and cached values formatted as markdown tables.
    Both images and charts inserted at their row anchor position in the sheet output.
```

---

### The Router Pattern — Single Entry Point

```python
parse_document(file_path)   # only function the rest of the system calls
    │
    ├── .pdf   → parse_pdf()
    ├── .docx  → parse_docx()
    ├── .xlsx  → parse_excel()
    ├── .csv   → parse_csv()
    ├── .html  → parse_html()
    ├── .png / .jpg / .webp → parse_image()
    └── .eml   → parse_email()
```

Nothing in the pipeline calls `parse_pdf()` or `parse_excel()` directly. Everything goes through `parse_document()`.

**Why this matters for extensibility:** Adding a new format requires writing one function and adding one line to the router dictionary. No other file in the system changes. The chunker, embedder, and Qdrant uploader are completely unaware that a new format was added.

---

### Uniform Output Contract

Every parser — regardless of input format — returns the same structure:

```python
List[{"page": int, "text": str}]
```

A PDF with 50 pages returns 50 dicts. An Excel file with 3 sheets returns 3 dicts. An image file returns 1 dict. The chunker receives the same structure in all cases and applies the same logic.

This contract is what makes the pipeline composable. Each component has one job and one expected input format.

---

### Recursive Email Parsing

Emails frequently contain attachments — PDFs, Excel files, images. The email parser handles this by routing attachments back through `parse_document()`:

```
parse_document("memo.eml")
    └── parse_email()
            ├── Extracts body text → page 1
            └── Finds attachment: "q4_earnings.pdf"
                    └── parse_document("q4_earnings.pdf")   ← recursive call
                            └── parse_pdf()
                                    └── returns pages 1–47 of the PDF

Final result: [email body] + [47 pages of the PDF attachment]
```

The recursion terminates naturally — attachments are not themselves emails with further attachments. The pipeline handles an email with any attachment type without any additional code — the router resolves the type automatically.

---

### Key Design Decisions

**Page-level granularity over full-document text**
Each page is stored as a separate dict with its page number. This enables citations — when the system answers a question in Phase 2, it can point to "page 47 of the 10-K filing" rather than just "somewhere in the document." In financial contexts, cited sources are non-negotiable.

**Tables converted at parse time, not query time**
Table structure is converted to markdown during ingestion, not when a query arrives. Converting at parse time means it happens once per document. Converting at query time would mean doing it on every retrieval, adding latency to every user request.

**Claude Vision for images, not Tesseract**
Traditional OCR (Tesseract) works by matching pixel patterns to known character shapes. It fails on complex financial layouts — multi-column tables, bar charts, pie charts with labels, watermarked scanned documents. Claude Vision understands layout and context, can describe numerical data in charts, and handles degraded scan quality. For enterprise financial documents, accuracy matters more than avoiding an API call.

**Reading order is preserved across all formats**
Content is never extracted in separate type-passes and then concatenated. Separating all text from all images breaks contextual connections — "as shown in the chart above" placed 300 tokens away from the chart it references gives the LLM no useful signal. Every parser outputs content in the exact order a human would read it. The implementation method differs per format (y-coordinate sorting for PDF, XML body iteration for DOCX, DOM traversal for HTML, row-anchor interleaving for Excel) but the contract is the same: output order = visual reading order.

**Pass count is determined by what content the format can contain**
PDF and DOCX run 3 passes because they can mix text, tables, and embedded images on the same page simultaneously. Excel runs 2 passes — cell values (Pass 1) and embedded images extracted from the xlsx ZIP archive (Pass 2). HTML runs 2 passes — text content (Pass 1) and `<img>` tags resolved from base64 URIs, relative paths, or remote URLs (Pass 2). CSV runs 1 pass because the format is plain text by definition — images are impossible in CSV. Standalone image files run 1 pass — the entire file is sent to Claude Vision. The pass count matches what the format is actually capable of containing, nothing more.

**Recursion only in email parsing, not in PDF or DOCX**
PDF and DOCX always run the same 3 passes — no conditions, no branching. Email is different because the content type of its attachments is unknown at parse time. Rather than hardcoding handling for every possible attachment type inside the email parser, attachments are routed back through `parse_document()`. The router resolves the type automatically. This means an email with a PDF attachment, an Excel file, and an image all get parsed correctly without a single line of attachment-specific logic in `parse_email()`.

**Single Qdrant collection with doc_id payload filter**
All documents are stored in one Qdrant collection. Document isolation at query time is achieved via payload filtering on `doc_id`, not by creating separate collections per document. Separate collections would require separate index management and make cross-document queries impossible. One collection, filtered by payload, handles both single-document and cross-document retrieval with no architectural change.

---

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Agent Orchestration | LangGraph | Stateful graphs, conditional edges, retry loops |
| LLM | Anthropic Claude (claude-sonnet-4-6) | Best reasoning, reliable citations |
| Embeddings | OpenAI text-embedding-3-large | Best retrieval performance at 3072 dimensions |
| Vector Store | Qdrant (Docker, self-hosted) | HNSW index, payload filtering, open source |
| Re-ranker | Cohere Rerank | Improves retrieval precision by 30–40% |
| Evaluation | RAGAS | Industry standard RAG evaluation framework |
| Tracing | LangSmith | Full agent trace visibility |
| Safety | Guardrails AI | PII redaction, prompt injection detection |
| API | FastAPI | Production REST API |
| Containerization | Docker + Docker Compose | Local dev parity with production |
| Cloud | AWS Lambda + S3 | Serverless deployment, document storage |
| Monitoring | Prometheus + Grafana | RAGAS score trends, latency, error rates |
| CI/CD | GitHub Actions | Automated deploy on push to main |

---

## Status

**Phase 1 — In progress**
- [x] Project structure, Docker Compose, environment setup
- [x] Document parser — all formats, 3-pass system
- [ ] Chunker — fixed-size and semantic strategies
- [ ] Embedder — OpenAI text-embedding-3-large
- [ ] Qdrant ingestion + similarity search
- [ ] Phase 1 milestone: end-to-end ingest and retrieve via Python script
