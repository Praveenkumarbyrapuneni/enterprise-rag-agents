# Enterprise RAG + Multi-Agent System

A production-grade financial document intelligence platform. Ingests earnings reports, SEC filings, analyst reports, and internal documents — then answers complex questions with cited sources, cross-references multiple documents simultaneously, and scores its own retrieval quality using RAGAS.

---

## Document Parser — Production-Grade Ingestion

Most RAG systems fail in production before a single agent runs. The failure point is the parser. A naive parser silently drops content — a scanned page returns empty, a table becomes an unreadable string, an embedded chart disappears entirely. The LLM downstream never knows what it missed.

This parser was built against every failure mode that surfaces in real enterprise financial environments. Every edge case below was identified, reasoned about, and explicitly handled.

---

### Failure Modes This Parser Does Not Have

**Silent content loss on scanned pages**
A fully scanned PDF page has no text layer. A naive parser calls `get_text()`, gets nothing, moves on. The page is gone. This parser detects zero-content pages after all extraction passes and renders the full page as a 150 DPI bitmap, then sends it to Claude Vision. Scanned annual reports, faxed documents, and legacy filings are fully recovered.

**Wrong reading order on multi-column layouts**
A naive parser sorts text blocks top-to-bottom across the full page width. On a two-column analyst report this interleaves content from both columns — the LLM reads gibberish. On a three-column academic paper the problem is worse. This parser detects any number of columns by finding gaps in the x-center distribution of text blocks — a gap wider than 8% of the page width with no text is a column gutter. Each column is output top-to-bottom, columns ordered left-to-right, exactly as a human reads it. Works for 1, 2, 3, or more columns without any hard-coded assumptions.

**Context broken by separating images from their captions**
A naive parser extracts all text first, all images last. "As shown in the chart above, revenue grew 23% YoY" is now 400 tokens away from the chart it references. This parser extracts content in exact visual reading order per page — text, image, text, table — sorted by y-coordinate. The chart and its caption are adjacent in the output.

**Duplicate image extraction wasting tokens and polluting retrieval**
A document header logo repeated across 40 pages. A naive parser sends the same image to Claude Vision 40 times, outputs the same text 40 times, stores 40 identical chunks in the vector database. This parser tracks image xrefs per page and processes each unique image exactly once.

**DOCX text boxes silently skipped**
Financial Word documents frequently use floating callout boxes and sidebar panels to highlight key statistics, warnings, and risk factors. These are `w:txbxContent` elements inside drawing objects — not paragraphs, not tables. A naive python-docx parser iterating paragraphs never sees them. This parser explicitly searches drawing elements for text box content and extracts it in document order.

**DOCX embedded charts invisible**
Word documents regularly embed Excel charts — revenue charts, margin trends, segment breakdowns. These are `c:chart` references inside drawing elements, not images. A naive parser scanning for `a:blip` (image references) misses them entirely. This parser resolves `c:chart` references through the document relationship map, parses the chart XML for cached series data, and outputs it as a structured markdown table.

**Excel native chart objects invisible**
Excel's built-in charts (bar charts, pie charts, line charts) are not image files. They live in `xl/charts/chart*.xml` inside the xlsx archive. A parser extracting from `xl/media/` misses every single one. This parser parses the drawing XML for both `a:blip` (images) and `c:chart` (native chart objects), extracts each chart's title, series names, category labels, and cached numeric values, and inserts them at the row they're anchored to in the sheet.

**Excel merged cell headers outputting NaN**
Financial spreadsheets almost always have merged header rows — a label spanning four columns. pandas reads merged cells as NaN except the top-left cell. The markdown table output then has `| Revenue | NaN | NaN | NaN |`. This parser forward-fills merged regions horizontally and vertically before converting to markdown, so the header appears correctly across all columns.

**HTML tables destroyed by text extraction**
Walking an HTML page with `get_text()` turns a structured revenue table into `"Revenue Q1 Q2 Q3 Apple 89B 90B Google 76B 78B"` — all column relationships gone. This parser detects `<table>` elements during DOM traversal and converts them to markdown before continuing the walk. Structure is preserved everywhere the LLM needs it.

**Corporate email inline images missed**
HTML corporate emails embed images inline using the `cid:` scheme — `<img src="cid:chart1@report">`. The MIME part containing the image has a matching `Content-ID` header. A parser calling `soup.get_text()` on the HTML body strips all images. This parser first collects all MIME parts by Content-ID, then walks the HTML body with a custom DOM traverser that matches `cid:` references to the collected images and sends each to Claude Vision.

**TIFF files unreadable**
Document scanners output TIFF. Financial document management systems archive as TIFF. Claude Vision does not accept TIFF natively. A parser without TIFF support returns an unsupported format error. This parser converts each TIFF frame to PNG in memory via Pillow and sends it to Vision. Multi-page TIFF files — entire document batches scanned into one file — are split into one output page per frame.

**CSV encoding and delimiter crashes**
Non-UTF-8 CSVs from legacy financial systems crash with `UnicodeDecodeError`. Tab-separated or semicolon-separated exports are misread as single-column data. This parser auto-detects delimiter using pandas' sniffing engine and falls back from UTF-8 to Latin-1 on decode errors.

**Password-protected files crashing with cryptic errors**
A raw `fitz` exception on a password-protected PDF gives the caller a C library error with no actionable information. This parser checks `doc.is_encrypted` immediately after open and raises a clear `ValueError` naming the file. Excel files are caught the same way.

**DOCX headers and footers silently skipped**
Financial documents put critical metadata in headers and footers — document title, classification level, date, legal notices, confidentiality disclaimers. A naive python-docx parser iterating body paragraphs never touches them. This parser extracts headers from all document sections before body content and footers after, with deduplication for consistent headers across sections.

---

### Known Limitation

**PDF vector graphics**
Charts drawn as PDF path instructions — `moveto`, `lineto`, `fill` — are invisible to both text extraction and image extraction. They are not text characters and they are not embedded image files. They are mathematical drawing commands that a PDF viewer renders on screen.

This affects PDFs exported from Excel, PowerPoint, or design tools where charts become vector objects rather than raster images. The scanned page fallback does not apply because these pages have readable text alongside the vector graphics — only the chart is missing, not the whole page.

The correct fix is detecting pages with significant vector drawing content and selectively rendering those regions as bitmaps. This is not yet implemented. It is the one category of content this parser cannot recover.

---

### Supported Formats

| Format | Extraction |
|--------|-----------|
| PDF | Text + tables + images sorted by vertical position. Scanned pages rendered to bitmap. N-column layout auto-detected and reordered (any number of columns). Duplicate images deduplicated. |
| DOCX | Body XML in document order. Inline images, floating text boxes, and embedded charts extracted at their paragraph. All section headers and footers included. |
| Excel (.xlsx) | Cell values with merged regions filled. Native chart objects parsed from XML. Images and charts inserted at anchor row positions. |
| Excel (.xls) | Cell values only — legacy binary format has no accessible drawing layer. |
| CSV | Delimiter and encoding auto-detected. Outputs as markdown table. |
| HTML | Single DOM walk. Text, tables (markdown), and images collected in document order. Images resolved from base64 URIs, relative paths, or remote URLs. |
| Images (.png .jpg .gif .webp) | Sent directly to Claude Vision. |
| TIFF (.tif .tiff) | Each frame converted to PNG via Pillow. Multi-frame files produce one page per frame. |
| Email (.eml) | Body text extracted. cid: inline images resolved from MIME parts. All attachments parsed recursively through the format router. |

---

### Architecture

**Single entry point**

```
parse_document(file_path)
    ├── .pdf         → parse_pdf()
    ├── .docx        → parse_docx()
    ├── .xlsx/.xls   → parse_excel()
    ├── .csv         → parse_csv()
    ├── .html/.htm   → parse_html()
    ├── .png/.jpg/.gif/.webp → parse_image()
    ├── .tif/.tiff   → parse_tiff()
    └── .eml         → parse_email()
```

Nothing downstream calls a format-specific function directly. Adding a new format requires writing one function and one line in the router — no other file changes.

**Uniform output**

Every parser returns `List[{"page": int, "text": str}]` regardless of input format. The chunker, embedder, and vector store are completely format-agnostic.

**Recursive email handling**

Email attachments are unknown format at parse time. Rather than branching on attachment type inside the email parser, each attachment is saved to a temp file and routed through `parse_document()`. The router resolves the type. An email containing a PDF, an Excel file, and a TIFF scan is fully parsed without a single format-specific line inside `parse_email()`.

**Reading order per format**

| Format | How reading order is achieved |
|--------|------------------------------|
| PDF | Text blocks, tables, images collected with x/y coordinates, sorted by y. N-column detection via x-center gap analysis — any column count, reordered left-to-right then top-to-bottom per column. |
| DOCX | Body XML iterated in document order. Images handled inside their parent paragraph, not in a separate pass. |
| HTML | Single DOM traversal. Tables and images processed where they appear, not after all text. |
| Excel | Drawing XML parsed for row anchor positions. Images and charts inserted at the row they visually belong to. |
| Email | cid: images resolved inline during HTML body traversal, not appended after text. |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                           │
│                                                                 │
│  Files arrive (any of 9 supported formats)                      │
│       ↓                                                         │
│  Hash check → PostgreSQL — duplicate? skip entirely            │
│       ↓                                                         │
│  Task queue → Celery + Redis (SQS on AWS)                       │
│       ↓                                                         │
│  Worker fleet — N parallel workers (ECS on AWS)                 │
│    Parser → Chunker → Embedder → Qdrant uploader               │
│    Every chunk carries: file, page, chunk_index, hash, ts       │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI REST API                            │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LANGGRAPH ORCHESTRATOR                        │
└──────┬──────────────┬──────────────┬──────────────┬────────────┘
       │              │              │              │
       ▼              ▼              ▼              ▼
┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────────┐
│  Agent 1 │  │   Agent 2    │  │  Agent 3 │  │   Agent 4    │
│  Query   │  │  Retriever   │  │Synthesizer│  │  Evaluator  │
│ Analyzer │  │  + Reranker  │  │ Citations│  │   (RAGAS)   │
└──────────┘  └──────┬───────┘  └──────────┘  └──────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      QDRANT VECTOR STORE                        │
│           Single collection — doc_id payload filtering          │
└─────────────────────────────────────────────────────────────────┘
```

**Agent 1 — Query Analyzer:** Decomposes complex questions into sub-queries. Identifies relevant documents. Detects cross-document vs single-document queries. Implements HyDE (Hypothetical Document Embeddings) for better retrieval.

**Agent 2 — Retriever:** Runs parallel retrieval across Qdrant filtered by doc_id. Applies Cohere reranker (30–40% precision improvement). Applies MMR to eliminate redundant chunks.

**Agent 3 — Synthesizer:** Generates grounded answers using only retrieved context. Every factual claim carries a citation. Refuses to answer when information is absent rather than hallucinating.

**Agent 4 — Evaluator:** Scores every response using RAGAS — faithfulness, answer relevance, context recall. Triggers Agent 2 retry with different retrieval parameters when scores fall below threshold. The system self-corrects.

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| Agent Orchestration | LangGraph | Stateful graphs, conditional edges, self-correcting retry loops |
| LLM | Anthropic Claude (claude-sonnet-4-6) | Best reasoning, reliable grounded citations |
| Embeddings | OpenAI text-embedding-3-large | 3072-dimension vectors, best retrieval performance |
| Vector Store | Qdrant (Docker) | HNSW index, payload filtering, open source |
| Re-ranker | Cohere Rerank | Precision improvement over raw vector search |
| Evaluation | RAGAS | Industry-standard RAG evaluation without ground truth |
| Tracing | LangSmith | Full agent trace visibility per request |
| Safety | Guardrails AI | PII redaction, prompt injection detection |
| API | FastAPI | Production REST API |
| Containerization | Docker + Docker Compose | Local/production environment parity |
| Cloud | AWS Lambda + S3 | Serverless deployment, document storage |
| Monitoring | Prometheus + Grafana | RAGAS score trends, latency, error rates |
| CI/CD | GitHub Actions | Automated deploy on merge to main |

---

## Status

**Phase 1 — Foundation**
- [x] Project structure, Docker Compose, environment setup
- [x] Document parser — 9 formats, production edge cases handled
- [ ] Chunker — fixed-size and semantic strategies
- [ ] Embedder — OpenAI text-embedding-3-large
- [ ] Qdrant ingestion + similarity search
- [ ] Milestone: end-to-end ingest and retrieve via Python script

**Phase 2 — Agents** (Week 2)
**Phase 3 — Evaluation** (Week 3)
**Phase 4 — Production Deployment** (Week 4)
