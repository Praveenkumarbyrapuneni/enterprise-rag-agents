# Enterprise RAG + Multi-Agent System

A production-grade financial document intelligence platform. Users upload earnings reports, SEC filings, and company documents. The system answers complex questions with cited sources, cross-references multiple documents simultaneously, and scores its own retrieval quality using RAGAS.

---

## Ingestion Pipeline — Document Parser

### Why the Parser is the Foundation

Every component downstream — chunker, embedder, vector store, agents — receives its input from the parser. If the parser loses context at this stage, that information is gone permanently. No retrieval logic can recover data that was never extracted.

The parser is therefore the most critical correctness boundary in the system. It was built first and built to handle every document format an enterprise financial environment produces.

---

### Supported Formats

| Format | Tool | Notes |
|--------|------|-------|
| PDF | PyMuPDF | Text layer + table detection + embedded images |
| DOCX | python-docx | Paragraphs + tables + embedded images |
| Excel (.xlsx, .xls) | pandas + openpyxl | Each sheet = one page |
| CSV | pandas | Entire file as one markdown table |
| HTML | BeautifulSoup | Tags stripped, content preserved |
| Images (.png, .jpg, .gif, .webp) | Claude Vision | OCR via vision LLM |
| Email (.eml) | Python stdlib `email` | Body + attachments parsed recursively |

---

### Architecture — 3-Pass System Per Page

PDF and DOCX documents run three passes on every page. No pass is optional — all three run regardless of what the page contains.

```
For each page:
    │
    ├── Pass 1: Text layer
    │   PyMuPDF / python-docx extracts the readable text.
    │   Fast. Handles the majority of content on most pages.
    │
    ├── Pass 2: Tables → Markdown
    │   PyMuPDF find_tables() / python-docx table objects detect embedded tables.
    │   Each table is converted to markdown format before storage.
    │
    │   Why markdown? Raw text extraction destroys table structure:
    │   "Revenue Q1 Q2 Apple 89B 90B Google 76B 78B" — column relationships gone.
    │   Markdown preserves them so the LLM in Phase 2 reads the data correctly.
    │
    └── Pass 3: Embedded Images → Claude Vision
        PyMuPDF extracts image bytes from the page.
        Each image is sent to Claude Vision API with a structured prompt.
        Claude reads tables in images, describes charts numerically, handles
        scanned content — output is merged back into the page text.

        Why Claude Vision instead of Tesseract (traditional OCR)?
        Tesseract matches pixel patterns to characters. It fails on complex layouts,
        multi-column text, and financial charts. Claude understands layout and context.
```

All three outputs are concatenated into a single text block per page. The rest of the pipeline receives clean, structured text regardless of what the original page contained.

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

**Single Qdrant collection with doc_id payload filter**
All documents are stored in one Qdrant collection. Document isolation at query time is achieved via payload filtering on `doc_id`, not by creating separate collections per document. Separate collections would require separate index management and make cross-document queries impossible. One collection, filtered by payload, handles both single-document and cross-document retrieval with no architectural change.

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
