# Ingestion Pipeline — Engineering Decisions

Documents the key architectural decisions made during ingestion pipeline development,
including production issues discovered and how they were resolved.

---

## Parser

### 9-Format Coverage Including TIFF

Financial institutions handle more than PDFs. SEC filings arrive as HTML, audit
reports as XLSX, client communications as EML, legacy documents as TIFF scans.
The parser handles all 9 formats through a single `parse_document()` entry point.

TIFF support specifically requires OCR via Pillow — it is the format used for
scanned legacy documents and is common in legal and compliance workflows.
Skipping it would leave a class of documents entirely invisible to the system.

Rejected: a PDF-only parser. Breaks immediately when a client sends anything else.

### Vision Client Singleton + 30-Second Timeout

```python
_vision_client    = None
_vision_client_lock = threading.Lock()
_VISION_TIMEOUT   = 30.0

_vision_client = anthropic.Anthropic(api_key=key, timeout=_VISION_TIMEOUT)
```

A financial PDF with 50 embedded images would create 50 separate Anthropic client
instances without the singleton — 50 connection pool setups in rapid succession.
The double-checked locking pattern ensures one client is created regardless of
concurrent calls.

The 30-second timeout kills hung Vision calls. Without it, a single unresponsive
Vision API call blocks the entire parser for up to 600 seconds (the library default),
holding a Celery worker slot that should be processing other documents.

### N-Column PDF Layout Detection

Financial filings frequently use 2-column layouts (MD&A sections, footnotes).
Standard PDF text extraction reads elements in the order they appear in the PDF
stream — not in visual reading order. A 2-column page read left-to-right produces
interleaved text from both columns, making it unreadable.

Fix: x-center positions of all text blocks are analyzed. Gaps wider than 8% of
page width indicate column gutters. Content is then emitted column-by-column,
top-to-bottom within each column, columns left-to-right. Full-width elements
(headers, wide tables spanning >60% of page width) interrupt column flow and are
emitted at their correct vertical position.

Works for 1, 2, 3, or more columns without configuration.

### Text Blocks Overlapping Table Regions Excluded

PDF text extraction and table extraction are separate operations. Without
coordination, text inside a detected table appears twice: once as raw text
blocks, once as a structured table from the table extractor.

Fix: table bounding boxes are collected first. Any text block whose bounding box
overlaps a table region is excluded from the text block list. Tables are always
rendered as markdown, never as raw text.

### Image Deduplication by xref Per Page

The same image object in a PDF (company logo, repeated chart) can be referenced
from multiple positions on the same page. Without deduplication, Vision is called
once per reference — the same image is extracted and described multiple times.

Fix: each image's `xref` (internal PDF object ID) is tracked in a set per page.
An image with an already-seen xref is skipped. Vision is called exactly once
per unique image per page.

### Images Under 50×50px Skipped

PDF documents contain thousands of decorative elements: horizontal rules,
bullet point graphics, watermarks, icon borders. These are typically <50px
in at least one dimension.

Sending these to Vision is wasted API cost and produces noise in the extracted
text ("small decorative line" repeated 200 times across a filing). The 50×50px
threshold filters decorative elements while retaining all meaningful content
images (charts, diagrams, photos).

### Scanned Page Fallback at 150 DPI

Some PDF pages contain no text layer at all — they are scanned images embedded
in a PDF container. Standard text extraction returns zero content for these pages.

When zero elements are extracted from a page, the parser treats it as a scanned
page: renders it as a full-page bitmap at 150 DPI and sends it to Claude Vision.

150 DPI: sufficient for OCR quality on financial text. Higher DPI increases
file size and Vision processing time with diminishing accuracy returns.

### PDF Vector Graphics Limitation — Acknowledged, Not Fixed

Charts drawn as PDF path instructions (lines, rectangles, fills) are mathematically
defined, not bitmaps. Both text extraction and image extraction are blind to them.

The only fix is rendering every page as a bitmap and sending it to Vision —
expensive for a 500-page filing with few charts. This limitation is documented
explicitly in the parser module docstring rather than hidden.

When this matters: quarterly earnings reports with bar charts showing revenue trends.
The chart data is invisible to the parser. The surrounding text usually contains
the same data in prose, which the parser does extract.

### Password-Protected File Detection

Both PDF and Excel parsers detect encrypted/password-protected files and raise
a clear `ValueError` with the file path and instructions.

Without this: PyMuPDF raises a cryptic `fitz.fitz.EmptyFileError` or processes
the file silently returning 0 pages. The document enters the DLQ with a
confusing error message. No one knows the file needs decryption.

The clear ValueError goes into the DLQ `error_message` field. The operations
team sees exactly what happened and what to do.

### DOCX Text Boxes and Floating Elements

Standard python-docx iterates `doc.paragraphs` — this misses floating text boxes
(`wps:txbx` elements). Financial presentations use text boxes extensively for
callout statistics, sidebar commentary, and highlighted figures.

Fix: the body XML is iterated directly (not through python-docx's high-level API)
using `doc.element.body`. Every `<w:drawing>` element is inspected for:
- Inline images (`a:blip` elements)
- Floating text boxes (`w:txbxContent`)
- Embedded Excel charts (`c:chart`)

All are extracted in their document position — not appended at the end.

### DOCX Headers and Footers — All Sections, Deduplicated

Financial filings use headers and footers for document identifiers, filing dates,
and page numbers. Standard body extraction misses them entirely.

Each DOCX section can have its own header and footer. Multiple sections with
identical headers (common when the filing repeats the same header across all
pages) would produce the same text multiple times without deduplication.

Fix: headers and footers are extracted from all sections, tracked in a seen set,
and included once each.

### Excel Merged Cells Forward-Filled

Excel reports frequently use merged cells for multi-column headers ("Q1 2024
Revenue" spanning 4 columns). When pandas reads a merged cell, all but the
first cell return `NaN`. The table appears to have empty columns.

Fix: `df.ffill(axis=0).ffill(axis=1)` — forward-fill along both axes. Each
merged cell's value propagates to all cells it logically covers.

### Excel Drawing Content at Anchor Row Positions

Excel images and charts are positioned at specific row anchors in the spreadsheet
(e.g., a chart appears after row 15, between revenue and operating income rows).

Without anchor position tracking, all embedded content is appended at the end
of the table — destroying the relationship between the chart and the data it
visualizes.

Fix: anchor row positions are extracted from drawing XML. Embedded content is
inserted immediately after its anchor row in the markdown output, preserving
the visual relationship with surrounding data.

### CSV Delimiter and Encoding Auto-Detection

Financial data exports use varying delimiters (comma, tab, semicolon, pipe) and
encodings (UTF-8, Latin-1, Windows-1252). Hardcoding either breaks on files
from different systems.

`sep=None, engine="python"` tells pandas to detect the delimiter automatically.
UTF-8-sig handles BOM-prefixed files from Excel exports. Latin-1 fallback
handles files from legacy financial systems that don't support Unicode.

### HTML: script/style/nav/footer/header Stripped

SEC filing HTML exports contain navigation menus, cookie banners, JavaScript
blocks, and CSS style definitions. None of this is document content.

Before DOM traversal, these tags are decomposed (removed from the tree). The
parser operates on content only — no script code or navigation labels appear
in the extracted text.

### HTML Images from Three Sources

HTML documents can embed images in three ways:
1. `data:image/png;base64,...` — inline base64 (self-contained)
2. `src="path/to/image.png"` — relative file path (local HTML exports)
3. `src="https://..."` — remote URL (linked images)

All three are resolved and sent to Vision. Missing any one of them leaves
charts and diagrams in the filing invisible to the pipeline.

Remote URLs use a 10-second timeout to avoid blocking the parser on slow
external servers.

### Email: cid: Inline Images Resolved Before HTML Traversal

Corporate HTML emails embed images using the `cid:` scheme:
`<img src="cid:uniqueid@domain.com">`. These reference MIME parts in the email,
not external URLs. Standard HTML image resolution ignores `cid:` references
entirely — all inline images are invisible.

Fix: before HTML traversal, all MIME parts with a `Content-ID` header are
collected into a `cid_images` dictionary mapping content ID to image bytes.
During HTML traversal, `cid:` references are matched against this dictionary
and resolved to Vision calls.

### Email: cid Image NameError Bug Fixed

The original `_walk_html_email` function did not accept `file_path` as a
parameter. When a `cid:` image failed to extract, the exception handler tried
to log `f"[{file_path}]..."` — but `file_path` was not in scope.

Result: the real error (image extraction failed) caused a secondary NameError.
The parser crashed with `NameError: name 'file_path' is not defined` — a
completely misleading error that hid the actual cause.

Fix: `file_path` added as a parameter to `_walk_html_email`, passed through
from `parse_email`. The real error is now logged correctly.

This is the category of bug that production reveals and local testing doesn't:
it only triggers when a cid image extraction fails, which requires a specific
email with a broken inline image.

### Email: Attachments Parsed Recursively via Temporary Files

Email attachments (PDFs, Word docs, Excel files) are parsed by extracting them
to a temporary file, calling `parse_document()` on the temp file, then deleting
the temp file in a `finally` block.

Recursive parsing means a PDF attached to an email gets the full PDF parser
treatment: N-column layout detection, table extraction, scanned page fallback.
The attachment is not treated as a plain text blob.

The `finally: os.unlink(tmp_path)` ensures temp files are always deleted even
if the parse fails — no disk leak on error.

---

## Chunker

### Content-Aware Strategy Routing

Early implementation used fixed chunking (512 tokens) for all documents.

Problem: financial filings contain deeply nested sections — an MD&A section
referencing figures from three pages earlier. A fixed 512-token chunk captures
the reference but not the context. Retrieval returned fragments with no usable
information.

Decision: content-aware routing. A classifier inspects each document and routes:
- Simple documents (news, summaries) → fixed chunking (512 tokens)
- Complex documents (10-K, earnings reports, legal) → hierarchical chunking

Hierarchical chunking creates two levels:
- Parent chunk: 1024 tokens — the full readable paragraph
- Child chunk: 256 tokens — the searchable unit stored as a vector in Qdrant

At query time, Qdrant finds the best-matching child chunk, then the retriever
fetches the full parent paragraph from PostgreSQL. The synthesizer reads
complete context, not sentence fragments.

### Token Counts: 1024 Parent / 256 Child

These numbers are not arbitrary. Cohere Embed v3 encodes up to 512 tokens
effectively. Child chunks at 256 tokens stay well within that ceiling.
Parent chunks at 1024 tokens provide enough surrounding context for Claude to
synthesize a complete, grounded answer.

Rejected: 512/128 split. Too small — parent paragraphs were incomplete sentences.
Rejected: 2048/512 split. Child chunks at 512 tokens caused 5-15% embedding
accuracy loss as the model compresses meaning across too long an input.

### Sentence Splitter Abbreviation Protection

The sentence splitter splits on `.`, `!`, `?` boundaries. Without protection,
"Dr. Smith reviewed Apple's Q3 revenue." splits into ["Dr.", "Smith reviewed
Apple's Q3 revenue."] — producing a 1-token chunk "Dr." that carries no
meaningful information and embeds as a near-zero vector.

Fix: before splitting, abbreviations are temporarily replaced with a placeholder
(`Dr\x00DOT\x00` instead of `Dr.`), the split runs, then the placeholder is
restored.

Financial documents are dense with abbreviations: Dr., Corp., Inc., U.S.,
e.g., i.e., etc. — every one of them would produce a false split without this.

### Child Chunks Carry Their Own Page Number

Each child chunk's `page` field reflects the actual page its content came from,
not the parent's first page.

A parent chunk spans pages 12-14. Without per-sentence page tracking, all three
child chunks created from it get `page=12`. When the synthesizer cites
`[Source: Goldman_10K.htm, page 12, Chunk #47]`, the analyst goes to page 12
and can't find the sentence — it's actually on page 13.

Fix: page numbers are tracked at the sentence level throughout the chunking
process. Each sentence carries its page number. When child chunks are assembled,
they inherit the page number of the first sentence they contain.

### First 3 Pages Only for Classification

```python
sample_text = " ".join(p.get("text", "") for p in pages[:3])
```

Document classification only needs the first 3 pages. The cover page, filing
header, and opening section are enough to determine whether a document is a
financial filing.

Without this: a 500-page Goldman Sachs 10-K builds a 2MB+ string to check
for financial keywords. At 1,000 documents/day, that's 2GB of string
construction purely for classification — memory pressure that slows workers
and can trigger OOM.

3 pages: ~4KB of text. Sufficient signal, negligible memory.

### Local Classifier Option — Regulated Environments

Some financial institution environments do not permit any data leaving their
infrastructure — including to AWS Bedrock. For these environments, a local
open-source model via Ollama can run document classification entirely on-premise.

Set `CHUNK_CLASSIFIER=local` and `LOCAL_CLASSIFIER_MODEL=gemma3:1b`. Falls
back to keyword scoring if Ollama is not running.

A 1B parameter model is sufficient for binary YES/NO financial document
classification. Accuracy is slightly lower than Haiku but remains above 90%.

### Classifier: Keyword First, Haiku Fallback

The content-aware router uses keyword scoring first (zero latency, no API cost).
Only when keyword confidence is low does it call Claude Haiku for classification.

This matters at Discover scale: classifying 1 million documents per day at
$0.00025 per Haiku call adds $250/day. Keyword routing handles 80%+ of cases
for free.

---

## Embedder

### Cohere Embed v3 via Bedrock, Not OpenAI

OpenAI text-embedding-3-large was the initial option (3072 dims, strong benchmark scores).

Rejected for one reason: data sovereignty. Financial institution clients require
that document content never leaves a controlled VPC. Sending SEC filings and
client transaction data to OpenAI's API is not permissible under most financial
institution security policies.

Cohere Embed v3 via AWS Bedrock produces 1024-dimensional vectors with
comparable retrieval quality. All data stays within the AWS VPC — the hard
requirement for this client class.

### Production Bug: Bedrock 2048-Char Limit

Discovery during testing: Bedrock validates the length of text inputs before
passing them to Cohere. Any text exceeding 2048 characters causes the entire
batch to fail with a ValidationException — even though Cohere itself accepts
longer inputs.

Fix: truncate all texts to 2048 characters at the application layer before
calling `invoke_model`. Applied in `_embed_batch_with_retry()`.

This is not documented prominently in Bedrock's API reference. It was found
through a production failure.

### input_type Asymmetry

Cohere Embed v3 requires different `input_type` values for indexing vs querying:
- Ingestion: `"search_document"`
- Query time: `"search_query"`

Mixing these degrades retrieval accuracy by 5-15%. The embedder enforces
`"search_document"` at ingestion. The retriever enforces `"search_query"` at
query time. These are never swapped.

### Double-Checked Locking on Bedrock Client — 20 Threads Race at Init

```python
_bedrock_client = None
_bedrock_lock   = threading.Lock()

def _get_bedrock():
    if _bedrock_client is None:
        with _bedrock_lock:
            if _bedrock_client is None:   # second check
                _bedrock_client = boto3.client(...)
```

The embedder runs 20 Haiku enrichment threads concurrently. All 20 start at
roughly the same time. Without locking, all 20 check `_bedrock_client is None`,
find None, and all 20 try to create a new boto3 client simultaneously — 20
separate TCP connections opened in the same millisecond.

The double-checked lock ensures exactly one client is created regardless of
how many threads start simultaneously. Same pattern in chunker.py for the
same reason.

### _enrich_single Never Raises — ThreadPoolExecutor Safety

```python
def _enrich_single(idx, chunk, context_text, file_name):
    try:
        # ... all enrichment logic
        return idx, enriched
    except Exception as e:
        logger.warning(...)
        return idx, chunk.get("text", "")  # always returns
```

If a thread function raises an unhandled exception inside `ThreadPoolExecutor`,
the exception is stored and re-raised when `future.result()` is called in the
main thread. This would crash `embed_chunks()` entirely — discarding all
enrichment work done by the other 19 threads.

`_enrich_single` is guaranteed to always return `(idx, text)`. Any exception
inside is caught, logged, and returns the raw chunk text as fallback. The main
thread always gets a result, never an exception.

### Return (idx, text) Tuple — Order Preservation in Concurrent Enrichment

```python
future_map[executor.submit(_enrich_single, idx, chunk, ctx, file_name)] = idx
...
for future in as_completed(future_map):
    result_idx, enriched = future.result()
    enriched_texts[result_idx] = enriched  # placed at original index
```

`as_completed()` returns futures in the order they finish — not the order they
were submitted. Thread 7 may finish before Thread 1. If results were appended
to a list in completion order, chunk 7's vector would be stored at index 0 —
silently pairing every chunk with the wrong vector.

The function returns `(original_index, text)`. Results are placed at their
original index in a pre-allocated list. Order is always correct regardless of
which thread finishes first.

### Vector Count Mismatch Detection — zip() Silently Corrupts Data

```python
if len(all_vectors) != len(enriched_texts):
    raise RuntimeError(
        f"EMBED MISMATCH: API returned {len(all_vectors)} vectors "
        f"for {len(enriched_texts)} chunks. All vectors discarded."
    )
```

`zip(chunks, vectors)` silently truncates to the shorter list. If the Bedrock
API returns 89 vectors for 90 chunks (API bug, network corruption), `zip()`
produces 89 results. One chunk gets no vector. It's dropped silently. No error.

The explicit length check fails loudly instead of storing corrupted data.
The document is re-queued rather than ingested with a missing chunk.

Silent data corruption in a financial document retrieval system is worse than
a loud failure that triggers a retry.

### Retriable Error Codes Whitelist — Not All API Errors

```python
_RETRIABLE = {"ThrottlingException", "ServiceUnavailableException", "InternalServerException"}

if error_code in _RETRIABLE:
    # retry with backoff
else:
    raise  # immediately
```

Not every Bedrock error is worth retrying. Three specific codes represent
transient server-side conditions that resolve on their own. All other errors
(ValidationException, AccessDeniedException, ResourceNotFoundException) indicate
a problem with the request itself. Retrying the same request 5 times produces
the same error 5 times and wastes 30+ seconds.

Only retry what the server says is retriable. Fail immediately on everything else.

### disallowed_special=() — SEC Filings Contain Special Tokens

```python
_tokenizer.encode(text, disallowed_special=())
```

tiktoken raises `ValueError` when it encounters special tokens like
`<|endoftext|>` in text. SEC filings occasionally contain these strings —
they appear in raw HTML exports and OCR output from scanned documents.

Without `disallowed_special=()`, a single SEC filing containing one of these
tokens crashes the entire chunking and embedding step for that document.

`disallowed_special=()`: encode special tokens as regular text instead of
raising. Token count remains accurate. Applied consistently in chunker.py
and embedder.py.

### Batch Size 90 vs Cohere Maximum of 96

The Cohere API on Bedrock accepts up to 96 texts per batch. The embedder uses 90.

The 6-item margin exists because Bedrock's enforced limits occasionally differ
from documented limits by a small margin. A batch of exactly 96 at peak load
has been observed to fail with a limit error that a batch of 90 never triggers.
Conservative headroom is cheaper than debugging an intermittent batch failure
at 3am.

### Contextual Enrichment Before Embedding

Before embedding each chunk, Haiku generates a 1-2 sentence context summary
describing how the chunk fits within the document. This enriched text is what
gets embedded — not the raw chunk.

Research basis: contextual retrieval (Anthropic, 2024) shows 49% reduction in
retrieval failures for domain-specific corpora. Financial documents have high
semantic density — chunks without context embed ambiguously and retrieve poorly.

20 parallel workers run enrichment concurrently. Each worker is stateless.
Haiku failure on any chunk degrades gracefully to raw chunk text — non-fatal.

---

## Qdrant Uploader

### TurboQuant 4-bit Over ScalarQuantization

Initial deployment used ScalarQuantization (int8): 4x memory compression.

Upgraded to TurboQuantization (4-bit, Google Research / ICLR 2026): 8x memory
compression, ~1% recall loss on text vectors.

At 10 million documents × 1024 dimensions × 4 bytes = ~41GB uncompressed.
ScalarQuant: ~10GB. TurboQuant: ~5GB.

The 5GB difference determines whether the index fits in RAM on a single node.
`always_ram=True` keeps the quantized index hot. Query vectors are scored at
full float32 precision (asymmetric quantization) — no rescore step needed.

Requires `qdrant-client >= 1.18.0`.

### HNSW Parameters: m=16, ef_construct=200

HNSW graph construction parameters directly control the recall/speed tradeoff.

`m=16`: each node connects to 16 neighbors during index build. Production
benchmark on financial document corpora: m=16 achieves 97%+ recall at 100ms
p99 latency. Raising to m=24 adds 20% memory with <1% recall gain at this scale.

`ef_construct=200`: search depth during index construction. Lower values build
faster but produce a lower-quality graph. 200 is the production-validated
baseline for financial document retrieval.

### 4-Phase Idempotent Write

The upload pipeline uses 4 committed phases to prevent duplicate data on
partial failure (Celery worker crash, network timeout, OOM):

1. Check status — if `complete`, return immediately (skip). If `processing`
   from a prior crashed run, delete the partial Qdrant write before retrying.
2. Store parent chunks in PostgreSQL (committed before Qdrant write).
3. Upload child chunks to Qdrant with batch retry.
4. Mark `complete` in PostgreSQL.

If any phase fails, status stays `processing`. The next invocation with the
same `file_hash` detects this and re-runs from a clean state.

Rejected: upsert-on-every-run without a status check. At 1 million documents/day,
re-processing a document that already succeeded wastes Bedrock embedding cost
and risks creating duplicate vectors.

### tenant_id Payload Index Created Before First Write

Qdrant allows adding payload indexes at any time — but adding one after data
exists triggers a full collection rebuild. At 10M vectors this takes hours and
blocks all queries.

The `tenant_id` KEYWORD payload index is created when the collection is
initialized, before any data is written. Same applies to `file_hash`,
`file_name`, `strategy`, and `page`.

### 4xx vs 5xx Error Distinction — Never Retry Caller Errors

The Qdrant batch uploader distinguishes between two categories of errors:

```python
except UnexpectedResponse as e:
    if e.status_code >= 500:
        pass   # Qdrant server error — retriable
    else:
        raise  # 4xx — not retriable, raise immediately
```

5xx errors: Qdrant server crashed, out of memory, overloaded. These are
transient — retrying after a backoff usually succeeds.

4xx errors: the request itself is malformed. Wrong collection name, invalid
vector dimensions, bad payload format. Retrying the same bad request 5 times
produces the same error 5 times. Raise immediately and let the DLQ record it.

This pattern applies everywhere in the system — Qdrant, Bedrock, PostgreSQL.
4xx = fix the code. 5xx = wait and retry.

### Jitter on Qdrant Batch Retry

The Qdrant batch uploader uses the same jitter principle as the orchestrator:

```python
wait = delay + random.uniform(0, 0.5 * delay)
delay *= 2
```

Starting at 1s, doubling each attempt, with 0-50% random jitter added.
At batch size 256 points, 100 parallel workers retrying Qdrant simultaneously
would saturate its write buffer. Jitter prevents the synchronized retry storm.

### wait=False on Upsert

`wait=False` tells Qdrant to acknowledge the write before updating the HNSW
index. This increases write throughput significantly.

Safety: PostgreSQL `ingestion_status` is the source of truth. If Qdrant crashes
between write acknowledgement and index commit, status stays `processing`.
The next run's Phase 1 detects this and re-uploads from scratch. No data loss.

---

## Orchestrator

### Exponential Backoff + Jitter — Thundering Herd Prevention

When a transient error happens (network blip, Qdrant momentarily unavailable),
a naive retry fires immediately. When 100 Celery workers all fail at the same
moment and all retry at the same moment, they hit the same overloaded service
simultaneously — making it worse. This is called the thundering herd problem.

The retry delay formula used:
```python
retry_in = min(2 ** self.request.retries + random.uniform(0, 1), 60)
```

What this means:
```
Worker A fails, retries=0: waits 2^0 + 0.73s = 1.73s
Worker B fails, retries=0: waits 2^0 + 0.12s = 1.12s
Worker C fails, retries=0: waits 2^0 + 0.91s = 1.91s

All three hit the service at different times — load is spread out.

On retry 1: ~2-3s wait
On retry 2: ~4-5s wait
On retry 3: ~8-9s wait
Max wait:   60s (capped — doesn't grow forever)
```

The `random.uniform(0, 1)` is the jitter — a random number added to each
worker's wait time so no two workers retry in sync. Without it, exponential
backoff alone still causes synchronized retries because all workers started
at the same time.

### Transient vs Non-Transient Error Classification

Not every error is worth retrying. Retrying a bad input wastes workers.

```python
TRANSIENT_ERRORS = (
    ConnectionError, TimeoutError, OSError,       # network issues
    BrokenPipeError, ConnectionResetError,         # connection drops
)
```

Transient errors → retry with backoff. These usually resolve on their own.

Everything else (ValidationError, bad file format, corrupt PDF, wrong model ID)
→ skip retries entirely → go straight to DLQ.

Retrying a corrupt PDF 3 times before giving up wastes 3 worker slots and
delays other documents in the queue. Detect immediately, fail fast, record it.

### Full Traceback Stored in DLQ

The failed_documents table stores the complete Python traceback, not just the
error message.

```
error_message:   "ValidationException: model ID invalid"        ← what
error_traceback: "File orchestrator.py, line 330, in ingest..." ← where exactly
retry_count:     3                                               ← how many times we tried
celery_task_id:  "abc-123-..."                                   ← which worker ran it
```

Without the traceback, debugging a production failure requires reproducing the
error. With the traceback, the engineer sees the exact file, line, and call
stack without touching the running system.

### Celery soft_time_limit=600

```python
@celery_app.task(soft_time_limit=600, ...)
def ingest_document(...):
```

`soft_time_limit` raises a `SoftTimeLimitExceeded` exception inside the task
after 600 seconds (10 minutes). The task can catch it and clean up gracefully
before the hard kill signal arrives.

Without this, a hung task (network call that never times out, infinite loop in
parser) holds a worker slot forever. Every worker eventually hangs, the queue
backs up, and the entire ingestion system stops processing new documents.

600 seconds is generous — normal ingestion takes 30-120 seconds per document.
A task running past 10 minutes has almost certainly hung.

### acks_late + reject_on_worker_lost — Task Survives Worker Death

```python
task_acks_late=True,
task_reject_on_worker_lost=True,
```

Default Celery behavior: a task is acknowledged (removed from queue) the moment
a worker picks it up. If that worker dies mid-processing (OOM, SIGKILL, EC2
instance terminated), the task disappears silently. The document is never ingested
and nobody knows.

`acks_late=True`: the task stays in the queue until the function returns
successfully. If the worker dies, the task goes back to the queue and another
worker picks it up.

`reject_on_worker_lost=True`: if the worker is killed by SIGKILL (not a graceful
shutdown), the task is explicitly rejected back to the queue rather than
silently lost.

At Discover scale, ECS terminates instances during scale-down events. Without
these settings, every scale-down event silently loses whatever documents were
in-flight.

### prefetch_multiplier=1 — No Task Hoarding

```python
worker_prefetch_multiplier=1,
```

Default Celery behavior: each worker pre-fetches 4 tasks from the queue.
On long-running ingestion jobs (10-K filing takes 90 seconds), one fast worker
grabs 4 tasks and holds them while other workers sit idle waiting for the queue
to refill. Queue appears empty but 3 tasks are just sitting in memory on one worker.

`prefetch_multiplier=1`: each worker holds exactly 1 task. The queue reflects
true backlog. Workers pull new tasks as soon as they finish. At 100 workers
processing 100-second jobs, this doubles effective throughput.

### JSON Serializer — Not Pickle

```python
task_serializer="json",
accept_content=["json"],
```

Celery's default serializer is pickle. Pickle deserializes arbitrary Python
objects — including executable code. A malicious message injected into the Redis
queue can execute arbitrary code on every worker process.

JSON serializes only data. A malicious message is just invalid JSON and is
rejected. At financial institution scale, the Redis queue is a potential attack
surface — this eliminates an entire class of code execution vulnerabilities.

### visibility_timeout=3600 > time_limit=660

```python
broker_transport_options={"visibility_timeout": 3600},
soft_time_limit=600,
time_limit=660,
```

Redis has a visibility timeout: if a task isn't acknowledged within that window,
Redis assumes the worker died and re-queues the task. With `acks_late=True`,
the task isn't acknowledged until it finishes.

If `visibility_timeout` (3600s) were less than `time_limit` (660s), Redis would
re-queue a still-running task before it finishes. Two workers would then process
the same document simultaneously, writing duplicate chunks to Qdrant.

The rule: `visibility_timeout` must always exceed `time_limit`.
3600s > 660s — the running task is always acknowledged before Redis can re-queue it.

### Streaming SHA-256 — Memory Safety for Large Files

```python
for block in iter(lambda: f.read(65536), b""):
    h.update(block)
```

Reading a file into memory to hash it fails on large documents. A 500MB PDF
loaded into memory for hashing exhausts RAM on a worker with 2GB allocated,
triggering OOM and losing the task.

64KB streaming blocks: the file is read and hashed in chunks. Memory usage
stays constant at ~64KB regardless of file size. Works on 500MB PDFs, 2GB
Excel files, anything.

### connect_timeout=5 — Fail Fast on Database Connections

```python
psycopg2.connect(url, connect_timeout=5)
```

Without a timeout, a `psycopg2.connect()` call to an unavailable PostgreSQL
server hangs indefinitely. Every worker trying to submit or check a document
blocks forever. The entire ingestion system freezes while waiting for a DB
that may never respond.

5 seconds: fast enough that a brief DB hiccup doesn't trip it, short enough
that a real outage is detected and logged within seconds instead of minutes.

### Double-Checked Locking on DLQ Table Creation

```python
_dlq_ready = False
_dlq_ready_lock = threading.Lock()

def _ensure_dlq_once(conn):
    global _dlq_ready
    if not _dlq_ready:
        with _dlq_ready_lock:
            if not _dlq_ready:   # second check inside the lock
                cur.execute("CREATE TABLE IF NOT EXISTS failed_documents ...")
                _dlq_ready = True
```

When 100 workers all fail simultaneously (e.g., Qdrant goes down), all 100 try
to create the DLQ table at the same moment. Without the lock, 100 concurrent
`CREATE TABLE` statements race. PostgreSQL handles this but wastes connections.

The double-check pattern: the outer `if not _dlq_ready` avoids acquiring the
lock on every call (fast path). The inner check inside the lock prevents a race
between two threads that both passed the outer check simultaneously (safe path).
The table is created exactly once per worker process.

### Celery + Redis Over Threading

Python threading cannot parallelize CPU-bound work (GIL). At 10M documents/day,
a single-process ingestion pipeline is the bottleneck.

Celery distributes work across any number of worker processes — on a single
machine today, across a fleet of ECS containers in Phase B. The same task code
runs everywhere. Only the broker URL changes (Redis → SQS).

### Hash Gate Before Queue

Every `submit_document()` call computes the SHA-256 of the file content and
checks `ingestion_status` before enqueuing a Celery task.

A document already marked `complete` never enters the queue. This prevents:
- Re-processing on a re-run of the pipeline script
- Duplicate processing when two workers receive the same file
- Wasted Bedrock embedding cost on already-indexed content

### DLQ in PostgreSQL, Not Redis

Failed documents land in a `failed_documents` PostgreSQL table, not a Redis
queue.

Redis is ephemeral. A failed document record in Redis can disappear on restart
or eviction. PostgreSQL is durable — failed documents are auditable, inspectable,
and retryable at any future time.

This matters for compliance: financial institutions must be able to demonstrate
that every submitted document was either successfully processed or explicitly
failed with a traceable error.
