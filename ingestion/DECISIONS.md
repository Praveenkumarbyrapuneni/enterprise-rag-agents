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

### wait=False on Upsert

`wait=False` tells Qdrant to acknowledge the write before updating the HNSW
index. This increases write throughput significantly.

Safety: PostgreSQL `ingestion_status` is the source of truth. If Qdrant crashes
between write acknowledgement and index commit, status stays `processing`.
The next run's Phase 1 detects this and re-uploads from scratch. No data loss.

---

## Orchestrator

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
