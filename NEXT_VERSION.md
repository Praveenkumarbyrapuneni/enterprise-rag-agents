# Version 2 — Planned Upgrades

This file documents every known limitation and improvement that was deliberately
excluded from Version 1. Each item was evaluated, understood, and parked with a
clear reason. This is not a wishlist — every item here has a specific trigger
condition that tells you exactly when to implement it.

---

## 1. PDF Vector Graphics Extraction

**What the problem is:**

Some PDF charts are drawn using mathematical path instructions — `moveto`,
`lineto`, `curveto`, `fill`. These are not images embedded in the file. They
are drawing commands that the PDF viewer executes to paint the chart on screen.

Our parser cannot see these. Text extraction finds no text. Image extraction
finds no embedded image. The chart is completely invisible to the pipeline.

This affects PDFs exported from Excel, PowerPoint, or design tools where charts
become vector objects rather than raster images. It does not affect scanned pages
(those are handled by the scanned-page fallback) — only pages that have readable
text alongside a vector-drawn chart.

**What the fix looks like:**

PyMuPDF exposes `page.get_drawings()` which returns all vector drawing commands
on a page. If a page has significant drawing content but the extracted text does
not mention numerical data that matches the drawing region, that region should be
selectively rendered as a bitmap and sent to Claude Vision.

```python
drawings = page.get_drawings()
if drawings and chart_region_detected(drawings):
    clip = fitz.Rect(drawing_bbox)
    pix = page.get_pixmap(clip=clip, dpi=150)
    chart_text = _extract_image_text(pix.tobytes("png"), "image/png")
```

**Why not in Version 1:**

Rendering every drawing region on every page adds significant compute per
document. The detection logic (distinguishing chart drawings from decorative
borders, lines, and page elements) requires calibration across many document
types. At Version 1 scope, the priority was correctness on the common cases.
This affects a minority of documents — primarily PDFs exported from design tools,
not natively created PDFs.

**When to implement:**

When users report missing chart data from specific PDF types, or when benchmark
testing on real financial PDFs shows meaningful content loss from this gap.

---

## 2. Self-Hosted OCR with Unlimited-OCR (Baidu)

**What it is:**

`baidu/Unlimited-OCR` (released June 2026, 11,000+ stars) is a neural
vision-language model that parses entire document pages in one pass. It was
built to push beyond DeepSeek-OCR and handles complex layouts, multi-column
text, tables, and mixed content natively.

**Where it replaces our current approach:**

Currently, scanned PDF pages (pages with no text layer) are rendered as bitmaps
and sent to the Claude Vision API. At Discover-scale — millions of scanned pages
per day — this generates significant API costs.

Unlimited-OCR, self-hosted on GPU instances, could replace these Claude Vision
calls entirely. The model runs inference locally, eliminating per-call API costs
at the expense of GPU infrastructure.

**What it does NOT replace:**

All structured extraction logic in our parser — Excel chart XML parsing, DOCX
text box extraction, email MIME resolution, HTML DOM walking. Those require
rule-based code that reads internal file structure. A vision model that reads
rendered page images cannot access file internals.

**Why not in Version 1:**

- Requires NVIDIA GPU (`model.eval().cuda()`). Not available during laptop development.
- Dependencies: torch 2.10.0, torchvision, transformers — adds 5+ GB to the environment.
- Adds GPU infrastructure management: model loading, memory, inference server.
- Claude Vision API already handles scanned pages correctly in Version 1.
- Cost benefit only materialises at very high scanned-page volume.

**When to implement:**

Phase B (AWS). After measuring actual Claude Vision API costs at production
volume. If GPU instance cost (AWS p3.2xlarge or g4dn.xlarge) is lower than
monthly Vision API spend for scanned pages, replace the scanned-page fallback
in `parse_pdf()` with an Unlimited-OCR inference call.

Implementation path: deploy Unlimited-OCR as a sidecar container on ECS,
expose a `/parse-image` endpoint, replace `_extract_image_text()` calls for
scanned pages to hit the sidecar instead of the Anthropic API.

---

## 3. Late Chunking

**What it is:**

All current chunking strategies (fixed-size, sentence-aware, hierarchical) chunk
text before embedding. Each chunk is then embedded in isolation — the model
cannot see the surrounding context when computing the vector.

Late chunking inverts this order:

1. Feed the entire document to a long-context embedding model.
2. The model produces one vector per token, computed with full document context.
3. Average the token vectors within each chunk's range to produce chunk vectors.

The result: the vector for "this growth rate exceeded records" was computed when
the model had already seen "revenue grew 23% in North America." The reference is
resolved. The vector encodes the full meaning, not the isolated fragment.

**Why it produces better retrieval:**

When a chunk contains pronouns or references that point to content in a prior
chunk ("this", "it", "the above figure"), traditional embedding produces a weak
vector because the reference is unresolved. Late chunking resolves all references
at embedding time because the model sees the full document.

**Why not in Version 1:**

- Requires models that expose token-level embeddings (jina-embeddings-v3,
  nomic-embed-text). OpenAI's `text-embedding-3-large` — our current embedder —
  does not expose token-level vectors. The entire embedder would need to change.
- Embedding 50,000-token documents at once is significantly more expensive than
  embedding 512-token chunks. At 10 million documents per day, the cost difference
  is not yet justified against the retrieval quality improvement.
- Infrastructure is more complex: you cannot call the API with a text string,
  you must collect token vectors and pool them in custom code.
- Not yet production-standard at any major company. Still primarily a research technique.

**When to implement:**

When RAGAS evaluation scores on Version 1 show a systematic retrieval failure
on documents with heavy cross-reference language (legal contracts, technical
specifications). At that point, switch the embedder to jina-embeddings-v3 and
implement the token pooling pipeline. Measure RAGAS improvement before committing
the infrastructure cost.

---

## 4. Hybrid Search (Dense + Sparse Vectors)

**What it is:**

Version 1 uses dense vector search — the query is embedded into a vector and
Qdrant finds the nearest chunk vectors by cosine similarity. This is semantic
search: it finds conceptually similar content even if the exact words differ.

Dense search has a known weakness: exact keyword matching. If a user asks about
"Section 4.2(b)" or "CUSIP 037833100" or a specific regulation code, the semantic
vector for that query may not find the exact chunk that contains that string,
because the model has never learned that this specific code is meaningful.

Hybrid search combines:
- **Dense vectors** (what we have) — semantic similarity
- **Sparse vectors / BM25** — exact keyword matching, like a search engine

The two scores are combined (reciprocal rank fusion) to produce a final ranking
that handles both "find me content about revenue growth" and "find the exact
clause numbered 4.2(b)."

**Why not in Version 1:**

Qdrant supports sparse vectors natively but requires a separate sparse vector
index alongside the dense index. The ingestion pipeline must compute BM25 weights
for every chunk at upload time. The retrieval agent must run two searches and
fuse the results. This doubles retrieval complexity for a gain that only
materialises on queries with specific identifiers or codes — which are a minority
of financial queries but an important minority.

**When to implement:**

After RAGAS evaluation on Version 1. If context recall scores are low on
queries that contain specific codes, clause numbers, or identifiers, add
hybrid search. This is a targeted fix for a specific retrieval failure mode,
not a general improvement.

---

## 5. Live Transaction Streaming Pipeline

**What it is:**

Version 1 processes documents in batch — files arrive, go into a queue, workers
process them. This works for historical documents and bulk ingestion.

Financial institutions also generate millions of live transactions per day that
must be queryable within seconds of occurrence, not hours. A batch pipeline
cannot provide this latency.

The streaming pipeline:

```
Transaction occurs
      ↓
AWS Kinesis Data Stream (event published immediately)
      ↓
AWS Lambda consumers (triggered per event, parse + chunk + embed)
      ↓
Qdrant cluster (vector written within seconds)
      ↓
Queryable in under 60 seconds from transaction occurrence
```

**Why not in Version 1:**

Streaming is an AWS-only feature — there is no local equivalent of Kinesis that
runs identically in Docker without complexity. Building and testing this requires
AWS infrastructure. All of Phase A (laptop development) processes documents in
batch, which is sufficient for testing correctness and quality. Streaming is
a Phase B requirement.

**When to implement:**

Phase B AWS migration. After the batch pipeline is deployed and stable on AWS,
add the Kinesis stream and Lambda consumers as a separate service that writes
to the same Qdrant cluster under a different collection. The batch and stream
pipelines coexist — batch for historical documents, stream for live transactions.

---

## 6. Semantic Chunking as an Optional Third Strategy

**What it is:**

Semantic chunking groups sentences by meaning similarity rather than cutting at
fixed token counts. When the topic shifts (detected by a drop in cosine similarity
between consecutive sentence embeddings), a new chunk starts.

**Why not in Version 1:**

- 10x slower than fixed-size chunking — every sentence must be embedded during
  ingestion. At millions of documents per day this is a pipeline bottleneck.
- Fragile on financial documents that naturally mix multiple topics per paragraph
  (revenue figures alongside employee counts alongside risk factors).
- Still subject to the token ceiling — if a single topic runs for 3,000 tokens,
  the chunker must cut mid-topic anyway, making the semantic detection pointless.
- Version 1's sentence-aware fixed-size strategy produces cleaner results on
  financial documents at a fraction of the compute cost.

**When to implement:**

As an optional third strategy in the chunker router, selectable per document
type. Consider enabling it for highly structured single-topic documents (academic
research papers, technical whitepapers) where the topic-boundary detection is
reliable and the 10x ingestion cost is acceptable for the quality gain.

---

## 7. HNSW Index Tuning for 10M+ Document Scale

**What it is:**

Qdrant uses the HNSW (Hierarchical Navigable Small World) algorithm for vector
search. HNSW has two key parameters:

- `m` — number of connections per node in the graph. Higher = better recall,
  more memory.
- `ef_construction` — search depth during index build. Higher = better quality
  index, slower ingestion.

Version 1 uses Qdrant defaults (`m=16`, `ef_construction=100`). These defaults
are calibrated for general use, not for a 10-million-vector financial document
corpus with high-precision recall requirements.

At 10M vectors, suboptimal HNSW parameters can mean the difference between 95%
recall and 85% recall on retrieval — meaning 15% of the time the system fails
to find the correct answer even though it is in the database.

**Why not in Version 1:**

HNSW tuning requires the full corpus to be indexed before meaningful benchmarking.
You cannot optimise parameters on 1,000 test documents and apply them to 10 million —
the graph structure changes at scale. Tuning belongs after Phase A testing
establishes the baseline recall numbers.

**When to implement:**

Phase B, after ingesting the full document corpus. Run RAGAS context recall
benchmarks at scale. If recall falls below 0.80 target, experiment with
`m=32`, `ef_construction=200` and measure the recall vs ingestion speed tradeoff.
Qdrant documentation provides the full tuning reference.

---

## Summary Table

| Upgrade | Trigger to Implement |
|---|---|
| PDF vector graphics extraction | Users report missing chart data from specific PDF types |
| Unlimited-OCR (self-hosted OCR) | Claude Vision API monthly cost exceeds GPU instance cost at production volume |
| Late chunking | RAGAS recall fails systematically on documents with cross-reference language |
| Hybrid search (dense + sparse) | RAGAS recall fails on queries with specific identifiers, codes, clause numbers |
| Live transaction streaming | Phase B AWS deployment — after batch pipeline is stable |
| Semantic chunking (optional) | Demand for processing single-topic academic / technical documents |
| HNSW index tuning | RAGAS context recall drops below 0.80 at full 10M document scale |
