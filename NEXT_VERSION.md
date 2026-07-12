# Next Version — Planned Upgrades

This file documents every known limitation and planned improvement with a specific
trigger condition for when to build each one. This is not a wishlist — every item
has a clear reason it was parked and a clear signal for when to pick it up.

---

## ✅ Completed (most recent first)

| Feature | Completed | Notes |
|---|---|---|
| Fine-tuning data export | 2026-07-11 | `scripts/export_finetune_data.py` — SFT / DPO / Reranker JSONL |
| Active feedback loop | 2026-07-10 | `POST /feedback` + Celery auto-tunes retrieval params per tenant |
| Semantic caching | 2026-07-10 | Qdrant collection, 0.92 cosine threshold, 1hr TTL, `⚡ cached` tag |
| Docker prod config | 2026-07-10 | `docker-compose.prod.yml` — pinned versions, health checks, resource limits |
| Hybrid search (dense + BM25) | 2026-07-03 | Cohere dense + BM25 sparse + RRF fusion in single Qdrant query |

---

## 🔲 What's Next (build in this order)

### 1. cited=True chunk marking in query_chunks ← build this first

**What:** When the synthesizer writes an answer, it cites specific sources. Those sources
map back to chunks in the `query_chunks` table. Right now all chunks are stored with
`cited=False`. We need to update the cited flag to `True` for chunks that actually
appear in the answer's source list.

**Why it matters:** The reranker training export (`data/reranker_train.jsonl`) uses
`cited=True` as positive training examples. Without this, the reranker dataset is
empty even if thousands of queries have been run.

**Where to implement:** `agents/synthesizer.py` — after writing the answer, extract
cited source names from the sources list and run:
```sql
UPDATE query_chunks SET cited = true
WHERE query_id = %s AND source = ANY(%s)
```

**Effort:** ~1 hour. One function, one SQL update.

---

### 2. Guardrails — PII detection + output validation

**What:** Two gaps before any enterprise demo:

*Input side — PII detection:*
Users sometimes paste personal data into questions ("What is the policy for SSN 123-45-6789?").
That SSN goes into the audit log, the cache, and possibly the LLM context. It must be
detected and redacted before processing.

```python
# Before passing question to the pipeline
question = redact_pii(question)  # mask SSN, credit card numbers, account numbers
```

Use `presidio-analyzer` (Microsoft, open source) — it detects 50+ PII entity types
with no external API call. Runs locally. Zero data leaves.

*Output side — answer validation:*
The synthesizer should never produce answers that contain raw PII from documents —
account numbers, personal identifiers, transaction IDs that reference other customers.
A second Presidio pass on the answer before returning it catches this.

**Why it's blocking:** Any financial institution demo will ask "how do you handle PII?"
A system with no answer to this question fails the security review immediately.

**Effort:** ~4 hours. Add `presidio-analyzer` to requirements, one redact function,
two call sites (before pipeline + after synthesizer).

---

### 3. Streaming responses (SSE)

**What:** Currently the full answer is generated before anything is returned to the caller.
For a 500-word answer this means 4-8 seconds of silence before the first word appears.
Streaming sends each word as it's generated — the user sees the answer building in real time.

**Why users expect it:** ChatGPT, Claude, Gemini all stream. Any demo against a non-streaming
system feels broken, even if the answer quality is better.

**How to implement:**
```python
# FastAPI SSE endpoint
@app.post("/query/stream")
async def query_stream(req: QueryRequest, user=Depends(_auth)):
    async def generate():
        async for chunk in synthesizer.stream(question, context):
            yield f"data: {json.dumps({'token': chunk})}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")
```

The cache check and retrieval stay synchronous. Only the synthesizer call becomes
a streaming Bedrock `invoke_model_with_response_stream` call.

**Effort:** ~1 day. New `/query/stream` endpoint alongside existing `/query`.
Existing endpoint stays unchanged — no breaking change.

---

### 4. One-command demo mode (Ollama local models)

**What:** Right now cloning this repo and running it requires:
- AWS account with Bedrock access
- Cohere API key
- Claude models enabled in Bedrock

That's a 45-minute setup for a senior engineer. For an enterprise evaluator or investor
who just wants to see it work, it's a blocker.

Demo mode replaces cloud APIs with local models via Ollama:
- Cohere Embed v3 → `nomic-embed-text` (Ollama, free, runs locally)
- Claude Sonnet → `llama3.1:8b` (Ollama, free, runs locally)
- Claude Haiku → `gemma3:4b` (Ollama, fast, runs locally)

```bash
# With demo mode:
git clone ...
cp .env.example .env       # fill in DATABASE_URL only
DEMO_MODE=true docker compose up
# Upload a PDF at localhost:8000/docs
# Ask a question
# Done — no AWS, no API keys
```

**Effort:** ~1 day. Add `DEMO_MODE` env var, swap provider clients based on flag,
add Ollama to docker-compose.yml as optional service.

---

### 5. GraphRAG — entity relationship graph on top of vector retrieval

**What:** Vector search finds semantically similar chunks. It cannot answer relationship
questions — "How are Goldman's Asia exposure risk factors connected to their subsidiary
holdings in Singapore?" That requires a knowledge graph.

GraphRAG extracts entities (companies, people, dates, financial metrics) and relationships
(owns, reports-to, affects, contradicts) from ingested documents and stores them as a
graph. Queries that need multi-hop reasoning traverse the graph first, then retrieve
supporting chunks.

**Why after the above items:** GraphRAG is the biggest quality upgrade remaining.
But it's wasted if the system doesn't have guardrails, streaming, and a clean demo flow.
Get those right first — then GraphRAG makes the product substantially stronger.

**Stack options:**
- Neo4j (most mature, native graph DB, Cypher query language)
- NetworkX (in-memory, no infra, fine for < 1M nodes)
- Amazon Neptune (AWS-native, Phase B option)

**Effort:** 1-2 weeks. Entity extraction (Claude Haiku), graph storage (Neo4j or NetworkX),
graph traversal node added to LangGraph pipeline, query router updated to detect
multi-hop questions.

---

## 🔲 Data-gated (build when training data is ready)

### Reranker fine-tuning

**Trigger:** `data/reranker_train.jsonl` has 200+ triples (query, pos_chunks, neg_chunks).

**What:** Replace Cohere's generic reranker with a cross-encoder fine-tuned on your
specific financial document domain. Expected improvement: 10-20% better retrieval
precision on domain-specific queries (exact financial terms, regulatory language).

**How:** Use `sentence-transformers` library, `cross-encoder/ms-marco-MiniLM-L-6-v2`
as base, fine-tune on exported triples. Training on an A40 GPU: ~2 hours, ~$3.

---

### Synthesizer fine-tuning (Qwen3 local model)

**Trigger:** `data/sft_synthesizer.jsonl` has 500+ examples.

**What:** Fine-tune Qwen3-7B (from `llm-fine-tuning` repo) on (question, answer) pairs
from helpful=True high-quality responses. Deploy as local synthesizer via vLLM.
Replace Claude Sonnet API calls for routine RAG queries — use Claude only for complex ones.

**Cost impact:** At 1M queries/day, Claude Sonnet costs ~$3,000/day. A local Qwen3
serving routine queries (est. 70%) cuts that to ~$900/day. ROI: ~$2.1M/month at scale.

**How:** QLoRA on RunPod A40, same pipeline as `llm-fine-tuning` repo. Add vLLM serving
container to docker-compose. Route by query complexity in the synthesizer.

---

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

## 4. Live Transaction Streaming Pipeline

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

## 8. AWS CloudWatch Logging (Phase B)

**What it is:**

Phase A (laptop) writes logs to `logs/rag.log` via file rotation.
Phase B (AWS ECS) must ship logs to CloudWatch so they are searchable,
alertable, and never lost when containers restart.

**Why data must stay in AWS:**
LangSmith (the obvious tracing alternative) is a SaaS — it sends queries,
documents, and answers to LangChain's servers. For financial institution clients
(Discover, JPMorgan), this violates data residency and compliance requirements.
CloudWatch keeps everything inside the AWS VPC. Zero data leaves.

**How to implement (zero code change):**

The logger already handles this. In `agents/logger.py`, set `LOG_OUTPUT=cloudwatch`
in the ECS task definition environment variables. The handler switches from file
to stdout, and ECS ships stdout to CloudWatch automatically via the awslogs driver.

Add this to every ECS task definition:
```json
"logConfiguration": {
  "logDriver": "awslogs",
  "options": {
    "awslogs-group": "/rag-agents/production",
    "awslogs-region": "us-east-1",
    "awslogs-stream-prefix": "rag"
  }
}
```

Set up CloudWatch alarms for: ERROR log count > 10/minute, retriever latency
p99 > 5s, evaluator faithfulness score < 0.85 sustained over 10 queries.

**When to implement:**
Phase B AWS migration. First thing before going live — without this you are
blind to production failures.

---

## 9. Nova Pro vs Claude Sonnet Model Benchmark

**What it is:**

We currently use Claude Sonnet 4.6 for the synthesizer and evaluator agents.
Amazon Nova Pro ($0.80/$3.20 per million tokens) is 4x cheaper than Sonnet
($3/$15) and is positioned by AWS for complex agentic RAG tasks.

Nova Pro scores 10-12 MMLU points lower than Sonnet. Whether this matters
for our specific use case (financial document Q&A) is unknown until measured.

**Why not switched yet:**
For a financial institution client, a wrong answer about revenue, risk, or
compliance can cause regulatory violations. We cannot guess — we must measure.

**How to implement:**
1. After Phase 2 agents are complete, run 50 representative financial queries
2. Run through pipeline with Sonnet → record RAGAS faithfulness + relevance scores
3. Swap synthesizer model to Nova Pro (one env var change: `BEDROCK_SYNTHESIS_MODEL`)
4. Run the same 50 queries → record RAGAS scores
5. If Nova Pro faithfulness ≥ 0.85 and relevance ≥ 0.80 → switch permanently

Model env vars to make this a config change only (no code edits):
- `BEDROCK_SYNTHESIS_MODEL` — synthesizer agent
- `BEDROCK_EVAL_MODEL` — evaluator agent
- `BEDROCK_HAIKU_MODEL_ID` — already exists for query analyzer + context enrichment

**When to implement:**
After evaluator agent is built and RAGAS scoring is working. Run the benchmark
before going to a client — cost savings of 4x are significant at Discover scale
(millions of queries per day).

---

## 10. OKF — Open Knowledge Format (Google, June 2026)

**What it is:**

OKF (Open Knowledge Format) is a Google Cloud specification published June 12, 2026.
It stores curated, pre-verified facts as a directory of markdown files with YAML
frontmatter. Unlike RAG (which re-derives knowledge from raw chunks every query),
OKF stores facts that are ALWAYS true and never need to be retrieved from documents.

For our financial RAG system, OKF would contain:
- Financial term definitions (EPS, EBITDA, Tier 1 Capital, LIBOR, Basel III)
- Company metadata (Apple's fiscal year ends September, Goldman Sachs ticker = GS)
- Metric schemas (how to calculate revenue growth, net interest margin)
- Regulatory definitions (SOX, PCI DSS, Dodd-Frank key terms)

**How it plugs into our system:**

OKF becomes a FIRST-CHECK in the query analyzer (Agent 1). Before the retriever
searches Qdrant, the query analyzer checks if the question is answered by a verified
OKF fact. If yes — return directly. No Qdrant search, no Claude synthesis, no
hallucination risk.

```
User question
     ↓
[Query Analyzer]
  → Check OKF first (instant, no vector search)
  → If found: return verified fact directly
  → If not found: proceed to Retriever → Synthesizer → Evaluator
```

**File structure:**
```
knowledge/
├── terms/
│   ├── eps.md           (type: definition, title: Earnings Per Share)
│   ├── ebitda.md
│   └── tier1_capital.md
├── companies/
│   ├── apple.md         (fiscal year ends September, ticker AAPL)
│   └── goldman.md
└── regulations/
    ├── libor.md
    └── basel3.md
```

**YAML frontmatter format (OKF v0.1 spec):**
```yaml
---
type: definition
title: Earnings Per Share (EPS)
tags: [financial-metric, earnings, profitability]
timestamp: 2026-06-30
---
EPS = Net Income / Weighted Average Shares Outstanding.
Diluted EPS includes stock options and convertible securities.
```

**Why not built yet:**
1. Content problem: the markdown files need to be written. Code to read them is 20
   lines. The financial definitions take hours to write correctly.
2. Can't test without a complete pipeline. The RAGAS evaluator will show us exactly
   WHERE the system gives wrong answers. Build OKF to fix those specific gaps.
3. OKF v0.1 is 18 days old (published June 12, 2026). No production financial system
   is using it yet. Let the spec mature slightly before building on it.

**When to implement:**
After the full pipeline (Agents 1-4 + graph.py) is complete and running end-to-end.
Run 50 test questions. Wherever the evaluator scores faithfulness < 0.85 on questions
about known definitions or company metadata — those are the OKF files to write first.

**Implementation steps (when ready):**
1. Create `knowledge/` directory at project root
2. Write markdown files for the 20 most common financial terms in your test queries
3. Add `_check_okf(question)` function to query_analyzer.py (~20 lines):
   - Embed the question
   - Compare against pre-embedded OKF fact titles (cosine similarity)
   - If score > 0.95 (high confidence it's a known fact) → return OKF content
   - Else → proceed to Qdrant retrieval
4. Pre-embed all OKF titles at startup (not per-query) — cache in memory

---

## Summary Table

| Upgrade | Status | Trigger to Implement |
|---|---|---|
| PDF vector graphics extraction | 🔲 Planned | Users report missing chart data from specific PDF types |
| Unlimited-OCR (self-hosted OCR) | 🔲 Planned | Claude Vision API monthly cost exceeds GPU instance cost |
| Late chunking | 🔲 Planned | RAGAS recall fails on cross-reference-heavy documents |
| ~~Hybrid search (dense + sparse)~~ | ✅ Done | Implemented — BM25 + RRF (2026-07-03) |
| ~~Semantic caching~~ | ✅ Done | Implemented — Qdrant collection, 0.92 threshold (2026-07-10) |
| ~~Active feedback loop~~ | ✅ Done | Implemented — POST /feedback + Celery tuner (2026-07-10) |
| ~~Docker prod config~~ | ✅ Done | Implemented — docker-compose.prod.yml (2026-07-10) |
| ~~Fine-tuning data export~~ | ✅ Done | Implemented — SFT / DPO / Reranker JSONL (2026-07-11) |
| GraphRAG | 🔲 Next | Add after guardrails + streaming are done |
| Guardrails (PII + injection) | 🔲 Next | Before first enterprise demo |
| Streaming responses (SSE) | 🔲 Next | Before first enterprise demo |
| One-command demo mode | 🔲 Next | Before sharing repo with enterprise evaluators |
| cited=True chunk marking | 🔲 Next | Required for reranker training data to work |
| Reranker fine-tuning | 🔲 Data-gated | After 200+ reranker training triples exported |
| Synthesizer fine-tuning (Qwen3) | 🔲 Data-gated | After 500+ SFT examples exported |
| Live transaction streaming | 🔲 Phase B | After batch pipeline stable on AWS |
| Semantic chunking (optional) | 🔲 Planned | Demand for single-topic academic documents |
| HNSW index tuning | 🔲 Phase B | RAGAS context recall < 0.80 at 10M documents |
| CloudWatch logging | 🔲 Phase B | First thing before going live on AWS |
| Nova Pro model benchmark | 🔲 Phase B | After RAGAS scoring works end-to-end |
| OKF (Google Open Knowledge Format) | 🔲 Planned | After RAGAS shows gaps on known financial facts |
| Redis → SQS | 🔲 Phase B | AWS migration |
| PostgreSQL Docker → RDS | 🔲 Phase B | AWS migration |
| Celery workers → ECS Fargate | 🔲 Phase B | AWS migration |
| Qdrant Docker → Qdrant Cloud | 🔲 Phase B | AWS migration |
| Local files → S3 | 🔲 Phase B | AWS migration |
| FastAPI local → ECS + ALB | 🔲 Phase B | AWS migration |
| Batch pipeline → Kinesis streaming | 🔲 Phase B | After batch pipeline stable on AWS |

---

## Phase B — Full AWS Migration Checklist

Everything below is a config-only change. No Python code rewrites.
The only things that change are environment variables and infrastructure.

### Step 1 — Replace Redis with SQS
**What:** Celery currently uses Redis as its message broker (task queue).
On AWS, Redis → Amazon SQS. SQS is infinite scale, never loses a message,
no single point of failure.

**Code change:** Zero. One env var:
```
CELERY_BROKER_URL=sqs://  (currently redis://localhost:6379)
```
Install `celery[sqs]` — already planned in requirements. SQS queue names
must match what Celery expects: `ingestion` and `dlq`.

**SQS setup:**
- Create two SQS queues: `rag-ingestion` and `rag-dlq`
- Set visibility timeout = 10 minutes (longer than max task runtime)
- Enable dead-letter queue on `rag-ingestion` → points to `rag-dlq`
- IAM role on ECS task must have `sqs:SendMessage`, `sqs:ReceiveMessage`,
  `sqs:DeleteMessage`, `sqs:GetQueueAttributes`

---

### Step 2 — Replace PostgreSQL Docker with RDS
**What:** PostgreSQL currently runs in a Docker container on laptop.
On AWS → Amazon RDS PostgreSQL (Multi-AZ for production, single-AZ for staging).

**Code change:** Zero. One env var:
```
DATABASE_URL=postgresql://user:pass@rds-endpoint:5432/ragdb
```
RDS runs the same PostgreSQL version. Same schema. Same queries. Identical.

**RDS setup:**
- Engine: PostgreSQL 15+
- Instance: db.t3.medium for staging, db.r6g.large for production
- Multi-AZ: enabled (automatic failover)
- Storage: 100GB gp3, autoscaling enabled
- VPC: same VPC as ECS cluster — no public access
- Security group: allow port 5432 from ECS security group only

---

### Step 3 — Replace Celery local workers with ECS
**What:** Celery workers currently run as terminal processes on laptop.
On AWS → ECS Fargate tasks (serverless containers, auto-scaling).

**Code change:** Zero. The Celery worker command stays identical:
```
celery -A ingestion.orchestrator.celery_app worker --queues=ingestion
```
ECS runs this inside a container. Auto Scaling Group scales worker count
based on SQS queue depth (CloudWatch metric: `ApproximateNumberOfMessagesVisible`).

**ECS setup:**
- Task definition: same Docker image as laptop
- CPU: 2 vCPU, Memory: 4GB per worker task
- Auto scaling: min 2 tasks, max 50 tasks
- Scale out: queue depth > 100 messages → add workers
- Scale in: queue depth = 0 for 5 minutes → remove workers
- IAM task role: Bedrock, S3, SQS, RDS access (least privilege)
- CloudWatch log group: `/rag-agents/workers`

---

### Step 4 — Replace Qdrant Docker with Qdrant Cloud or EC2 cluster
**What:** Qdrant currently runs in a Docker container on laptop.
On AWS → Qdrant Cloud (managed) or self-hosted Qdrant on EC2.

**Code change:** Zero. Two env vars:
```
QDRANT_HOST=your-qdrant-cloud-endpoint
QDRANT_PORT=6333
QDRANT_API_KEY=your-key  (if using Qdrant Cloud)
```

**Two options:**
- **Qdrant Cloud (recommended):** Managed service, automatic backups, scaling.
  Runs inside AWS region — data stays within AWS. Cost: ~$0.05/GB/month.
- **Self-hosted EC2:** More control, lower cost at large scale.
  Use r6g.2xlarge (memory-optimised) — TurboQuant keeps vectors in RAM.
  At 10M vectors × 1024 dims × 4-bit = ~5GB. r6g.large is sufficient.
  Enable EBS snapshot backups daily.

**Re-ingestion required:** Current collection has dense vectors only.
Phase B is the right time to add sparse vectors for hybrid BM25 search
(see item 4 in this file). Re-ingest with both dense + sparse vectors.
Use Kinesis stream to re-ingest from S3 source documents.

---

### Step 5 — Replace local file storage with S3
**What:** Source documents currently sit in the `tests/` folder on laptop.
On AWS → S3 bucket. The ingestion pipeline reads from S3 instead of disk.

**Code change:** Minimal. The orchestrator's `submit_document()` function
currently takes a file path. On AWS it takes an S3 key instead.
One function change in `ingestion/orchestrator.py` — the rest of the pipeline
(parser, chunker, embedder, uploader) is unchanged.

**S3 setup:**
- Bucket: `rag-source-documents-{account-id}` (globally unique name)
- Block all public access: enabled
- Server-side encryption: SSE-KMS (not SSE-S3) — requirement for financial data
- Versioning: enabled — never lose a source document
- Lifecycle policy: move to S3 Glacier after 2 years (cost optimisation)
- IAM: ECS task role has `s3:GetObject` on this bucket only

---

### Step 6 — Replace FastAPI local with ECS + API Gateway
**What:** FastAPI currently runs locally (`uvicorn main:app`).
On AWS → FastAPI in ECS container behind API Gateway (or ALB).

**Code change:** Zero. The FastAPI app code is identical.
Container runs `uvicorn agents.api:app --host 0.0.0.0 --port 8080`.

**Setup:**
- ECS service: 2 tasks minimum, behind Application Load Balancer
- API Gateway: optional (adds auth, rate limiting, API keys per client)
- HTTPS: ACM certificate on ALB — HTTP redirects to HTTPS
- WAF: AWS WAF on API Gateway — blocks SQL injection, XSS, rate abuse
- Target group health check: `GET /health` → 200

---

### Step 7 — Add Kinesis Streaming Pipeline
**What:** Batch ingestion handles historical documents.
Kinesis handles live transactions that must be searchable within seconds.

See item 5 in this file for full details.
Implement after batch pipeline is stable on AWS.

---

### Phase B — Migration Order (Do Not Skip Steps)

```
1. RDS first          → data layer must exist before workers start
2. SQS second         → queue must exist before workers connect
3. S3 third           → source documents uploaded before re-ingestion
4. ECS workers fourth → connect to RDS + SQS + S3 + Bedrock
5. Qdrant Cloud fifth → re-ingest all documents with dense + sparse vectors
6. ECS API sixth      → connect to Qdrant + RDS, expose via ALB
7. CloudWatch seventh → verify all logs flowing before going live
8. Kinesis last       → only after batch pipeline proven stable
```

**The golden rule:** Each step is just changing env vars in the ECS task
definition. The Python code never changes. If you find yourself editing
Python files during Phase B migration, stop — something is wrong with
the architecture decision made in Phase A.
