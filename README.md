# Enterprise RAG + Multi-Agent System

Production-grade financial document intelligence. Ingests earnings reports, SEC filings, and analyst reports at scale — answers complex questions with cited sources, cross-document reasoning, and self-scored retrieval quality.

Designed for Discover/JPMorgan-scale: 10M+ documents, millions of daily ingestions, concurrent workers, zero data loss.

---

## Quick Start

```bash
# 1. Start infrastructure
docker compose up -d

# 2. Configure environment
cp .env.example .env
# Fill in AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start workers (3 separate terminals)
celery -A ingestion.orchestrator.celery_app worker --queues=ingestion --loglevel=info
celery -A ingestion.orchestrator.celery_app worker --queues=dlq --loglevel=info --hostname=dlq@%h
celery -A ingestion.orchestrator.celery_app flower --port=5555

# 5. Run the pipeline — edit file paths in run_pipeline.py then:
python run_pipeline.py

# Monitor at http://localhost:5555 (Flower) and http://localhost:6333/dashboard (Qdrant)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                           │
│                                                                 │
│  Files arrive (PDF, DOCX, Excel, CSV, HTML, image, TIFF, email)│
│       ↓                                                         │
│  Hash check → PostgreSQL — duplicate? skip entirely            │
│       ↓                                                         │
│  Task queue → Celery + Redis (SQS on AWS)                       │
│       ↓                                                         │
│  Worker fleet — N parallel workers (ECS on AWS)                 │
│    Parser → Chunker → Embedder → Qdrant Uploader               │
│    Every chunk carries: file, page, chunk_index, hash, ts       │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI REST API                            │
└─────────────────────┬───────────────────────────────────────────┘
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
│      Qdrant (child chunks)  +  PostgreSQL (parent chunks)       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Ingestion Pipeline

| Component | What it does | Key design decisions |
|---|---|---|
| **Parser** `ingestion/parser.py` | Extracts clean text from 9 formats | N-column reading order; scanned pages → Claude Vision; inline images in reading order; DOCX text boxes + embedded charts; email CID images; password-protected file detection |
| **Chunker** `ingestion/chunker.py` | Routes to strategy by content, not file type | Fixed (512 tok, 50 overlap) for general docs; Hierarchical parent-child (1024/256 tok) for financial docs; Claude Haiku or local Ollama classifier |
| **Embedder** `ingestion/embedder.py` | Context-enriches then embeds | Claude Haiku via Bedrock for contextual retrieval (20 parallel workers); Cohere Embed v3 via Bedrock (1024 dims); exponential backoff; data never leaves AWS VPC |
| **Uploader** `ingestion/qdrant_uploader.py` | Idempotent write to Qdrant + PostgreSQL | hash + status in PostgreSQL prevents duplicate writes on retry; HNSW m=16/ef=200; TurboQuant 4-bit (8× memory reduction, ~1% recall loss); payload indexes before first write |

**Two-level storage for financial documents:**
- Child chunks (256 tokens) → Qdrant. Small = precise vector match.
- Parent chunks (1024 tokens) → PostgreSQL. Fetched by `parent_id` at query time to give the LLM full context.
- Fixed-strategy chunks (512 tokens) → Qdrant only. No parent needed.

---

## Intelligence Layer

| Agent | Role |
|---|---|
| **Query Analyzer** | Decomposes complex questions, detects cross-document queries, implements HyDE for better retrieval |
| **Retriever** | Hybrid dense+sparse search, Cohere reranker (+30–40% precision), MMR for diversity |
| **Synthesizer** | Grounded answers with citations; refuses to answer when context is absent |
| **Evaluator** | RAGAS scoring (faithfulness, relevance, recall); triggers retrieval retry when below threshold |

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Claude claude-sonnet-4-6 (Anthropic / Bedrock on AWS) |
| Context enrichment | Claude Haiku 4.5 via Bedrock (data stays in VPC) |
| Embeddings | Cohere Embed v3 via Bedrock (1024 dims) |
| Vector store | Qdrant — HNSW + TurboQuant 4-bit (Google Research, ICLR 2026) |
| Metadata + parents | PostgreSQL (Docker → RDS on AWS) |
| Task queue | Celery + Redis (Docker → SQS on AWS) |
| Agent orchestration | LangGraph |
| Re-ranking | Cohere Rerank |
| Evaluation | RAGAS |
| Logging (Phase A) | File logs → logs/rag.log |
| Logging (Phase B) | AWS CloudWatch (ECS ships stdout automatically) |
| Safety | Guardrails AI (PII redaction, prompt injection) |
| API | FastAPI |
| Monitoring | Prometheus + Grafana |

**AWS migration (Phase B):** Zero code changes. Only environment variables change — Docker → managed services (RDS, SQS, ECS, Kinesis, Bedrock, Cognito).

---

## Environment Variables

```bash
# AWS Bedrock — all AI calls route through here (data never leaves VPC)
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1
BEDROCK_HAIKU_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0

# Infrastructure (defaults work with docker compose up -d)
DATABASE_URL=postgresql://raguser:ragpassword@localhost:5432/ragdb
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Tuning
CHUNK_CLASSIFIER=haiku      # keyword | haiku
EMBEDDER_CONTEXT_MODE=haiku # haiku | skip
QDRANT_COLLECTION_NAME=documents
QDRANT_BATCH_SIZE=256
```

---

## Build Status

**Phase 1 — Ingestion** ✅ Complete
- [x] Infrastructure — Docker Compose (Qdrant + PostgreSQL + Redis)
- [x] Parser — 9 formats, 19 production gaps fixed
- [x] Chunker — fixed + hierarchical strategies, content-aware router, 8 gaps fixed
- [x] Embedder — Bedrock Cohere v3 (1024 dims), contextual retrieval via Haiku, 10 gaps fixed
- [x] Qdrant Uploader — idempotent writes, HNSW tuning, scalar quantization
- [x] Ingestion Orchestrator — Celery worker pool, hash gate, dead letter queue
- [x] End-to-end test — 3 SEC filings (Apple, Goldman Sachs, JPMorgan), 3,233 vectors, zero duplicates
- [x] AWS Bedrock migration — all AI calls route through Bedrock, data never leaves VPC

**Phase 2 — Agents** ✅ Complete
- [x] `agents/state.py` — shared TypedDict state flowing through all agents
- [x] `agents/logger.py` — file logging (Phase A), CloudWatch-ready (Phase B, zero code change)
- [x] `agents/query_analyzer.py` — Claude Haiku classifies query type, splits multi-part questions, generates HyDE query (conceptual only — never for numerical to prevent hallucination)
- [x] `agents/retriever.py` — dual-query dense search (question + HyDE), Cohere rerank, MMR deduplication, parent context expansion from PostgreSQL
- [x] `agents/synthesizer.py` — lost-in-middle chunk reordering, grounded answer with inline citations, cannot-answer guard
- [x] `agents/evaluator.py` — deterministic checks + Claude Haiku judge (custom, not RAGAS — RAGAS fails 83.5% on financial docs), IDK handling, confidence warning on max retries
- [x] `agents/graph.py` — LangGraph pipeline: analyze → retrieve → synthesize → evaluate → retry loop (max 1 retry, dedicated increment_retry node)
- [x] LangGraph Studio — visual pipeline dashboard via `langgraph dev`

**Phase 3 — API + End-to-End Testing** (FastAPI wrapper, real query testing)

**Phase 4 — AWS Migration** (RDS, SQS, ECS, CloudWatch — zero code changes, config only)
