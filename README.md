# Enterprise RAG System

A production-grade document intelligence platform built for financial institutions.
Users log in, ask questions in plain English, and get grounded answers from both
their live transaction data and their institution's policy documents — with source citations.

Retrieval uses **hybrid dense + BM25 sparse search** fused via Reciprocal Rank Fusion:
dense vectors for semantic understanding, sparse vectors for exact financial term matching
("SOFR rate", "Form 10-Q", specific clause numbers). Both run in a single Qdrant query.

Answers are **semantically cached** — identical or near-identical questions served from
cache in milliseconds. Every response is **feedback-rated** — thumbs up/down signals
automatically tune retrieval parameters per tenant over time.

Designed to scale to 10M+ documents, millions of daily transactions, and millions of users.

---

## What it does

**Ingest any financial document** → PDF, Word, Excel, HTML, images, emails — parsed, chunked, and stored as searchable vectors.

**Answer questions in three ways automatically:**

| Query type | Example question | How it answers |
|---|---|---|
| Document search | "What are the biggest risk factors in the annual report?" | Searches ingested documents, returns cited answer |
| Live transaction data | "What is my current balance?" | Queries the transaction database directly |
| Hybrid | "Why was I charged a foreign fee on June 26?" | Combines the transaction record + the fee policy document |

**Self-evaluates every answer** → scores faithfulness and relevance, retries automatically if quality is low.

**Semantic cache** → identical or near-identical questions return cached answers tagged `⚡ cached` — skipping the full pipeline and saving cost.

**Active feedback loop** → every answer gets a `query_id`. Users submit thumbs up/down via `POST /feedback`. After every 10 feedbacks, a Celery task automatically tunes retrieval parameters (top_k, rerank_top_n, mmr_final_k) per tenant — the system gets measurably better over time without retraining.

**Fine-tuning data export** → once enough feedback accumulates, one command exports three JSONL datasets ready for training: SFT (synthesizer), DPO (preference pairs), and reranker training data.

**Full tenant isolation** → each institution's users can only ever see their own data, enforced at the vector search level.

---

## Real-world use case

A customer opens their bank's mobile app and types:

> "Why was I charged a $340 fee on June 26th?"

Without this system, a human agent has to look up the transaction in one system, find the fee policy in another, connect the two, and explain it. That takes minutes and costs money at scale.

With this system:

```
User logs in → JWT token issued with their institution ID baked in
        ↓
Semantic cache checked first — if similar question answered recently, return instantly
        ↓
Question classified automatically: needs transaction data + policy document
        ↓
PostgreSQL: finds the June 26 transaction ($340, flagged, foreign merchant)
Qdrant:     finds the foreign transaction fee policy in the institution's documents
        ↓
Claude synthesizes both into a grounded answer with citations
        ↓
Answer returned in ~2 seconds. No human involved.
        ↓
User rates the answer → system learns and improves retrieval for this tenant
```

This scales to millions of daily questions across millions of users from multiple institutions — all isolated from each other.

---

## Authentication & Tenant Isolation

Every user belongs to exactly one institution (tenant). That identity is set at registration and baked into their JWT token at login.

```
POST /auth/register  →  create account, assign to an institution
POST /auth/login     →  returns a signed JWT (valid 24h)
POST /query          →  JWT decoded → institution extracted → all searches filtered to that institution only
POST /feedback       →  submit helpful/unhelpful rating on a query response
```

The caller never passes `institution_id` manually. It flows automatically from the token.
A user from Institution A physically cannot retrieve Institution B's documents — the Qdrant filter is enforced at the vector search level, not in application logic.

---

## How it works internally

### Ingestion pipeline

```
Document arrives (PDF / Word / Excel / HTML / image / email)
      ↓
Parser — extracts text in reading order, 9 formats, scanned pages via OCR
      ↓
Content classifier — financial document? → hierarchical chunking
                   — simple document?    → fixed-size chunking
      ↓
Chunker — hierarchical: 1024-token parents (PostgreSQL) + 256-token children (Qdrant)
        — fixed: 512-token sentence-aware windows
      ↓
Contextual enrichment — Claude Haiku adds 1-2 sentences of document context
                        to every chunk (20 parallel workers)
      ↓
Dual embedding — Cohere Embed v3 (1024-dim dense) + BM25 sparse vector
      ↓
Qdrant upload — HNSW m=16/ef=200 + TurboQuant 4-bit (8x smaller, ~1% recall loss)
              — idempotent: safe to re-run, deduplicates by file hash
              — tenant_id payload filter enforced on every stored point
```

### Query pipeline

```
User asks a question
      ↓
Semantic cache check — embedding similarity ≥ 0.92 + same tenant + created < 1hr ago?
  → HIT:  return cached answer tagged ⚡ cached (< 50ms)
  → MISS: continue below
      ↓
Query Analyzer (Claude Haiku) classifies: sql / rag / hybrid
      ↓
  sql    → PostgreSQL transaction lookup (exact data, no LLM synthesis)
  rag    → Qdrant hybrid search (dense + BM25, RRF fusion, tenant-filtered)
             → Cohere Rerank v3 → MMR dedup → parent context expansion
             → chunks logged to query_chunks table (for fine-tuning export)
  hybrid → both paths combined
      ↓
Synthesizer (Claude Sonnet) writes grounded answer with inline citations
      ↓
Evaluator (Claude Haiku) scores faithfulness + relevance → retries if below threshold
      ↓
Answer returned via FastAPI with query_id
      ↓
User submits POST /feedback { query_id, helpful: true/false }
      ↓
Every 10 feedbacks → Celery tunes retrieval params for this tenant
```

### Self-improvement loop

```
Feedback accumulates per tenant
        ↓
Celery task: reads last 50 feedbacks
        ↓
helpful_rate < 50% → increase top_k (retrieve more candidates)
helpful_rate > 80% → decrease top_k (it's working, reduce cost)
        ↓
Updated params saved to retrieval_params table
        ↓
Retriever reads tuned params on next query — no restart needed
        ↓
python -m scripts.export_finetune_data  (when enough data accumulated)
        ↓
data/sft_synthesizer.jsonl   → fine-tune Qwen3 as local synthesizer
data/dpo_pairs.jsonl         → DPO preference training
data/reranker_train.jsonl    → domain-specific reranker training
```

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| POST | `/auth/register` | None | Create user account under a tenant |
| POST | `/auth/login` | None | Returns signed JWT (valid 24h) |
| POST | `/query` | JWT | Run a question through the RAG pipeline |
| POST | `/feedback` | JWT | Submit helpful/unhelpful rating for a query |
| GET | `/health` | None | Liveness probe |
| GET | `/ready` | None | Readiness check (verifies Qdrant + PostgreSQL) |

### Query response fields

```json
{
  "query_id":     "uuid",      // submit to POST /feedback to rate this answer
  "answer":       "...",
  "sources":      ["..."],
  "query_type":   "rag",
  "data_source":  "rag",
  "faithfulness": 0.92,
  "relevance":    0.88,
  "latency_ms":   1847.3,
  "cached":       false        // true = served from semantic cache (⚡ tag in UI)
}
```

---

## Stack

| Component | Technology |
|---|---|
| LLM — synthesis | Claude Sonnet 4 via AWS Bedrock |
| LLM — classification, evaluation, enrichment | Claude Haiku 4.5 via AWS Bedrock |
| Dense embeddings | Cohere Embed v3 via Bedrock (1024 dims) |
| Sparse embeddings | BM25 (pure Python, zero new deps, djb2 hash, sublinear TF scaling) |
| Retrieval fusion | Qdrant Prefetch + Reciprocal Rank Fusion (RRF) |
| Vector store | Qdrant — HNSW m=16/ef=200 + TurboQuant 4-bit (8× smaller, ~1% recall loss) |
| Semantic cache | Qdrant `semantic_cache` collection — cosine similarity ≥ 0.92, 1hr TTL |
| Metadata + transactions | PostgreSQL |
| Task queue + feedback tuner | Celery + Redis |
| Agent framework | LangGraph (6-node pipeline) |
| Re-ranking | Cohere Rerank v3 |
| API | FastAPI — JWT auth, per-token revocation, sliding-window rate limiting |
| Fine-tuning export | `scripts/export_finetune_data.py` — SFT / DPO / Reranker JSONL |

All AI calls go through AWS Bedrock. Data never leaves the VPC.

---

## Quick start (development)

**Prerequisites:** Docker Desktop, Python 3.11+, AWS account with Bedrock access (Claude + Cohere models enabled)

```bash
# 1. Clone and install
git clone https://github.com/Praveenkumarbyrapuneni/enterprise-rag-agents.git
cd enterprise-rag-agents
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Required: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION,
#           DATABASE_URL, JWT_SECRET_KEY, COHERE_API_KEY
# Generate JWT secret: openssl rand -hex 32

# 3. Start infrastructure
docker compose up -d

# 4. Set up database tables (includes feedback + retrieval_params + query_chunks)
python -m db.schema

# 5. Ingest documents (place your documents in tests/ folder first)
EMBEDDER_CONTEXT_MODE=skip python -m scripts.reingest_with_tenants

# 6. Start the API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000/docs** — Swagger UI with all endpoints.

```bash
# Register a user
POST /auth/register
{ "username": "analyst1", "email": "analyst@firm.com", "password": "yourpassword",
  "tenant_id": "goldman", "customer_id": "goldman_user_001" }

# Login → get JWT token
POST /auth/login
{ "username": "analyst1", "password": "yourpassword" }

# Query — just the question, no tenant_id needed
POST /query  (Authorization: Bearer <token>)
{ "question": "How did the firm manage credit risk in 2025?" }
# Response includes query_id — use it to submit feedback

# Submit feedback
POST /feedback  (Authorization: Bearer <token>)
{ "query_id": "uuid-from-query-response", "helpful": true, "comment": "Great answer" }

# Export fine-tuning datasets (after enough feedback accumulates)
python -m scripts.export_finetune_data --tenant goldman
```

---

## Production deployment

```bash
# Build and run in production mode (pinned versions, health checks, resource limits)
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Includes:
#   api        — FastAPI with 4 Uvicorn workers
#   worker     — Celery worker for document ingestion + feedback tuning
#   qdrant     — pinned v1.13.4, 4GB memory limit
#   postgres   — 16-alpine, health-checked
#   redis      — 7.4-alpine, 512MB LRU, persistence enabled
```

**POSTGRES_PASSWORD must be set in .env** — the prod compose has no hardcoded default.

---

## Laptop → AWS (zero code changes)

Every component has a direct AWS equivalent. Only `.env` changes.

| Laptop (Phase A) | AWS (Phase B) |
|---|---|
| Qdrant Docker | Qdrant Cloud or EC2 cluster |
| PostgreSQL Docker | RDS PostgreSQL (Multi-AZ) |
| Redis Docker | ElastiCache / SQS |
| Celery local | ECS Fargate auto-scaling workers |
| File logs | CloudWatch (`LOG_OUTPUT=cloudwatch`) |
| Local files | S3 (SSE-KMS encrypted) |
| JWT local secret | AWS Secrets Manager |
| Company SSO (already works today, any OIDC provider) | Point `OIDC_ISSUER_URL` at Cognito's OIDC endpoint — no code change |
| API local | ECS Fargate + Application Load Balancer |

---

## Fine-tuning data export

Once feedback accumulates, export ready-to-train JSONL datasets:

```bash
python -m scripts.export_finetune_data               # all tenants
python -m scripts.export_finetune_data --tenant goldman
```

Output:

| File | Format | Use |
|---|---|---|
| `data/sft_synthesizer.jsonl` | Qwen3 chat template | Fine-tune local synthesizer to replace Claude Sonnet |
| `data/dpo_pairs.jsonl` | `{prompt, chosen, rejected}` | DPO preference training |
| `data/reranker_train.jsonl` | `{query, pos, neg}` | Fine-tune domain-specific reranker |

The script prints readiness warnings (e.g. "< 100 examples — collect more feedback") so you know exactly when to start training.

---

## Go deeper

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — full system design and tenant isolation model
- [`NEXT_VERSION.md`](NEXT_VERSION.md) — what's planned next and exactly when to build each item
- [`ingestion/DECISIONS.md`](ingestion/DECISIONS.md) — every production decision in the ingestion pipeline
- [`agents/DECISIONS.md`](agents/DECISIONS.md) — every decision in the agent pipeline
