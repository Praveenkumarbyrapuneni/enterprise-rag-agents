# Enterprise RAG System

A production-grade document intelligence system built for financial institutions.
Users log in, ask questions in plain English, and get grounded answers from both
their live transaction data and their institution's policy documents — with source citations.

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
Question classified automatically: needs transaction data + policy document
        ↓
PostgreSQL: finds the June 26 transaction ($340, flagged, foreign merchant)
Qdrant:     finds the foreign transaction fee policy in the institution's documents
        ↓
Claude synthesizes both into a grounded answer with citations
        ↓
Answer returned in ~2 seconds. No human involved.
```

This scales to millions of daily questions across millions of users from multiple institutions — all isolated from each other.

---

## Authentication & Tenant Isolation

Every user belongs to exactly one institution (tenant). That identity is set at registration and baked into their JWT token at login.

```
POST /auth/register  →  create account, assign to an institution
POST /auth/login     →  returns a signed JWT (valid 24h)
POST /query          →  JWT decoded → institution extracted → all searches filtered to that institution only
```

The caller never passes `institution_id` manually. It flows automatically from the token.
A user from Institution A physically cannot retrieve Institution B's documents — the Qdrant filter is enforced at the vector search level, not in application logic.

---

## How it works internally

```
Document comes in
      ↓
Parse (PDF/Word/Excel/HTML/email) → Chunk (hierarchical) → Embed (Cohere v3)
      ↓
Store: vectors + tenant_id → Qdrant | parent chunks + tenant_id → PostgreSQL
      ↓
User logs in → JWT with tenant_id issued
      ↓
User asks a question
      ↓
Query Analyzer (Claude Haiku) classifies: sql / rag / hybrid
      ↓
  sql    → PostgreSQL transaction lookup
  rag    → Qdrant vector search (tenant-filtered) → Cohere rerank → MMR dedup
  hybrid → both paths combined
      ↓
Synthesizer (Claude Sonnet) writes grounded answer with inline citations
      ↓
Evaluator scores faithfulness + relevance → retries if below threshold
      ↓
Answer returned via FastAPI
```

---

## Stack

| Component | Technology |
|---|---|
| LLM | Claude Sonnet 4 + Haiku 4.5 via AWS Bedrock |
| Embeddings | Cohere Embed v3 via Bedrock (1024 dims) |
| Vector store | Qdrant — HNSW + TurboQuant 4-bit (8× smaller, ~1% recall loss) |
| Metadata + transactions | PostgreSQL |
| Task queue | Celery + Redis |
| Agent framework | LangGraph (6-node pipeline) |
| Re-ranking | Cohere Rerank v3 |
| API | FastAPI — JWT auth, sliding-window rate limiting |

All AI calls go through AWS Bedrock. Data never leaves the VPC.

---

## Quick start

**Prerequisites:** Docker Desktop, Python 3.11+, AWS account with Bedrock access (Claude + Cohere models enabled)

```bash
# 1. Clone and install
git clone <repo-url>
cd enterprise-rag-agents
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure — copy and fill in your keys
cp .env.example .env
# Required: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, DATABASE_URL, JWT_SECRET_KEY
# Generate JWT secret: openssl rand -hex 32

# 3. Start infrastructure
docker compose up -d

# 4. Set up database tables
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
```

---

## Laptop → AWS (zero code changes)

Every component has a direct AWS equivalent. Only `.env` changes.

| Laptop (Phase A) | AWS (Phase B) |
|---|---|
| Qdrant Docker | Qdrant Cloud |
| PostgreSQL Docker | RDS PostgreSQL |
| Redis Docker | ElastiCache / SQS |
| Celery local | ECS auto-scaling workers |
| File logs | CloudWatch (LOG_OUTPUT=cloudwatch) |
| Local files | S3 |
| JWT (local secret) | Cognito User Pools |

---

## Go deeper

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — full system design and tenant isolation model
- [`ingestion/DECISIONS.md`](ingestion/DECISIONS.md) — every production decision in the ingestion pipeline
- [`agents/DECISIONS.md`](agents/DECISIONS.md) — every decision in the agent pipeline
