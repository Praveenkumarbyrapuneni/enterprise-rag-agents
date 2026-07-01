# Enterprise RAG System

A production-grade document intelligence system built for financial institutions.
Ask any question about your documents or transactions — it finds the answer, cites the source, and tells you how confident it is.

Designed to scale to Discover/JPMorgan level: 10M+ documents, millions of users, zero data loss.

---

## What it does

**Ingest any financial document** → PDF, Word, Excel, HTML, images, emails — parsed, chunked, and stored as searchable vectors.

**Answer questions in three ways:**

| Query type | Example | How it answers |
|---|---|---|
| Document search | "What are Apple's biggest risk factors?" | Searches SEC filings, returns cited answer |
| Live data | "What is my current balance?" | Queries transaction database directly |
| Hybrid | "Why was I charged a foreign fee on June 26?" | Combines transaction record + fee policy document |

**Self-evaluates every answer** → scores faithfulness and relevance, retries automatically if quality is low.

---

## How it works

```
Document comes in
      ↓
Parse → Chunk → Embed (Cohere v3) → Store in Qdrant + PostgreSQL
                                           ↓
User asks a question
      ↓
Classify question (Claude Haiku) → SQL? RAG? Both?
      ↓
Retrieve relevant chunks (tenant-filtered)
      ↓
Synthesize grounded answer with citations (Claude Sonnet 4)
      ↓
Evaluate quality → retry if needed → return answer via FastAPI
```

---

## Stack

| | |
|---|---|
| LLM | Claude Sonnet 4 + Haiku 4.5 via AWS Bedrock |
| Embeddings | Cohere Embed v3 via Bedrock (1024 dims) |
| Vector store | Qdrant — TurboQuant 4-bit compression (8× smaller, ~1% recall loss) |
| Metadata | PostgreSQL — parent chunks, tenant registry, transactions |
| Task queue | Celery + Redis |
| Agent framework | LangGraph |
| Re-ranking | Cohere Rerank v3 |
| API | FastAPI |

All AI calls go through AWS Bedrock. Data never leaves the VPC — required for financial institution clients.

---

## Quick start

```bash
# Infrastructure
docker compose up -d

# Install
pip install -r requirements.txt

# Configure (copy .env.example → .env, fill in AWS credentials)

# Set up database
python -m db.schema

# Ingest documents
python run_pipeline.py

# Query
python test_query.py

# Or run the API
uvicorn api.main:app --port 8000
```

---

## Laptop → AWS (zero code changes)

Every component has a direct AWS equivalent. Only `.env` changes.

| Laptop | AWS |
|---|---|
| Qdrant Docker | Qdrant Cloud |
| PostgreSQL Docker | RDS |
| Redis Docker | SQS |
| Celery local | ECS auto-scaling |
| File logs | CloudWatch |
| API key auth | Cognito JWT |

---

## Go deeper

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — full system design and tenant isolation
- [`ingestion/DECISIONS.md`](ingestion/DECISIONS.md) — every production decision in the ingestion pipeline
- [`agents/DECISIONS.md`](agents/DECISIONS.md) — every decision in the agent pipeline, with research citations
