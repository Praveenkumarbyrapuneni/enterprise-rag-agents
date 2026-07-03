# Enterprise RAG System — Architecture

A production-grade Retrieval-Augmented Generation system designed for financial
institutions at the scale of Discover, JPMorgan, or Goldman Sachs.

---

## Scale Target

| Dimension | Target |
|---|---|
| Documents stored | 10 million+ |
| New documents per day | Millions |
| Tenants | Millions (one per institution or customer) |
| Query concurrency | Thousands simultaneous |
| Data loss tolerance | Zero |
| Compliance | PCI DSS, SOC 2 infrastructure thinking |

Every architecture decision is evaluated against this target. If a choice breaks
at 10M documents or millions of tenants, it is not used.

---

## System Overview

```
                        ┌─────────────────────────────────┐
                        │           FastAPI                │
                        │   POST /query  GET /health       │
                        │   API key auth + rate limiting   │
                        └────────────────┬────────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │    LangGraph Graph   │
                              │    (Phase 3 routing) │
                              └──────────┬──────────┘
                                         │
               ┌─────────────────────────┼─────────────────────────┐
               │                         │                         │
          data_source=sql           data_source=hybrid       data_source=rag
               │                         │                         │
         ┌─────▼─────┐           ┌───────▼──────┐          ┌──────▼──────┐
         │ DB Lookup  │           │  DB Lookup   │          │  Retriever  │
         │ PostgreSQL │           │  +           │          │  Qdrant     │
         │ transactions│          │  Retriever   │          │  (tenant    │
         └─────┬─────┘           │  Qdrant      │          │   filtered) │
               │                 └───────┬──────┘          └──────┬──────┘
               │                         │                         │
               └─────────────────────────┼─────────────────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │     Synthesizer      │
                              │   Claude Sonnet 4    │
                              │   (or direct format  │
                              │    for sql-only)     │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │      Evaluator       │
                              │   Haiku judge +      │
                              │   retry loop         │
                              └─────────────────────┘
```

---

## Three Query Paths

### Path 1 — SQL (live structured data)
**Triggers:** balance, transactions, spending totals, flagged charges
**Flow:** Query Analyzer → DB Lookup → Synthesizer (no LLM, direct format)
**Evaluator score:** Always 1.0/1.0 — database answers cannot hallucinate

### Path 2 — RAG (document knowledge)
**Triggers:** filing analysis, risk assessment, regulatory questions, strategy
**Flow:** Query Analyzer → Retriever → Synthesizer (Claude) → Evaluator → retry?
**Retriever internals:** dense Prefetch + BM25 sparse Prefetch → RRF fusion → Cohere rerank → MMR → parent expansion
**Evaluator score:** LLM-as-judge on faithfulness + relevance

### Path 3 — Hybrid (live data + document knowledge)
**Triggers:** explain a charge, why was I flagged, does this comply with policy
**Flow:** Query Analyzer → DB Lookup → Retriever → Synthesizer (Claude, both sources)
**Evaluator score:** LLM-as-judge on combined answer

---

## Ingestion Pipeline

```
Document file
      │
      ▼
   Parser              9 formats: PDF, DOCX, XLSX, CSV, HTML,
      │                PNG/JPG/GIF/WEBP, TIFF (OCR), EML
      ▼
   Chunker             Content-aware: keyword classifier routes
      │                simple docs → fixed (512 tok)
      │                financial/complex → hierarchical (1024 parent / 256 child)
      ▼
   Embedder            Step 1: Cohere Embed v3 via Bedrock → 1024-dim dense vector
      │                         Haiku contextual enrichment (20 parallel workers)
      │                Step 2: BM25 sparse vector from raw chunk text (pure Python)
      │                         djb2 hash, sublinear TF scaling, no external deps
      ▼
   Qdrant Uploader     4-phase idempotent write:
      │                check → parents(PostgreSQL) → vectors(Qdrant) → mark complete
      │                Named vectors: "dense" (TurboQuant 4-bit) + "sparse" (BM25)
      │                TurboQuant 4-bit on dense: 8x memory compression, ~1% recall loss
      ▼
   PostgreSQL           parent_chunks: full paragraph text for context expansion
                        ingestion_status: hash gate, deduplication
                        failed_documents: DLQ for failed ingestion tasks
```

---

## Tenant Isolation

Every document chunk stored in Qdrant carries a `tenant_id` payload field.
Every query filters on `tenant_id` before returning any results.

```
Goldman analyst query → filter: tenant_id="goldman" → only Goldman chunks returned
Apple analyst query   → filter: tenant_id="apple"   → only Apple chunks returned
```

**Why payload filter over separate collections:**
Separate collections = one HNSW graph per tenant. At millions of tenants this
breaks Qdrant's internals. Payload filter with a KEYWORD index is O(1) lookup
regardless of tenant count — the only architecture that survives at this scale.

Research basis: without tenant isolation, 95% of benign queries leak data across
tenants via shared entity connections (arxiv, 2025).

---

## Phase A → Phase B Migration

The system runs identically on laptop and AWS. Only environment variables change.
Zero code rewrites between phases.

| Component | Phase A (Laptop) | Phase B (AWS) |
|---|---|---|
| Vector store | Qdrant via Docker | Qdrant Cloud / EC2 cluster |
| Metadata store | PostgreSQL via Docker | AWS RDS PostgreSQL |
| Task queue broker | Redis via Docker | AWS SQS |
| Worker pool | Celery local | AWS ECS auto-scaling fleet |
| File storage | Local disk | AWS S3 |
| API | uvicorn local | ECS + API Gateway |
| LLM + Embeddings | AWS Bedrock | AWS Bedrock (unchanged) |
| Logging | File → logs/rag.log | ECS stdout → CloudWatch |
| Auth | API key | AWS Cognito JWT |
| Real-time data | Mock transactions table | Kafka → PostgreSQL stream |

**LangSmith is never used.** Financial document data must never leave the AWS VPC.
CloudWatch replaces LangSmith for observability — zero code change required.

---

## AI Stack

All AI calls go through AWS Bedrock. Data never leaves the AWS VPC.

| Role | Model |
|---|---|
| Answer synthesis | Claude Sonnet 4 via Bedrock |
| Query classification, evaluation, enrichment | Claude Haiku 4.5 via Bedrock |
| Embeddings | Cohere Embed v3 via Bedrock (1024 dims) |
| Re-ranking | Cohere Rerank v3 (direct API) |
