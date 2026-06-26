# Enterprise RAG + Multi-Agent System
## Project 1 of 2 — Praveen Kumar Byrapuneni

---

## What This Project Is

A production-grade financial document intelligence platform powered by a multi-agent LangGraph architecture and RAG pipeline. Users upload earnings reports, SEC filings, and company documents. The system answers complex questions with cited sources, cross-references multiple documents simultaneously, fact-checks its own answers, and scores its own retrieval quality using RAGAS.

This is not a tutorial clone. It is a production system designed to demonstrate every core AI Engineer skill that FAANG companies test — RAG pipeline design, multi-agent orchestration, vector database management, model evaluation, AI safety, and cloud deployment.

**Companion project:** `llmops-finetune-pipeline` (Project 2) covers fine-tuning, MLflow, vLLM, and LLMOps. Build this project first.

---

## The Problem Being Solved

Analysts at investment firms, hedge funds, and financial institutions spend 6-8 hours per day reading documents — earnings calls, 10-K filings, analyst reports. They need answers that span multiple documents, require cross-referencing, and must be factually grounded with citations. Generic LLMs hallucinate on financial data and cannot cite sources. This system solves that.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│              FastAPI REST API + Optional Streamlit UI           │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LANGRAPH ORCHESTRATOR                         │
│         Stateful graph — controls agent flow and memory         │
└──────┬──────────────┬──────────────┬──────────────┬────────────┘
       │              │              │              │
       ▼              ▼              ▼              ▼
┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────────┐
│  Agent 1 │  │   Agent 2    │  │  Agent 3 │  │   Agent 4    │
│  Query   │  │  Retriever   │  │Synthesizer│  │  Evaluator  │
│ Analyzer │  │              │  │          │  │  (RAGAS)    │
└──────────┘  └──────┬───────┘  └──────────┘  └──────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                      QDRANT VECTOR STORE                        │
│     Single collection — doc_id payload filter per document      │
└─────────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE LAYER                         │
│   AWS Lambda (API) │ S3 (documents) │ LangSmith (tracing)      │
│   Grafana (monitoring) │ Guardrails AI (safety)                │
└─────────────────────────────────────────────────────────────────┘
```

---

## The 4 Agents — What Each One Does

### Agent 1 — Query Analyzer
**Input:** Raw user question
**Job:** Decompose complex questions into sub-questions. Identify which documents are relevant. Detect if the question requires cross-document reasoning or single-document lookup.
**Output:** List of sub-queries + list of relevant document namespaces to search
**Key concept learned:** Query decomposition, HyDE (Hypothetical Document Embeddings)

### Agent 2 — Retriever
**Input:** Sub-queries from Agent 1
**Job:** Run parallel retrieval across Qdrant collections filtered by doc_id. Apply re-ranking (Cohere reranker). Apply MMR (Maximum Marginal Relevance) to avoid redundant chunks.
**Output:** Top-K relevant chunks with metadata (source document, page, section)
**Key concept learned:** Semantic search, re-ranking, MMR, hybrid search (dense + sparse), HNSW index tuning

### Agent 3 — Synthesizer
**Input:** Chunks from Agent 2
**Job:** Write a grounded answer using only retrieved context. Every factual claim must have a citation. Apply chain-of-thought reasoning. If information is missing, say so.
**Output:** Answer with inline citations + confidence score
**Key concept learned:** Grounded generation, citation extraction, chain-of-thought prompting

### Agent 4 — Evaluator
**Input:** Query, retrieved chunks, generated answer
**Job:** Score the answer using RAGAS metrics — faithfulness (is the answer grounded?), answer relevance (does it answer the question?), context recall (did retrieval find the right chunks?). If scores are below threshold, trigger Agent 2 again with different retrieval parameters.
**Output:** RAGAS scores + pass/fail decision + retry trigger if needed
**Key concept learned:** RAGAS evaluation framework, self-correcting agents, evaluation-driven development

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Agent Orchestration | LangGraph | Stateful graphs, conditional edges, retry loops |
| LLM | Anthropic Claude (claude-sonnet-4-6) | Best reasoning, reliable citations |
| Embeddings | OpenAI text-embedding-3-large | Best retrieval performance |
| Vector Store | Qdrant (Docker, self-hosted) | Open-source, HNSW internals visible, payload filtering for doc isolation |
| Re-ranker | Cohere Rerank | Improves retrieval precision by 30-40% |
| Evaluation | RAGAS | Industry standard RAG evaluation |
| Tracing | LangSmith | Trace every agent call, debug retrieval failures |
| Safety | Guardrails AI | PII redaction, prompt injection detection |
| API | FastAPI | Production REST API |
| Containerization | Docker + Docker Compose | Local dev parity |
| Cloud | AWS Lambda + S3 | Serverless deployment, document storage |
| Monitoring | Prometheus + Grafana | RAGAS score trends, latency, error rates |
| CI/CD | GitHub Actions | Auto deploy on push to main |

---

## Project Folder Structure

```
enterprise-rag-agents/
├── README.md                          ← This file
├── docker-compose.yml                 ← Local dev: API + Qdrant
├── .env.example                       ← All required env vars (never commit .env)
├── .gitignore
├── requirements.txt
│
├── ingestion/                         ← Document processing pipeline
│   ├── chunker.py                     ← Semantic + hierarchical chunking
│   ├── embedder.py                    ← OpenAI embedding wrapper
│   ├── qdrant_uploader.py             ← Upload chunks to Qdrant with doc_id payload
│   └── supported_formats/            ← PDF, DOCX, HTML parsers
│
├── agents/                            ← All 4 LangGraph agents
│   ├── graph.py                       ← LangGraph state machine — main orchestrator
│   ├── state.py                       ← Shared state schema (TypedDict)
│   ├── query_analyzer.py              ← Agent 1
│   ├── retriever.py                   ← Agent 2
│   ├── synthesizer.py                 ← Agent 3
│   └── evaluator.py                   ← Agent 4 (RAGAS)
│
├── api/                               ← FastAPI app
│   ├── main.py
│   ├── routers/
│   │   ├── ingest.py                  ← POST /ingest — upload document
│   │   ├── query.py                   ← POST /query — ask question
│   │   └── health.py                  ← GET /health
│   └── middleware/
│       ├── auth.py                    ← API key auth
│       └── guardrails.py              ← PII redaction, injection detection
│
├── evaluation/                        ← RAGAS evaluation suite
│   ├── ragas_evaluator.py             ← Faithfulness, relevance, recall scoring
│   ├── test_dataset.json              ← 50 question-answer pairs for benchmarking
│   └── benchmark.py                  ← Run full eval suite, generate report
│
├── monitoring/                        ← Observability
│   ├── prometheus_metrics.py          ← Custom metrics: RAGAS score, latency
│   └── grafana_dashboard.json         ← Import this into Grafana
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_agents.py
│   └── test_api.py
│
├── .github/
│   └── workflows/
│       └── deploy.yml                 ← GitHub Actions CI/CD → AWS Lambda
│
└── notebooks/
    └── rag_experimentation.ipynb      ← Chunking strategy experiments
```

---

## Data Flow — Ingestion

```
1. User uploads PDF via POST /ingest
2. Parser extracts text (PyMuPDF for PDF, python-docx for DOCX)
3. Chunker splits text:
   - Strategy A: Fixed-size chunks (512 tokens, 50 overlap)
   - Strategy B: Semantic chunking (sentence transformer similarity)
   - Strategy C: Hierarchical (document → section → paragraph)
   [LEARNING: compare all 3 strategies, measure RAGAS recall]
4. Embedder calls OpenAI text-embedding-3-large for each chunk
5. QdrantUploader stores chunks in "documents" collection with payload: {doc_id, source, page, section, chunk_index, timestamp}
6. Doc isolation via payload filter on doc_id — no separate namespace per doc
```

## Data Flow — Query

```
1. User sends POST /query {question: "What was Apple's Q4 revenue?"}
2. Guardrails middleware: check for PII, prompt injection
3. LangGraph graph starts:
   Agent1: "Apple Q4 revenue" → sub_queries=["Q4 revenue", "Q4 2024 earnings"]
                               → doc_ids=["apple_10k", "apple_earnings_call"]
   Agent2: Qdrant search with payload filter {doc_id: [...]}, top-20 → rerank → top-5
   Agent3: Generate answer with citations → confidence=0.87
   Agent4: RAGAS scores → faithfulness=0.91, relevance=0.88 → PASS
4. Return answer + citations + RAGAS scores to user
5. LangSmith logs full trace
6. Prometheus records: latency=1.4s, ragas_faithfulness=0.91
```

---

## Phase-by-Phase Build Plan

### Phase 1 — Foundation (Week 1)
- [ ] Set up project structure, Docker Compose, environment variables
- [ ] Build document ingestion pipeline (PDF parsing → chunking → embedding)
- [ ] Spin up Qdrant via Docker — upload first document, run first similarity search
- [ ] Test chunking strategies — measure which gives best recall
- [ ] Tune Qdrant HNSW params (ef_construction, m) — observe recall vs. speed tradeoff
- [ ] **Milestone:** Can ingest a PDF and retrieve relevant chunks via Python script

### Phase 2 — Agents (Week 2)
- [ ] Build LangGraph state schema
- [ ] Build Agent 1 (Query Analyzer) — test with 10 sample questions
- [ ] Build Agent 2 (Retriever) — add Cohere reranker, test MMR
- [ ] Build Agent 3 (Synthesizer) — prompt engineer for citations
- [ ] Wire all 3 agents into LangGraph graph
- [ ] **Milestone:** End-to-end query works without evaluation

### Phase 3 — Evaluation + Self-Correction (Week 3)
- [ ] Integrate RAGAS — compute faithfulness, relevance, recall
- [ ] Build Agent 4 (Evaluator) — trigger retry if RAGAS below threshold
- [ ] Build 50-question test dataset from sample documents
- [ ] Run benchmark — measure baseline RAGAS scores
- [ ] Add LangSmith tracing — trace every agent step
- [ ] **Milestone:** System self-corrects on low-quality answers

### Phase 4 — Production (Week 4)
- [ ] Build FastAPI app with all routers
- [ ] Add Guardrails AI — PII redaction, prompt injection detection
- [ ] Dockerize the application
- [ ] Deploy to AWS Lambda + S3
- [ ] Set up Prometheus + Grafana dashboard
- [ ] Write GitHub Actions CI/CD pipeline
- [ ] Write README with architecture diagram
- [ ] **Milestone:** Live URL, working demo, GitHub repo public

---

## Environment Variables Required

```bash
# LLM
ANTHROPIC_API_KEY=
OPENAI_API_KEY=

# Vector Store (Qdrant — runs locally via Docker, no API key needed for local)
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=documents

# Re-ranking
COHERE_API_KEY=

# Tracing
LANGSMITH_API_KEY=
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=enterprise-rag-agents

# AWS
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1
S3_BUCKET_NAME=

# API
API_KEY=
```

---

## FAANG Interview Questions This Project Answers

**System Design:**
- "Design a document QA system for 10M documents" → you built this
- "How do you ensure LLM answers are grounded in facts?" → RAGAS faithfulness + citations
- "How would you handle 10,000 concurrent queries?" → Lambda auto-scaling + Qdrant cluster (or Qdrant Cloud for managed scale)

**AI/ML:**
- "What chunking strategy gives best RAG performance?" → you measured it
- "How do you evaluate a RAG system without ground truth?" → RAGAS reference-free evaluation
- "What is MMR and when do you use it?" → you used it in Agent 2
- "What is HyDE and how does it improve retrieval?" → you implemented it in Agent 1

**Production:**
- "How do you monitor an LLM in production?" → Prometheus metrics + Grafana
- "How do you prevent prompt injection?" → Guardrails AI middleware
- "Walk me through your CI/CD pipeline" → GitHub Actions → AWS Lambda

---

## Success Metrics

| Metric | Target |
|---|---|
| RAGAS Faithfulness | > 0.85 |
| RAGAS Answer Relevance | > 0.80 |
| RAGAS Context Recall | > 0.75 |
| API Latency (p95) | < 3 seconds |
| Uptime | > 99% |

---

## Status

**Current phase:** Not started
**Next action:** Create project structure and set up Docker Compose
