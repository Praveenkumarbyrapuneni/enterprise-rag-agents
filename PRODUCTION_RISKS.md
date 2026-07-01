# Production Risk Register

A living document of known production failure patterns, their severity, and the current
mitigation status in this codebase. Updated after each security/production audit.

Last audited: 2026-07-01

---

## How to Read This

- **MITIGATED** — addressed in code, with DECISIONS.md documentation
- **PARTIAL** — partially addressed; known gap with a documented upgrade path
- **OPEN** — not yet addressed; must be resolved before production client use
- **PHASE B** — intentionally deferred to AWS migration; safe for laptop/demo use

---

## Critical — Must Resolve Before Any Client Data Touches This System

### C1 — BadRAG Corpus Poisoning (98.2% attack success rate at 0.04% corpus contamination)

**Status: OPEN**

An attacker who can influence document ingestion can poison 0.04% of the corpus with
adversarially-optimized text. For every query about the target topic, the poisoned
document is retrieved and the LLM response is hijacked.

**Required before production:**
- Authenticate all ingest sources (no anonymous document uploads)
- Add content scanning before ingest — check for instruction-like text in unexpected contexts
- Add `source_url` and `ingested_by` fields to every Qdrant payload for provenance tracking
- Monitor retrieval frequency: if any single document is returned for >20% of diverse queries, flag it

---

### C2 — Missing SEC/FINRA Audit Trail (enforcement risk)

**Status: MITIGATED in Phase 3**

The SEC fined 16 firms $81M in 2024 for AI communication recordkeeping failures. Every
AI response to a customer is a business communication that must be retained for 7 years.

**What was added:**
- `query_audit_log` table in PostgreSQL — every query/response pair is logged with
  `tenant_id`, `customer_id`, `question`, retrieved sources, model scores, and timestamp
- `_write_audit_log()` called on every `/query` response (non-fatal — log failure never breaks response)

**Remaining gap:**
- Table is append-only by convention, not by database constraint. For regulatory use,
  enforce immutability via PostgreSQL row-level triggers or S3 with Object Lock.

---

## High — Address Before Scaling to Production Load

### H1 — PostgreSQL Connection Pool Exhaustion

**Status: OPEN (safe for demo; critical at scale)**

PostgreSQL default `max_connections=100`. With multiple FastAPI instances and Celery workers,
concurrent connections can exceed this limit. All new requests fail.

**Required for Phase B:**
- Deploy PgBouncer in transaction mode as a connection multiplexer
- Set pool size to `(CPU_cores × 2) + 1` per application instance
- Monitor `pg_stat_activity` count as a production metric

---

### H2 — Bedrock max_tokens Pre-Allocation Burns Quota 100x Faster

**Status: MITIGATED**

Bedrock reserves quota based on `max_tokens` BEFORE processing. An unset `max_tokens` defaults
to 64,000 on Claude Sonnet 4+, burning 64,000 TPM per request even for 200-token responses.
Output tokens also carry a 5x multiplier on TPM calculation.

**Current mitigations:**
- `synthesizer.py`: `_MAX_TOKENS = 1024`
- `evaluator.py`: `_MAX_TOKENS = 256`
- `query_analyzer.py`: `_MAX_TOKENS = 512`
- `db_lookup.py`: `_MAX_TOKENS = 256`

All Bedrock calls have explicit `max_tokens` set. This is why the daily quota lasted
longer on some days than others — earlier runs without explicit limits burned 64x faster.

---

### H3 — HNSW CPU Exhaustion During Bulk Re-Ingest

**Status: PARTIAL**

Re-ingesting 3,000+ chunks with HNSW indexing enabled causes every insert to update the
graph. At 10M+ documents, this becomes untenable.

**Current state:** Acceptable at current scale (3,233 chunks).

**Required at scale:**
```python
# Before re-ingest: disable HNSW construction
client.update_collection(COLLECTION_NAME, optimizer_config=OptimizersConfigDiff(indexing_threshold=0))
# ... run all upserts ...
# After re-ingest: re-enable
client.update_collection(COLLECTION_NAME, optimizer_config=OptimizersConfigDiff(indexing_threshold=20000))
```

---

### H4 — Multi-Tenant RLS Bypass (CVE-2025-8713)

**Status: PARTIAL (defense-in-depth applied)**

PostgreSQL Row Level Security has documented bypass vulnerabilities via optimizer statistics.
RLS alone cannot be trusted for financial data isolation.

**Current mitigations:**
- `tenant_id` filter enforced in application code on every Qdrant search (retriever.py)
- `tenant_id` in all SQL WHERE clauses (db_lookup.py)
- Defense-in-depth: two independent isolation layers

**Remaining gap:** Separate PostgreSQL schemas per tenant for the most sensitive data (Phase B).

---

## Medium — Address Before High-Traffic Launch

### M1 — 40% Retrieval Miss Rate Without Hybrid Search

**Status: PARTIAL — documented known gap**

Dense vector embeddings optimize for semantic similarity, not keyword precision. Financial
queries requiring exact terms (ticker symbols, fiscal quarters, regulation codes) miss
relevant documents at a ~40% rate compared to hybrid dense+sparse search.

**Current mitigation:** Cohere Embed v3 encodes significant lexical information. Cohere
reranker adds 30-48% precision improvement on top.

**Upgrade path:** BM25 sparse vectors in Qdrant — requires re-ingestion with sparse+dense fields.
Planned for Phase B.

---

### M2 — LangGraph Infinite Loop Detection

**Status: PARTIAL**

LangGraph has no native circuit breaker. The current architecture caps retries at 1
(documented in evaluator DECISIONS.md). However, a malformed document that consistently
produces low-quality responses will exhaust the single retry without improvement.

**Current mitigation:** `_MAX_RETRIES = 1` in evaluator.py — hard cap.

**Upgrade:** Compare consecutive answers for content similarity — if identical, route to END
without retry (identical = retry cannot help).

---

### M3 — LangGraph MemorySaver State Lost on Restart

**Status: PHASE B**

MemorySaver stores graph state in process memory. Process restart destroys all in-flight states.

**Current architecture:** All queries are single-shot (stateless). This risk only materializes
if multi-turn conversation is added.

**If multi-turn is added:** Switch to `PostgresSaver` immediately. Never use `SqliteSaver`
in production (single-writer limitation).

---

### M4 — Bedrock Cross-Region Failover Not Automatic

**Status: PHASE B**

A regional Bedrock outage takes down all inference without Cross-Region Inference configured.
24/7 financial SLAs cannot tolerate single-region dependency.

**Required for Phase B:**
- Enable Bedrock Cross-Region Inference profiles
- Primary: `us-east-1`, Failover: `us-west-2`
- Zero code change — only the ARN in `.env` changes

---

## Low — Best Practices, Address When Convenient

### L1 — JWT Token Revocation Not Supported

**Current:** JWTs are valid for 24 hours after issue with no revocation mechanism.
If a token is stolen, it's valid until expiry.

**Mitigation path:** Token blacklist in Redis (store `jti` claim of revoked tokens,
check on each request). Or shorten TTL to 1 hour and implement refresh tokens.

---

### L2 — Stale Embedding Model Deprecation

**Current:** Cohere Embed v3 and Rerank v3 model IDs are pinned in code.

**Risk:** Model deprecations can happen without breaking API errors — the endpoint still
works but returns degraded results.

**Mitigation:** Monitor Cohere and AWS deprecation pages. Add a startup check that
validates model IDs against a known-active list.

---

### L3 — Old Parent Chunks Not Deleted on Re-Ingest

**Current:** When a document is re-ingested with new content (new file hash), old
parent_chunks rows in PostgreSQL remain forever. Qdrant vectors are cleaned up
(by file_hash filter), but PostgreSQL rows accumulate.

**Impact:** Storage growth over time. No data correctness issue (orphaned parents
are never retrieved because their Qdrant child points don't exist).

**Mitigation path:** Add a cleanup step to `reingest_with_tenants.py` that deletes
`parent_chunks WHERE file_name = X` before re-ingesting.

---

### L4 — HNSW Index Degradation Is Silent

**Current:** HNSW graph degradation produces wrong results without raising errors.
No monitoring exists for this condition.

**Detection:** Monitor the gap between `vectors_count` and `indexed_vectors_count`
in Qdrant metrics. Alert when gap exceeds 5% — indicates fragmented segments
that need optimizer runs.

---

## Security Mitigations Already In Place

| Threat | Mitigation |
|--------|-----------|
| SQL injection | Parameterized queries only (psycopg2 `%s`), template SQL, no LLM-generated SQL |
| Prompt injection via user question | Input length cap (2000 chars), strict system prompt, tenant_id never from user input |
| Cross-tenant data access | Qdrant payload filter on every search + SQL WHERE on every query |
| Username enumeration via timing | `_DUMMY_HASH` ensures constant-time bcrypt even when user doesn't exist |
| Password brute force | Rate limiting on `/auth/login` (5 req/min per IP) |
| JWT forgery | HS256 with 256-bit random secret, explicit algorithm enforcement |
| Stack trace leakage | Generic error handler returns no internal detail |
| API key weakness | Replaced with JWT — no static keys in requests |
| Pickle deserialization RCE | Celery JSON serializer enforced (`task_serializer="json"`) |
| Corpus poisoning (partial) | Authenticated ingest only — no public upload endpoint |
