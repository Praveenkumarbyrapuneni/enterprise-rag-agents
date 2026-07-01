# Agent Pipeline — Engineering Decisions

Documents the key architectural decisions made in the LangGraph multi-agent
query pipeline, including research basis for each technique and production
issues discovered during development.

---

## Logger

### TimedRotatingFileHandler — Midnight Rotation, 30-Day Retention

```python
logging.handlers.TimedRotatingFileHandler(
    filename=_LOG_FILE,
    when="midnight",
    backupCount=30,
)
```

A single log file that never rotates grows forever. At 1,000 queries/minute
with verbose logging, `rag.log` reaches gigabytes within weeks. Disk fills up.
The process crashes. No logs from the crash period survive because they were
never rotated out — they're in the file that just caused the crash.

Midnight rotation creates a new file each day. `backupCount=30` keeps 30 days
of history. Enough to cover audit requirements and post-incident investigations
while preventing unbounded disk growth.

### LOG_OUTPUT=cloudwatch → Stdout Only, Zero Code Change

```python
_LOG_OUTPUT = os.getenv("LOG_OUTPUT", "file").lower()

if _LOG_OUTPUT == "file":
    root.addHandler(file_handler)
# cloudwatch: stdout only — ECS captures it automatically
```

ECS with the `awslogs` log driver ships everything written to stdout directly
to CloudWatch Logs. No SDK call, no CloudWatch client, no IAM permission for
`logs:PutLogEvents` from application code.

Phase A (laptop): `LOG_OUTPUT=file` → logs go to `logs/rag.log`
Phase B (AWS ECS): `LOG_OUTPUT=cloudwatch` → logs go to stdout → ECS ships to CloudWatch

One environment variable change. Zero code changes. The logger's design
anticipates the AWS migration from day one.

### Global _INITIALIZED Guard — Prevents Duplicate Handlers

```python
_INITIALIZED = False

def _setup():
    global _INITIALIZED
    if _INITIALIZED:
        return
    ...
    _INITIALIZED = True
```

Python's logging module attaches handlers to the root logger globally. If
`get_logger()` is called from 10 different modules at import time, without
the guard it would attach 10 StreamHandlers and 10 FileHandlers. Every log
line would print 10 times — once per handler.

The `_INITIALIZED` flag ensures `_setup()` runs exactly once per process
regardless of how many modules import `get_logger()`.

## Query Analyzer

### Adaptive HyDE — Not Always-On

HyDE (Hypothetical Document Embeddings) generates a hypothetical answer to the
query, embeds it, and uses that vector for retrieval instead of the question
itself. The hypothesis often matches document language better than a question does.

Problem: HyDE is dangerous on numerical financial queries. Asked "What was
Apple's iPhone revenue in Q3 2024?", the LLM generates a hypothesis like
"Apple reported iPhone revenue of $47.2 billion" — a fabricated number. That
vector then retrieves chunks containing figures near $47B instead of the actual
number. The retrieved context is biased toward the hallucinated hypothesis.

Decision: HyDE is enabled only for `conceptual` and `temporal` query types —
questions about strategy, risk, outlook, and trends — where the LLM can
generate accurate qualitative language without fabricating numbers.

HyDE is explicitly disabled for: `numerical`, `comparative`, `regulatory`,
`sql`, `hybrid`, `multi_company`, `general`.

Research basis: arxiv 2404.07221 confirms HyDE degrades accuracy on numerical
queries; arxiv 2507.16754 confirms the 25% of cases where HyDE hurts are
precision-focused questions.

### Sub-Question Decomposition — Only When Beneficial

Early implementation decomposed every multi-part question into sub-questions.

Problem: decomposition on single-entity queries hurts retrieval. "What were
Apple's Q3 2024 iPhone and Mac revenues?" decomposed into two sub-questions
causes two separate searches, each returning only partial context. A single
search on the original question returns the full earnings section.

Decision: decomposition is enabled only for `comparative` and `multi_company`
queries — where two or more distinct companies or entities require independent
searches. Sub-questions are capped at 4 to prevent over-decomposition.

Research basis: ACL 2025 (arxiv 2507.00355) — query decomposition increases
MRR by 36.7% on multi-hop queries but degrades single-hop performance.

### Silent Failure Is Not Acceptable — Fallback Instead of Crash

The Query Analyzer makes a Bedrock call on every single query. If that call
fails — timeout, throttle, wrong model ID, network blip — there are two options:

Option A: raise the exception → entire pipeline crashes → user gets a 500 error
Option B: catch the exception → log a warning → fall back to safe defaults

Option A was rejected. A classification failure is not a reason to give the
user zero answer. The fallback returns `query_type=general, data_source=rag`
which routes the question to Qdrant. The answer quality may be slightly lower
(no HyDE, no sub-question splitting) but the system still returns something useful.

```python
except (ClientError, json.JSONDecodeError, KeyError, IndexError) as e:
    logger.warning(f"[analyzer] Failed ({type(e).__name__}: {e}) — falling back")
    result = _fallback(question)
```

The warning is always logged — the failure is visible in CloudWatch. It is
never swallowed silently. The difference between this and silent failure:
silent failure hides the problem; this surfaces it while keeping the system alive.

### HyDE Minimum Length — Discard Useless Hypotheses

```python
_MIN_HYDE_LEN = 20

if hyde_applicable and len(hyde_query) < _MIN_HYDE_LEN:
    logger.warning("[analyzer] HyDE query too short — discarding")
    hyde_query = ""
    hyde_applicable = False
```

When Haiku is asked to generate a hypothetical answer for HyDE, it occasionally
returns a single word or a very short phrase — a signal that the model couldn't
generate a useful hypothesis for this particular question.

Embedding a 5-character hypothesis produces a near-meaningless vector that
retrieves random chunks — worse than embedding the original question directly.

The 20-character minimum discards these degenerate cases. The system falls
back to direct question embedding, which is the safer path when HyDE fails.

### sql and hybrid Query Types

Standard RAG systems treat every question as a document retrieval problem.
For financial account queries this is wrong.

"What is my current balance?" requires a live database lookup — the answer
changes every transaction. Embedding this question and searching a vector store
returns policy documents about balance calculation, not an actual number.

Two new query types were added:
- `sql` — requires only live transaction data (balance, recent transactions, spending)
- `hybrid` — requires both live data AND document knowledge (explain a charge
  using both the transaction record and the fee policy document)

---

## Retriever

### Process-Level Singleton Clients — Not Per-Request

The Qdrant client and Bedrock client are created once per worker process and
reused across all queries:

```python
_bedrock_client: Optional[object] = None
_qdrant_client:  Optional[QdrantClient] = None

def _get_qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(...)
    return _qdrant_client
```

Creating a new client on every query means:
- TCP handshake on every request (~50-200ms overhead)
- TLS negotiation on every HTTPS call to Bedrock (~100ms)
- Connection pool rebuilt from scratch every time

At 1,000 queries/minute, per-request clients add 150-300ms to every query and
exhaust the system's file descriptor limit (too many open connections).

Singleton clients reuse the same connection pool. The worker initializes once,
then serves thousands of queries through the same warm connection.

### Qdrant Retry with Exponential Backoff — 3 Attempts

Every Qdrant search retries up to 3 times on connection errors and 5xx responses:

```python
_QDRANT_RETRIES = 3
_RETRY_DELAYS   = [1, 2, 4]   # seconds between attempts
```

Attempt 1 fails → wait 1s → attempt 2 fails → wait 2s → attempt 3 fails → raise.

Why 3 retries and not more: Qdrant transient errors (network blip, brief
overload) typically resolve within 1-2 seconds. If it hasn't recovered after
7 seconds (1+2+4), it is likely a real outage, not a transient blip. Retrying
more just delays the error response to the user.

### 4xx vs 5xx Distinction — Never Retry Caller Errors

Qdrant errors are categorized before deciding whether to retry:

```python
except UnexpectedResponse as e:
    if e.status_code >= 500:
        last_exc = e    # server error — retriable
    else:
        raise           # 4xx — caller error, raise immediately
```

A 400 means the query itself is malformed — wrong dimensions, invalid filter
syntax, collection doesn't exist. Retrying the same malformed query 3 times
wastes 7 seconds and returns the same error. Raise immediately.

A 503 means Qdrant is temporarily overloaded. Retry after backoff — it will
likely recover.

### Cohere Timeout via ThreadPoolExecutor — SDK Has No Timeout Param

The Cohere reranker SDK does not expose a timeout parameter. If Cohere's API
hangs, the call blocks indefinitely — freezing the worker.

Fix: run the Cohere call inside a `ThreadPoolExecutor` with a timeout:

```python
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(_call_cohere)
    return future.result(timeout=_COHERE_TIMEOUT)  # 12 seconds
```

If the call doesn't complete in 12 seconds, `TimeoutError` is raised and the
system falls back to raw vector order (reranking skipped). The query still
completes — it just returns slightly lower-quality results.

Cohere's p50 latency is 100-400ms. A 12-second timeout catches genuine hangs
while never triggering on normal slow responses.

### Cohere Reranker Over BM25 Hybrid Search

Common production RAG advice is to use hybrid search: dense vectors + BM25
keyword scoring combined via Reciprocal Rank Fusion.

For this system, BM25 hybrid was evaluated and rejected for one reason: Cohere
Embed v3 already encodes strong lexical information. Adding BM25 to a top-tier
dense embedder adds noise, not signal, and increases latency.

The Cohere Reranker provides a larger gain (30-48% precision improvement) with
less complexity. The reranker receives all candidate chunks and re-scores them
using a cross-encoder — a fundamentally different approach from BM25 fusion.

BM25 hybrid remains planned for Phase B where full re-ingestion with sparse
vector fields is feasible.

### MMR: lambda=0.6

After reranking, Maximal Marginal Relevance (MMR) removes near-duplicate chunks
while preserving relevance. The lambda parameter controls the tradeoff:
- lambda=1.0 → pure relevance (keep highest-scored chunks regardless of similarity)
- lambda=0.0 → pure diversity (maximize chunk difference)

Lambda=0.6 was chosen to favor precision slightly over diversity. In financial
document retrieval, a redundant-but-correct chunk is less dangerous than a
diverse-but-irrelevant one. The synthesizer can handle some redundancy; it
cannot handle wrong context.

### Parent Context Expansion

Qdrant stores and searches child chunks (256 tokens). When a child chunk is
the best match, the retriever fetches its parent paragraph (1024 tokens) from
PostgreSQL and passes that to the synthesizer instead.

The synthesizer receives complete, readable paragraphs — not sentence fragments
that happened to contain the matching keywords. This is the single largest
contributor to answer quality in financial document QA.

### Mandatory Tenant Isolation on Every Search

Every Qdrant search includes a payload filter on `tenant_id`. There are no
exceptions — not in development, not in testing, not in fallback paths.

A missing `tenant_id` logs a critical warning and still applies no filter
(searches all data) — this is deliberate so the system degrades visibly rather
than silently blocking all queries. In production the API layer guarantees
`tenant_id` is always set before reaching the retriever.

Research basis: without explicit tenant filtering, 95% of benign queries
triggered cross-tenant data leakage via shared entity connections in a
4-tenant test corpus (arxiv, 2025).

---

## Synthesizer

### Lost-in-the-Middle Reordering

LLMs exhibit a U-shaped attention curve on long context: they attend strongly
to the beginning and end of the context window, weakly to the middle.

Before passing chunks to Claude, the synthesizer reorders them:
- Best chunk → position 1 (strong attention)
- Second-best chunk → last position (strong attention)
- Remaining chunks → middle positions

This ensures the two highest-quality chunks receive the strongest attention
regardless of context length.

Research basis: ICLR 2025 (arxiv 2410.05983) — lost-in-the-middle attention
pattern confirmed across all major LLMs on long-context tasks.

### Numbered Chunks and Inline Citations

Production finding: without explicit chunk numbering, Claude attributes claims
to the wrong source during long-form synthesis. The model generates text from
memory and then finds a nearby chunk to cite — "attributional drift."

Fix: every chunk is labeled `[Chunk #N]` in the prompt. The system prompt
requires every factual claim to include an inline citation in the format
`[Source: filename, page N, Chunk #K]`. The chunk number creates a traceable
link between each claim and its source.

Research basis: FACTUM benchmark (arxiv 2601.05866) — citation hallucination
occurs in 34% of financial RAG answers without explicit chunk referencing.

### Financial Hallucination Guards

Three specific hallucination patterns were identified in financial document QA
(arxiv 2602.05723):
- 55%: temporal confusion — Q3 figure attributed to Q4
- 28%: fiscal/calendar year confusion
- 17%: rounding — exact value becomes an approximation

The synthesizer prompt explicitly forbids: paraphrasing numbers, rounding,
using approximate language, and omitting time periods from financial figures.

### Chunk Truncation at 3,000 Characters in Synthesizer Prompt

```python
_MAX_CHUNK_LEN = 3000

if len(text) > _MAX_CHUNK_LEN:
    text = text[:_MAX_CHUNK_LEN] + "... [truncated]"
```

Parent chunks can be up to 1,024 tokens (~4,000 characters). With 6 parent
chunks, the synthesizer prompt reaches ~24,000 characters before the question
is added — approaching Claude's optimal context window for single-call synthesis.

3,000 characters per chunk keeps the total prompt manageable while retaining
the most important content. Parent chunks front-load their key information
(the parent was built from top-to-bottom reading order), so truncation at
the tail rarely removes critical information.

The `"... [truncated]"` suffix signals to Claude that the chunk was cut. Without
it, Claude might cite a claim from text that doesn't exist in the truncated chunk.

### SQL Path: No Claude Call

For `data_source=sql` queries, the synthesizer formats the database result
directly without calling Claude. There is no LLM synthesis step.

This is not a cost optimization — it is a correctness decision. A database
query returns exact data. Routing that exact data through Claude introduces
a hallucination surface where none is needed. The formatted transaction data
is the answer.

---

## Evaluator

### Not RAGAS

RAGAS is the standard evaluation library for RAG systems. It was evaluated and
rejected.

Three failures on financial document corpora:
1. RAGAS failed to produce scores on 83.5% of FinanceBench examples. Its
   claim-extraction mechanism breaks on numerical financial reasoning.
2. RAGAS uses OpenAI by default. Financial document content cannot leave the
   AWS VPC.
3. RAGAS produces nonsense scores on "I cannot find this information" responses
   — it was not designed for IDK handling.

Decision: LLM-as-judge using Claude Haiku via Bedrock. Haiku stays within the
VPC, handles IDK responses correctly via a special-case rule, and never silently
fails on financial numerical reasoning.

### Two-Layer Evaluation

Deterministic checks run before the LLM judge to avoid unnecessary Bedrock calls:
- Empty answer → 0.0/0.0 (instant)
- "Cannot find" response → 1.0/0.0 (valid IDK, no hallucination, question unanswered)
- Answer with no chunks → 0.0/0.0 (synthesizer answered from memory — not permitted)

Only answers that pass deterministic checks reach the Haiku judge.

### _safe_score — LLM Scores Clamped to [0.0, 1.0]

```python
def _safe_score(value, default: float = 0.7) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default
```

The Haiku judge returns scores as JSON floats. LLMs occasionally return
values outside the expected range — `1.5`, `-0.2`, `"high"`, or `None`.

Without clamping: a faithfulness score of `1.5` passes the threshold check
(`1.5 >= 0.85`) correctly but a score of `-0.2` would always fail even
for perfect answers, causing infinite retries that exhaust the max retry
limit on every query.

`_safe_score` clamps to `[0.0, 1.0]` and returns `0.7` (borderline, triggers
retry) for any non-numeric value. The retry is the conservative safe action —
it never passes a bad answer and never permanently breaks on bad LLM output.

### Evaluator Judge Prompt Truncated at 600 Characters Per Chunk

```python
text = (c.get("text") or "").strip()[:600]
```

The evaluator's Haiku judge receives the question, the answer, and all retrieved
chunks. Parent chunks can be up to 3,000 characters each. With 6 chunks, the
judge prompt reaches ~20,000 characters before the question and answer are added.

Haiku at 256 max output tokens only needs to read enough of each chunk to verify
whether the answer's claims are supported. 600 characters per chunk (roughly
one substantial paragraph) is sufficient for faithfulness evaluation.

Truncating at 600 characters keeps the judge prompt under ~5,000 characters
total — fast, cheap, and within Haiku's effective context window for this task.

### Conservative Fallback When Judge Fails

If the Haiku judge call fails (Bedrock timeout, throttle, JSON parse error),
the evaluator does not crash or return 0/0:

```python
except (ClientError, json.JSONDecodeError, KeyError) as e:
    logger.warning(f"[evaluator] Haiku judge failed — conservative scores (0.7, 0.7)")
    return 0.7, 0.7, f"judge unavailable: {type(e).__name__}"
```

0.7/0.7 sits below the pass threshold (0.85/0.80) — so the graph routes to
a retry. After the retry, if the judge fails again, max retries are reached
and the answer is returned with a confidence warning.

Why 0.7 and not 0.0: returning 0.0/0.0 on judge failure would make the
system look like it has a bad answer when it might have a perfectly good one.
0.7 says "borderline — try once more" which is the correct conservative action.
It never passes a bad answer, and it never wrongly discards a good one.

### SQL Answers Auto-Pass

Answers produced from the `sql` path receive faithfulness=1.0, relevance=1.0
automatically. The data comes directly from a parameterized database query —
hallucination is not possible by construction. Running a faithfulness judge on
database output would be measuring nothing real.

### Max 1 Retry

Research shows that more than one retrieval retry rarely improves answer quality
and risks infinite loops in the graph. After 1 retry, the evaluator appends a
confidence notice to the answer and the graph terminates.

A visible confidence notice is safer than a silent wrong answer. Financial
analysts must verify figures manually anyway; the notice makes uncertainty
explicit rather than hiding it.

---

## Graph Architecture

### increment_retry as a Separate Node

The retry counter `retry_count` is managed exclusively by a dedicated
`increment_retry` node. The evaluator never touches it.

The reason: if the evaluator incremented `retry_count` to signal "retry needed,"
the graph router would immediately read the incremented value and conclude
"retries exhausted → END" — ending the pipeline before the retry ran.

By separating the increment into its own node that runs only on the retry path,
`retry_count` always reflects completed retries. The router reads it after
evaluation, before incrementing — the semantics are unambiguous.

### analyze_query Skipped on Retry

The retry path goes directly from `increment_retry` to `retrieve`, bypassing
`analyze_query`. The question classification, sub-questions, and HyDE query
are set correctly on the first run. Re-analyzing the same question produces
identical output — a wasted Bedrock call on the hot path.

The retry targets the retriever specifically: re-running Qdrant search may
surface different chunks if the first search had transient ranking variance.

### Three-Path Routing After analyze_query

The conditional edge after `analyze_query` routes based on `data_source`:
- `sql` or `hybrid` → `db_lookup` (structured data needed)
- everything else → `retrieve` (document search only)

A second conditional edge after `db_lookup` routes:
- `hybrid` → `retrieve` (also need document chunks)
- `sql` → `synthesize` (database result is sufficient)

This keeps the graph linear and readable while supporting all three data paths.

---

## API (FastAPI)

### Liveness vs Readiness — Two Separate Probes

```
GET /health → always returns 200 if the process is alive
GET /ready  → checks Qdrant + PostgreSQL connectivity
```

These solve different problems and must not be merged into one endpoint.

`/health` (liveness): is the process running? If this fails, Kubernetes/ECS
restarts the container. It never checks dependencies — a slow Qdrant should not
cause the API container to restart. It just needs to know the process is alive.

`/ready` (readiness): can the process serve traffic? If this fails, the load
balancer stops sending requests to this instance but does NOT restart it. The
instance waits for its dependencies to recover.

Merging them into one endpoint means a Qdrant outage restarts all API containers
in a loop instead of quietly taking them out of rotation until Qdrant recovers.

### tenant_id Character Whitelist — Input Sanitization at the Boundary

```python
@field_validator("tenant_id")
def tenant_id_not_empty(cls, v):
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_-")
    if not all(c in allowed for c in v):
        raise ValueError("tenant_id contains invalid characters")
    return v
```

`tenant_id` is used directly in Qdrant payload filters and PostgreSQL queries.
Even though psycopg2 parameterization prevents SQL injection, a `tenant_id`
containing `../`, null bytes, or control characters could:
- Create confusing log entries that obscure audit trails
- Potentially exploit filter logic in less-mature vector DBs
- Fail in unexpected ways if the value reaches a file system path

The whitelist enforces `[a-z0-9_-]` only — a character set that is safe in
all contexts: SQL, JSON, URLs, log files, and Qdrant filter values.
Validation happens at the API boundary before the value touches any system.

### Generic Error Handler — Never Expose Stack Traces

```python
@app.exception_handler(Exception)
async def generic_error_handler(request, exc):
    logger.error(f"Unhandled exception: {type(exc).__name__}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
```

Without this handler, FastAPI returns Python stack traces on unhandled exceptions.
Stack traces contain: file paths, function names, variable values, library
versions. This information is directly useful for identifying attack vectors.

The handler logs the full exception internally (visible in CloudWatch) and
returns only a generic message to the caller. The engineer gets the debug
information; the attacker gets nothing.

### Retry-After Header on 429 Responses

```python
raise HTTPException(
    status_code=429,
    detail=f"Rate limit exceeded. Max {_RATE_LIMIT} requests per minute.",
    headers={"Retry-After": str(_RATE_WINDOW_S)},
)
```

A 429 response without `Retry-After` forces the client to guess when to retry.
Aggressive clients retry immediately and receive another 429, burning request
slots and adding noise to logs.

`Retry-After: 60` tells the client exactly how long to wait. Well-behaved HTTP
clients (and any SDK built on top of them) honour this header automatically.
The server's rate limit window is respected without the client needing custom
backoff logic.

This is standard HTTP protocol (RFC 6585). Omitting it is technically correct
but operationally careless.

### Sliding Window Rate Limiting — Per API Key

```python
_RATE_WINDOW_S = 60
_RATE_LIMIT    = 10   # requests per minute per key

def _check_rate_limit(api_key):
    now = time.time()
    bucket = _rate_buckets[api_key]
    while bucket and bucket[0] < now - _RATE_WINDOW_S:
        bucket.popleft()    # evict timestamps outside the window
    if len(bucket) >= _RATE_LIMIT:
        raise HTTPException(429, ...)
    bucket.append(now)
```

Fixed window rate limiting (reset counter every 60s) allows burst abuse: send
10 requests at 23:59:50, counter resets at midnight, send 10 more at 00:00:00 —
20 requests in 20 seconds while staying within the limit.

Sliding window: any 60-second window contains at most 10 requests. The deque
holds timestamps of recent requests. Timestamps older than 60 seconds are evicted.
No burst exploitation possible.

Note: this is in-memory per-instance. Phase B replaces it with Redis-backed
rate limiting shared across all ECS instances.

## DB Lookup

### _ROW_LIMIT = 50 — Hard Cap on All DB Queries

```python
_ROW_LIMIT = 50
rows = cur.fetchmany(_ROW_LIMIT)
```

Without a row cap, a `SELECT * FROM transactions WHERE tenant_id=%s` query
on a customer with years of transaction history returns thousands of rows.
All of them get sent through the synthesizer prompt — a prompt that could
reach hundreds of thousands of tokens.

The 50-row hard cap applies at the database layer (`fetchmany`), not just at
the intent parameter layer. Even if Claude requests 200 recent transactions,
the database never returns more than 50. This prevents:
- Accidental data dumps to the caller
- Prompt overflow in the synthesizer
- Excessive data exposure in a single API response

50 rows covers every practical use case: nobody needs 50 transactions listed
in a single answer. If they do, that is a reporting query that belongs in a
dedicated reporting system, not a conversational RAG pipeline.

### Template-Based SQL, Not Raw Text2SQL

The initial design called for Claude to generate arbitrary SQL from natural
language. This was rejected on security grounds.

A Text2SQL approach where the LLM output is executed directly creates multiple
risks: SQL injection via adversarial prompts, accidental data deletion if
output validation fails, and cross-tenant data access if tenant filters are
placed inside the LLM-generated WHERE clause.

Decision: Claude classifies the question into one of 6 predefined intents
(balance, recent, flagged, spending_period, by_category, by_merchant). The
application executes a pre-written parameterized SQL template for that intent.

`tenant_id` and `customer_id` are never part of the LLM output. They are always
injected by the application as psycopg2 parameters (`%s`). SQL injection via
merchant names or category strings is prevented by the same parameterization.
