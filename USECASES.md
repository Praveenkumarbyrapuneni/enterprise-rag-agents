# Use Cases

A practical guide to what this system answers, how to integrate it, and how to extend it to new institutions.

---

## The Core Problem This Solves

Financial institutions have two types of knowledge that customers ask about:

1. **Live data** — balances, transactions, account history (lives in a database)
2. **Document knowledge** — fee policies, product terms, annual reports, regulatory filings (lives in PDFs and documents)

Most systems handle one or the other. This system handles both — automatically routing each question to the right source, or combining both when needed.

---

## The Three Query Paths

### Path 1 — SQL (Live Account Data)

Questions that need exact numbers from the database.

| Customer asks | System does |
|---|---|
| "What is my current balance?" | Queries the transactions table, sums credits minus debits |
| "Show me my last 5 transactions" | Fetches recent rows ordered by date |
| "Do I have any flagged or suspicious activity?" | Queries flagged=true rows |
| "How much did I spend this month?" | Aggregates debit amounts for the current month |

**Why not RAG for these?** Documents don't have your real-time balance. The database does. Asking an LLM to "retrieve" your balance from a document would hallucinate a number.

---

### Path 2 — RAG (Document Search)

Questions that need knowledge from ingested documents.

| Customer asks | System does |
|---|---|
| "What are the risk factors in the annual report?" | Finds relevant chunks from the SEC filing, returns cited answer |
| "What is the foreign transaction fee policy?" | Retrieves the fee policy section from product documents |
| "How did the company perform in Q3?" | Pulls the earnings report chunks that answer the question |
| "What are the LIBOR transition risks?" | Retrieves regulatory compliance sections from filings |

**Why not SQL for these?** The database has no knowledge of fee policies or regulatory filings. That knowledge lives in documents that must be searched semantically.

---

### Path 3 — Hybrid (Both Combined)

Questions that need live data AND document knowledge together.

| Customer asks | System does |
|---|---|
| "Why was I charged a $340 fee on June 26?" | Fetches the June 26 transaction from DB + retrieves the fee policy document → Claude explains both |
| "Is this charge compliant with your fee schedule?" | Looks up the specific charge + pulls the fee schedule document → synthesizes an answer |
| "Was my foreign transaction fee correct?" | Gets the exact transaction amount + retrieves the foreign fee policy → compares and explains |

**Why hybrid?** Neither source alone can answer this. The DB tells you what happened. The document tells you why it's allowed. The LLM connects them.

---

## How to Integrate This Into an Existing System

### Scenario: A financial institution wants to add an AI assistant to their mobile app

**Step 1 — Load your users**

If you already have a users database, migrate them into the `users` table:

```sql
INSERT INTO users (username, email, password_hash, tenant_id, customer_id)
SELECT username, email, hashed_password, 'your_institution', customer_id
FROM your_existing_users_table;
```

Every user gets `tenant_id = 'your_institution'` — set by your backend, never by the user.

**Step 2 — Load your documents**

Place your institution's documents (PDFs, HTML filings, Word docs) in the `tests/` folder and update `scripts/reingest_with_tenants.py`:

```python
TENANT_MAP = {
    "annual_report_2025.pdf": "your_institution",
    "fee_schedule.pdf":       "your_institution",
    "product_terms.htm":      "your_institution",
}
```

Run ingestion:
```bash
python -m scripts.reingest_with_tenants
```

**Step 3 — Load your transaction data**

Insert your customers' transactions into the `transactions` table in the same format as the seed data in `db/schema.py`. The system queries this table for all SQL-path questions.

**Step 4 — Connect your mobile app**

Your mobile app calls three endpoints:

```
1. POST /auth/login
   → user sends username + password
   → system returns JWT token

2. POST /query  (Authorization: Bearer <token>)
   → user sends their question
   → system returns grounded answer with citations

3. GET /health  (no auth)
   → load balancer probe
```

The `tenant_id` is never in the mobile app request. It flows from the JWT automatically.

---

## What Happens When a New Customer Signs Up

Your institution's backend handles the signup flow. When a new customer is created in your system, your backend also calls:

```
POST /auth/register
{
  "username":    "new_customer_username",
  "email":       "customer@email.com",
  "password":    "their_password",
  "tenant_id":   "your_institution",   ← always set by your backend
  "customer_id": "CUST_12345"          ← links to their transactions
}
```

The customer never sees or touches `tenant_id`. Your backend always sets it to your institution's identifier. After registration, the customer logs in normally and their token carries the correct tenant identity automatically.

---

## How Tenant Isolation Works

Every vector in Qdrant has `tenant_id` in its payload. Every search applies a filter:

```python
Filter(must=[FieldCondition(key="tenant_id", match=MatchValue(value="your_institution"))])
```

Institution A's customers can only search Institution A's documents. Institution B's documents are not scanned — the filter is enforced at the vector database level, not in application logic. Even a bug in the application layer cannot leak data across tenants.

---

## Extending to a New Document Type

The parser supports 9 formats out of the box: PDF, Word (DOCX), Excel (XLSX/CSV), HTML, plain text, images (PNG/JPG/TIFF), and email (EML/MSG).

To add a new document:
1. Place it in the `tests/` folder
2. Add it to `TENANT_MAP` in `scripts/reingest_with_tenants.py`
3. Run `python -m scripts.reingest_with_tenants`

The parser auto-detects format by file extension. No code changes needed for supported formats.

---

## Scaling to Millions of Users

The system is designed to handle this without code changes:

| Bottleneck | Current (laptop) | At scale (AWS) |
|---|---|---|
| Vector search | Qdrant single node | Qdrant multi-node cluster (horizontal) |
| Database | PostgreSQL single instance | RDS read replicas |
| Worker pool | Celery local | ECS auto-scaling worker fleet |
| API | Single uvicorn process | ECS multi-instance behind ALB |
| Rate limiting | In-memory per process | Redis-backed sliding window |

All changes are `.env` only — no code rewrites.
