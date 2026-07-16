"""
agents/db_lookup.py — Structured data lookup via intent classification + safe SQL.

Handles queries that need exact database answers rather than semantic search:
  "What is my current balance?"
  "Show me my transactions from last week"
  "Do I have any flagged transactions?"
  "How much did I spend on food this month?"

Design:
  Two-step approach — safer than raw Text2SQL:
  1. Claude Haiku classifies the question into a known intent + extracts params.
  2. We execute a pre-written, parameterized SQL template for that intent.
     tenant_id and customer_id are NEVER in the LLM output — always injected by code.

Security invariants (never broken):
  - SQL is never constructed from raw LLM text — only intent + numeric/string params
  - tenant_id and customer_id are ALWAYS injected as psycopg2 parameters (%s)
  - Only SELECT queries — no DDL/DML possible by design (no template allows writes)
  - Merchant/category strings from LLM are passed as %s params (SQL injection proof)
  - All results capped at 50 rows max

Failure handling:
  - Bedrock timeout / error → set error in state, return None sql_result
  - DB connection error    → set error in state, return None sql_result
  - Unknown intent        → fallback to "recent" (always safe)
  - Empty result          → return "No transactions found" string (not an error)
"""

import json
import os
import re
from typing import Optional

import boto3
import psycopg2
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from .logger import get_logger
from .state import RAGState

load_dotenv()
logger = get_logger(__name__)

_HAIKU_MODEL = os.getenv("BEDROCK_HAIKU_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
_MAX_TOKENS  = 256
_ROW_LIMIT   = 50   # hard cap — no query returns more than this

# ── SQL templates — tenant_id and customer_id always injected as params ───────
#
# Each template is (sql_string, [extra_param_names]).
# The base params are always (tenant_id, customer_id).
# Extra params are appended in the listed order.

_TEMPLATES: dict[str, tuple[str, list[str]]] = {
    "balance": (
        """
        SELECT
            SUM(amount) AS balance,
            COUNT(*) AS total_transactions,
            SUM(CASE WHEN type = 'credit' THEN amount ELSE 0 END) AS total_credits,
            SUM(CASE WHEN type = 'debit'  THEN ABS(amount) ELSE 0 END) AS total_debits
        FROM transactions
        WHERE tenant_id = %s AND customer_id = %s
        """,
        [],
    ),
    "recent": (
        """
        SELECT date, merchant, amount, type, category, description, flagged
        FROM transactions
        WHERE tenant_id = %s AND customer_id = %s
        ORDER BY date DESC
        LIMIT %s
        """,
        ["limit"],
    ),
    "flagged": (
        """
        SELECT date, merchant, amount, type, description
        FROM transactions
        WHERE tenant_id = %s AND customer_id = %s AND flagged = true
        ORDER BY date DESC
        LIMIT 20
        """,
        [],
    ),
    "spending_period": (
        """
        SELECT
            SUM(ABS(amount)) AS total_spent,
            COUNT(*)    AS num_transactions,
            MIN(date)   AS from_date,
            MAX(date)   AS to_date
        FROM transactions
        WHERE tenant_id = %s AND customer_id = %s
          AND type = 'debit'
          AND date >= CURRENT_DATE - %s * INTERVAL '1 day'
        """,
        ["days"],
    ),
    "by_category": (
        """
        SELECT
            category,
            SUM(ABS(amount))  AS total,
            COUNT(*)     AS count
        FROM transactions
        WHERE tenant_id = %s AND customer_id = %s AND type = 'debit'
        GROUP BY category
        ORDER BY total DESC
        LIMIT 20
        """,
        [],
    ),
    "by_merchant": (
        """
        SELECT date, amount, type, description, flagged
        FROM transactions
        WHERE tenant_id = %s AND customer_id = %s
          AND merchant ILIKE %s
        ORDER BY date DESC
        LIMIT 20
        """,
        ["merchant"],
    ),
}

_DEFAULT_INTENT = "recent"
_DEFAULT_LIMIT  = 10
_DEFAULT_DAYS   = 30


# ── Intent classification via Claude Haiku ────────────────────────────────────

_SYSTEM = """You classify financial account questions into query intents. Return ONLY valid JSON.

Available intents:
- "balance"         : current account balance, total money, how much in account
- "recent"          : recent transactions, transaction history, last N transactions
- "flagged"         : suspicious, flagged, unusual, or blocked transactions
- "spending_period" : total spending in last N days / this week / this month
- "by_category"     : spending breakdown by category, what category spent most
- "by_merchant"     : transactions at a specific store/merchant/company

Return this exact JSON:
{
  "intent": "<one of the intents above>",
  "limit": <integer 1-20, default 10>,
  "days": <integer 1-365, default 30>,
  "merchant": "<merchant name or null>"
}"""


def _classify_intent(question: str) -> dict:
    """Ask Haiku to classify the question into a known intent + extract params."""
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": _MAX_TOKENS,
        "messages": [
            {
                "role": "user",
                "content": f"{_SYSTEM}\n\nQuestion: \"{question}\"",
            }
        ],
    })
    client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))
    response = client.invoke_model(
        modelId=_HAIKU_MODEL,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    raw  = json.loads(response["body"].read())
    text = raw["content"][0]["text"].strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


def _safe_intent(raw: dict) -> dict:
    """Validate and sanitise the Haiku classification output."""
    intent   = raw.get("intent", _DEFAULT_INTENT)
    if intent not in _TEMPLATES:
        logger.warning(f"[db_lookup] Unknown intent '{intent}' — falling back to 'recent'")
        intent = _DEFAULT_INTENT

    limit    = int(raw.get("limit") or _DEFAULT_LIMIT)
    limit    = max(1, min(limit, _ROW_LIMIT))

    days     = int(raw.get("days") or _DEFAULT_DAYS)
    days     = max(1, min(days, 365))

    merchant = raw.get("merchant") or None
    if merchant:
        # strip any SQL metacharacters — the value goes through %s anyway,
        # but clean it here for good measure
        merchant = str(merchant)[:100].replace("%", "").replace("_", " ").strip()
        merchant = f"%{merchant}%"   # ILIKE pattern

    return {"intent": intent, "limit": limit, "days": days, "merchant": merchant}


def _fallback_intent(question: str) -> dict:
    """
    Deterministic fallback when Haiku intent classification is unavailable.
    Keeps common account-data questions useful during Bedrock quota throttling.
    """
    q = question.lower()

    if any(s in q for s in ("balance", "account total", "how much money")):
        intent = "balance"
    elif any(s in q for s in ("flagged", "suspicious", "unusual", "blocked")):
        intent = "flagged"
    elif "category" in q or "breakdown" in q:
        intent = "by_category"
    elif any(s in q for s in ("spending", "spent", "this month", "this week")):
        intent = "spending_period"
    else:
        intent = "recent"

    limit = _DEFAULT_LIMIT
    m = re.search(r"\b(?:last|recent)\s+(\d{1,2})\b", q)
    if m:
        limit = max(1, min(int(m.group(1)), _ROW_LIMIT))

    days = _DEFAULT_DAYS
    if "week" in q:
        days = 7
    elif "month" in q:
        days = 30

    return {"intent": intent, "limit": limit, "days": days, "merchant": None}


# ── SQL execution ─────────────────────────────────────────────────────────────


def _run_query(tenant_id: str, customer_id: str, intent_params: dict) -> list[dict]:
    """
    Execute the parameterized SQL template for the given intent.
    tenant_id and customer_id are ALWAYS the first two params — never from LLM.

    Returns list of row dicts. Empty list if no rows found.
    Raises on DB connection error (caller handles).
    """
    intent  = intent_params["intent"]
    sql, extra_keys = _TEMPLATES[intent]

    # Build param list: base (tenant, customer) + intent-specific extras
    params: list = [tenant_id, customer_id]
    for key in extra_keys:
        if key == "limit":
            params.append(intent_params["limit"])
        elif key == "days":
            params.append(intent_params["days"])
        elif key == "merchant":
            params.append(intent_params["merchant"] or "%")

    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set")

    conn = psycopg2.connect(url)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [desc[0] for desc in cur.description]
            rows = cur.fetchmany(_ROW_LIMIT)
        return [dict(zip(cols, row)) for row in rows]
    finally:
        conn.close()


# ── Result formatting ─────────────────────────────────────────────────────────


def _format_result(intent: str, rows: list[dict], params: dict) -> str:
    """Convert DB rows into a human-readable string for the synthesizer."""
    if not rows:
        return "No transactions found matching your query."

    if intent == "balance":
        r = rows[0]
        bal      = float(r.get("balance") or 0)
        credits  = float(r.get("total_credits") or 0)
        debits   = float(r.get("total_debits") or 0)
        n        = int(r.get("total_transactions") or 0)
        sign     = "+" if bal >= 0 else ""
        return (
            f"Current account balance: {sign}${bal:,.2f}\n"
            f"Total credits received: ${credits:,.2f}\n"
            f"Total debits made: ${debits:,.2f}\n"
            f"Total transactions on record: {n}"
        )

    if intent == "spending_period":
        r     = rows[0]
        total = float(r.get("total_spent") or 0)
        count = int(r.get("num_transactions") or 0)
        days  = params["days"]
        return (
            f"Spending in the last {days} days: ${total:,.2f}\n"
            f"Number of debit transactions: {count}\n"
            f"Average per transaction: ${(total / count if count else 0):,.2f}"
        )

    if intent == "by_category":
        lines = ["Spending breakdown by category:"]
        for r in rows:
            cat   = r.get("category", "Unknown")
            total = float(r.get("total") or 0)
            count = int(r.get("count") or 0)
            lines.append(f"  {cat}: ${total:,.2f} ({count} transactions)")
        return "\n".join(lines)

    if intent == "flagged":
        lines = [f"Flagged / suspicious transactions ({len(rows)} found):"]
        for r in rows:
            flag_str = " ⚠️ FLAGGED" if r.get("flagged") else ""
            lines.append(
                f"  {r['date']} | {r['merchant']} | ${abs(float(r['amount'])):,.2f} "
                f"({r['type']}){flag_str}"
            )
        if not rows:
            lines = ["No flagged transactions found on your account."]
        return "\n".join(lines)

    # recent / by_merchant — tabular
    lines = [f"Transactions ({len(rows)} results):"]
    for r in rows:
        amount   = abs(float(r.get("amount") or 0))
        flag_str = " ⚠️" if r.get("flagged") else ""
        desc     = f" — {r['description']}" if r.get("description") else ""
        lines.append(
            f"  {r['date']} | {r['merchant']} | ${amount:,.2f} ({r.get('type','?')})"
            f"{desc}{flag_str}"
        )
    return "\n".join(lines)


# ── LangGraph node ────────────────────────────────────────────────────────────


def db_lookup(state: RAGState) -> RAGState:
    """
    LangGraph node: classify question into SQL intent and fetch from transactions table.

    Reads:  state["question"], state["tenant_id"], state["customer_id"]
    Writes: state["sql_result"], state["error"]

    Non-fatal: on any failure, sets sql_result=None and error string.
    The synthesizer handles None sql_result with a "cannot find" response.
    """
    question    = (state.get("question")    or "").strip()
    tenant_id   = (state.get("tenant_id")   or "demo").strip()
    customer_id = (state.get("customer_id") or "demo_user_001").strip()

    if not question:
        return {**state, "sql_result": None, "error": "db_lookup: empty question"}

    logger.info(f"[db_lookup] tenant={tenant_id} customer={customer_id} question='{question[:80]}'")

    try:
        # Step 1: classify intent
        try:
            raw_intent = _classify_intent(question)
            intent_params = _safe_intent(raw_intent)
        except (ClientError, json.JSONDecodeError, KeyError) as e:
            logger.warning(
                f"[db_lookup] Haiku classification failed ({type(e).__name__}: {e}) "
                "— using deterministic fallback intent"
            )
            intent_params = _fallback_intent(question)

        logger.info(
            f"[db_lookup] intent={intent_params['intent']} "
            f"limit={intent_params['limit']} days={intent_params['days']} "
            f"merchant={intent_params['merchant']}"
        )

        # Step 2: execute safe SQL
        rows = _run_query(tenant_id, customer_id, intent_params)

        # Step 3: format result
        result_text = _format_result(intent_params["intent"], rows, intent_params)
        logger.info(f"[db_lookup] returned {len(rows)} rows for intent={intent_params['intent']}")

        return {**state, "sql_result": result_text, "error": None}

    except psycopg2.Error as e:
        msg = f"db_lookup: PostgreSQL error: {type(e).__name__}: {e}"
        logger.error(f"[db_lookup] {msg}")
        return {**state, "sql_result": None, "error": msg}

    except Exception as e:
        msg = f"db_lookup: unexpected error: {type(e).__name__}: {e}"
        logger.error(f"[db_lookup] {msg}")
        return {**state, "sql_result": None, "error": msg}


# ── Self-check ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Tests non-network logic only.

    # 1. _safe_intent clamps limit to ROW_LIMIT
    p = _safe_intent({"intent": "recent", "limit": 9999, "days": 7, "merchant": None})
    assert p["limit"] == _ROW_LIMIT, f"Expected {_ROW_LIMIT}, got {p['limit']}"

    # 2. Unknown intent falls back to 'recent'
    p = _safe_intent({"intent": "UNKNOWN_INTENT", "limit": 5})
    assert p["intent"] == _DEFAULT_INTENT

    # 3. Negative days clamped to 1
    p = _safe_intent({"intent": "spending_period", "days": -10})
    assert p["days"] == 1

    # 4. Merchant string gets ILIKE wrapping
    p = _safe_intent({"intent": "by_merchant", "merchant": "Starbucks"})
    assert p["merchant"] == "%Starbucks%"

    # 5. SQL metacharacter stripped from merchant
    p = _safe_intent({"intent": "by_merchant", "merchant": "Store%100"})
    assert "%" not in p["merchant"].replace("%Storeitem100%", "").replace("%Store100%", "")

    # 6. Balance format — correct output structure
    fake_balance_rows = [{"balance": 5842.51, "total_credits": 9000.00,
                          "total_debits": 3157.49, "total_transactions": 8}]
    result = _format_result("balance", fake_balance_rows, {})
    assert "$5,842.51" in result
    assert "$9,000.00" in result

    # 7. Empty rows → "No transactions found"
    result = _format_result("recent", [], {"limit": 10})
    assert "No transactions found" in result

    # 8. Flagged format includes ⚠️ marker
    fake_flagged = [{"date": "2026-06-26", "merchant": "FX Merchant", "amount": -340.00,
                     "type": "debit", "description": "Foreign fee", "flagged": True}]
    result = _format_result("flagged", fake_flagged, {})
    assert "⚠️" in result
    assert "FX Merchant" in result

    print("db_lookup.py self-check passed — intent clamping, merchant safety, result formatting all correct")
