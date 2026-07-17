"""
db/schema.py — PostgreSQL schema: tenant registry, transactions, feedback, retrieval params.

Run once before starting the API:
    python -m db.schema

Safe to re-run: all creates use IF NOT EXISTS, inserts use ON CONFLICT DO NOTHING.
"""

import hashlib
import os
import secrets
import sys

import psycopg2
from dotenv import load_dotenv

load_dotenv()

# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA = """
-- Maps email domain → tenant_id. Used by FastAPI to identify caller's tenant.
CREATE TABLE IF NOT EXISTS tenant_registry (
    email_domain  TEXT         PRIMARY KEY,
    tenant_id     TEXT         NOT NULL,
    company_name  TEXT         NOT NULL,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_tenant_registry_tenant_id
    ON tenant_registry (tenant_id);

-- SHA-256 hash of the tenant's server-to-server secret, used to authenticate
-- their backend (never the customer's browser) when calling
-- POST /internal/mint-customer-token. NULL = tenant not yet provisioned for
-- this integration. See db.schema.provision_service_key().
ALTER TABLE tenant_registry ADD COLUMN IF NOT EXISTS service_key_hash TEXT;

-- Immutable query audit log — required for SEC/FINRA compliance.
-- Every AI response must log: what was asked, what data was used, what was returned.
-- Never delete rows from this table. Regulatory retention minimum is 7 years.
CREATE TABLE IF NOT EXISTS query_audit_log (
    id              UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       TEXT         NOT NULL,
    customer_id     TEXT         NOT NULL,
    question        TEXT         NOT NULL,
    query_type      TEXT,
    data_source     TEXT,
    answer          TEXT,
    sources         JSONB        NOT NULL DEFAULT '[]',
    faithfulness    NUMERIC(4,3),
    relevance       NUMERIC(4,3),
    latency_ms      NUMERIC(10,2),
    model_id        TEXT,
    error           TEXT,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_query_audit_tenant_created
    ON query_audit_log (tenant_id, created_at DESC);

-- Users: one row per login. tenant_id links them to their company's data.
-- password_hash is NULL for auth_provider='sso' — the company's own identity
-- provider verifies the password, we never see or store one for those users.
CREATE TABLE IF NOT EXISTS users (
    user_id       UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    username      TEXT         UNIQUE NOT NULL,
    email         TEXT         UNIQUE NOT NULL,
    password_hash TEXT,
    auth_provider TEXT         NOT NULL DEFAULT 'local' CHECK (auth_provider IN ('local', 'sso')),
    tenant_id     TEXT         NOT NULL,
    customer_id   TEXT         NOT NULL,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
    CONSTRAINT password_required_for_local CHECK (auth_provider = 'sso' OR password_hash IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS idx_users_tenant_id ON users (tenant_id);

-- Migration for pre-existing databases: add SSO support to a users table that
-- was created before auth_provider existed. ponytail: no CHECK-constraint
-- retrofit on existing tables (Postgres has no ADD CONSTRAINT IF NOT EXISTS) —
-- the invariant is enforced at the application layer (api/auth.py, api/sso.py)
-- for rows created after this migration; fresh installs get the CHECK above.
ALTER TABLE users ADD COLUMN IF NOT EXISTS auth_provider TEXT NOT NULL DEFAULT 'local';
ALTER TABLE users ALTER COLUMN password_hash DROP NOT NULL;

-- Mock financial transactions for hybrid RAG demo.
-- In production: replaced by a Kafka → PostgreSQL real-time pipeline.
CREATE TABLE IF NOT EXISTS transactions (
    id            UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id     TEXT          NOT NULL,
    customer_id   TEXT          NOT NULL,
    amount        NUMERIC(12,2) NOT NULL,
    merchant      TEXT          NOT NULL,
    category      TEXT          NOT NULL,
    date          DATE          NOT NULL,
    type          TEXT          NOT NULL CHECK (type IN ('debit', 'credit')),
    description   TEXT,
    flagged       BOOLEAN       NOT NULL DEFAULT false,
    created_at    TIMESTAMPTZ   NOT NULL DEFAULT now()
);
-- Composite index covers the WHERE tenant_id=X AND customer_id=Y pattern
CREATE INDEX IF NOT EXISTS idx_transactions_tenant_customer
    ON transactions (tenant_id, customer_id);
-- Descending date for recent-transactions queries
CREATE INDEX IF NOT EXISTS idx_transactions_date
    ON transactions (tenant_id, customer_id, date DESC);
-- Partial index for flagged-only queries (most rows are unflagged — avoid full scan)
CREATE INDEX IF NOT EXISTS idx_transactions_flagged
    ON transactions (tenant_id, customer_id) WHERE flagged = true;

-- Seed idempotency: older versions had no natural uniqueness, so re-running
-- this script inserted duplicate mock transactions. Keep one copy of each
-- seeded transaction, then enforce a unique index so ON CONFLICT DO NOTHING
-- actually works on future runs.
DELETE FROM transactions a
USING transactions b
WHERE a.ctid < b.ctid
  AND a.tenant_id = b.tenant_id
  AND a.customer_id = b.customer_id
  AND a.amount = b.amount
  AND a.merchant = b.merchant
  AND a.category = b.category
  AND a.date = b.date
  AND a.type = b.type
  AND COALESCE(a.description, '') = COALESCE(b.description, '')
  AND a.flagged = b.flagged;

CREATE UNIQUE INDEX IF NOT EXISTS idx_transactions_seed_unique
    ON transactions (
        tenant_id, customer_id, amount, merchant, category, date, type,
        COALESCE(description, ''), flagged
    );

-- User feedback on RAG answers. Links back to query_audit_log for full context.
CREATE TABLE IF NOT EXISTS query_feedback (
    id          UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id    UUID         NOT NULL REFERENCES query_audit_log(id) ON DELETE CASCADE,
    tenant_id   TEXT         NOT NULL,
    helpful     BOOLEAN      NOT NULL,
    comment     TEXT,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_feedback_tenant_created
    ON query_feedback (tenant_id, created_at DESC);

-- Chunks delivered to the synthesizer per query. Soft link via query_id (no FK
-- to avoid slowing down retrieval with a constraint check on every insert).
-- Used to build reranker training data: cited=true chunks are positives, false are negatives.
CREATE TABLE IF NOT EXISTS query_chunks (
    id          UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id    UUID         NOT NULL,
    tenant_id   TEXT         NOT NULL,
    chunk_text  TEXT         NOT NULL,
    source      TEXT         NOT NULL,
    score       NUMERIC(6,4),
    cited       BOOLEAN      NOT NULL DEFAULT false,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_query_chunks_query_id ON query_chunks (query_id);
CREATE INDEX IF NOT EXISTS idx_query_chunks_tenant   ON query_chunks (tenant_id, created_at DESC);

-- Per-tenant tuned retrieval parameters. Updated automatically by feedback tuner.
-- Falls back to hardcoded defaults in retriever.py if no row exists for a tenant.
CREATE TABLE IF NOT EXISTS retrieval_params (
    tenant_id        TEXT         PRIMARY KEY,
    top_k            INTEGER      NOT NULL DEFAULT 20,
    rerank_top_n     INTEGER      NOT NULL DEFAULT 8,
    mmr_final_k      INTEGER      NOT NULL DEFAULT 6,
    feedback_count   INTEGER      NOT NULL DEFAULT 0,
    updated_at       TIMESTAMPTZ  NOT NULL DEFAULT now()
);
"""

# ── Seed data ─────────────────────────────────────────────────────────────────

_SEED = """
-- Tenant registry: email domain → tenant mapping
INSERT INTO tenant_registry (email_domain, tenant_id, company_name) VALUES
    ('goldmansachs.com', 'goldman', 'Goldman Sachs'),
    ('apple.com',        'apple',   'Apple Inc.'),
    ('jpmchase.com',     'jpmorgan','JPMorgan Chase'),
    ('gmail.com',        'demo',    'Demo User')
ON CONFLICT (email_domain) DO NOTHING;

-- Apple Inc. transactions
INSERT INTO transactions
    (tenant_id, customer_id, amount, merchant, category, date, type, description, flagged)
VALUES
    ('apple','apple_user_001', -45.00,    'Starbucks',       'Food & Drink',  '2026-06-29','debit', 'Coffee purchase',              false),
    ('apple','apple_user_001', -1200.00,  'Apple Store',     'Electronics',   '2026-06-28','debit', 'iPhone accessory',             false),
    ('apple','apple_user_001',  8500.00,  'Payroll Direct',  'Income',        '2026-06-27','credit','Monthly salary',               false),
    ('apple','apple_user_001', -340.00,   'FX Merchant Ltd', 'International', '2026-06-26','debit', 'Foreign transaction fee',      true),
    ('apple','apple_user_001', -89.99,    'Netflix',         'Entertainment', '2026-06-25','debit', 'Monthly subscription',         false),
    ('apple','apple_user_001', -2100.00,  'Rent Corp',       'Housing',       '2026-06-24','debit', 'Monthly rent',                 false),
    ('apple','apple_user_001', -67.50,    'Whole Foods',     'Groceries',     '2026-06-23','debit', 'Grocery shopping',             false),
    ('apple','apple_user_001',  500.00,   'Transfer In',     'Transfer',      '2026-06-22','credit','Bank transfer received',        false),
    ('apple','apple_user_001', -230.00,   'Delta Airlines',  'Travel',        '2026-06-21','debit', 'Business flight',              false),
    ('apple','apple_user_001', -15.99,    'Spotify',         'Entertainment', '2026-06-20','debit', 'Music subscription',           false)
ON CONFLICT DO NOTHING;

-- Goldman Sachs transactions
INSERT INTO transactions
    (tenant_id, customer_id, amount, merchant, category, date, type, description, flagged)
VALUES
    ('goldman','goldman_user_001', -200.00,    'Amazon',          'Shopping',    '2026-06-29','debit', 'Office supplies',              false),
    ('goldman','goldman_user_001',  50000.00,  'Bonus Payment',   'Income',      '2026-06-28','credit','Q2 performance bonus',         false),
    ('goldman','goldman_user_001', -5000.00,   'Bloomberg LP',    'Professional','2026-06-27','debit', 'Terminal subscription',        false),
    ('goldman','goldman_user_001', -180.00,    'Business Dinner', 'Food & Drink','2026-06-26','debit', 'Client entertainment',         false),
    ('goldman','goldman_user_001',  125000.00, 'Wire Transfer',   'Transfer',    '2026-06-25','credit','Investment proceeds',           false),
    ('goldman','goldman_user_001', -12000.00,  'Luxury Rentals',  'Housing',     '2026-06-24','debit', 'Monthly rent',                 false),
    ('goldman','goldman_user_001', -3500.00,   'First Class Air', 'Travel',      '2026-06-23','debit', 'Business travel',              false),
    ('goldman','goldman_user_001', -890.00,    'Equinox',         'Health',      '2026-06-22','debit', 'Annual gym membership',        false)
ON CONFLICT DO NOTHING;

-- JPMorgan Chase transactions
INSERT INTO transactions
    (tenant_id, customer_id, amount, merchant, category, date, type, description, flagged)
VALUES
    ('jpmorgan','jpmorgan_user_001', -150.00,   'Uber',             'Transport',   '2026-06-29','debit', 'Business travel',              false),
    ('jpmorgan','jpmorgan_user_001',  75000.00, 'Salary Deposit',   'Income',      '2026-06-28','credit','Monthly salary',               false),
    ('jpmorgan','jpmorgan_user_001', -3200.00,  'Investment Acc',   'Investment',  '2026-06-27','debit', 'Portfolio contribution',       false),
    ('jpmorgan','jpmorgan_user_001', -890.00,   'Electronics Co',   'Electronics', '2026-06-26','debit', 'Work equipment',               false),
    ('jpmorgan','jpmorgan_user_001', -45.00,    'Starbucks',        'Food & Drink','2026-06-25','debit', 'Coffee',                       false),
    ('jpmorgan','jpmorgan_user_001',  2000.00,  'Refund Corp',      'Refund',      '2026-06-24','credit','Purchase refund',               false),
    ('jpmorgan','jpmorgan_user_001', -560.00,   'Marriott Hotels',  'Travel',      '2026-06-23','debit', 'Conference hotel',             false),
    ('jpmorgan','jpmorgan_user_001', -1200.00,  'Legal Services',   'Professional','2026-06-22','debit', 'Legal consultation',           false)
ON CONFLICT DO NOTHING;

-- Demo user (for testing without a real company email)
INSERT INTO transactions
    (tenant_id, customer_id, amount, merchant, category, date, type, description, flagged)
VALUES
    ('demo','demo_user_001', -50.00,   'Test Merchant', 'Test', '2026-06-29','debit', 'Demo debit transaction',   false),
    ('demo','demo_user_001',  1000.00, 'Demo Income',   'Test', '2026-06-28','credit','Demo credit transaction',  false),
    ('demo','demo_user_001', -25.00,   'Demo Store',    'Test', '2026-06-27','debit', 'Another demo transaction', false)
ON CONFLICT DO NOTHING;
"""


# ── Public functions ──────────────────────────────────────────────────────────


def setup_phase3_schema() -> None:
    """Create Phase 3 tables and seed data. Safe to call multiple times."""
    url = os.getenv("DATABASE_URL")
    if not url:
        print("ERROR: DATABASE_URL not set in .env", file=sys.stderr)
        sys.exit(1)

    conn = psycopg2.connect(url)
    try:
        with conn.cursor() as cur:
            cur.execute(_SCHEMA)
            cur.execute(_SEED)
        conn.commit()
        print("Phase 3 schema ready: tenant_registry + transactions tables created and seeded.")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def provision_service_key(tenant_id: str) -> str:
    """
    Generate a new server-to-server secret for a tenant's backend to call
    POST /internal/mint-customer-token on behalf of its already-authenticated
    customers. Only the SHA-256 hash is stored — the raw key is returned
    exactly once and must be handed to the tenant out-of-band (never logged,
    never stored in our own DB in plaintext).

    Re-running this immediately invalidates the tenant's previous key.
    """
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set")

    raw_key  = secrets.token_hex(32)
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    conn = psycopg2.connect(url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE tenant_registry SET service_key_hash = %s WHERE tenant_id = %s",
                (key_hash, tenant_id),
            )
            if cur.rowcount == 0:
                raise ValueError(f"No tenant_registry row for tenant_id={tenant_id!r}")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return raw_key


def get_tenant_id(email: str) -> str | None:
    """
    Look up tenant_id for a given email address by matching the domain.
    Returns None if the domain is not registered.

    Example:
        get_tenant_id("analyst@goldmansachs.com") → "goldman"
        get_tenant_id("user@unknown.com")          → None
    """
    url = os.getenv("DATABASE_URL")
    if not url:
        return None

    domain = email.split("@")[-1].lower().strip() if "@" in email else email.lower().strip()
    if not domain:
        return None

    try:
        conn = psycopg2.connect(url)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT tenant_id FROM tenant_registry WHERE email_domain = %s",
                    (domain,),
                )
                row = cur.fetchone()
            return row[0] if row else None
        finally:
            conn.close()
    except Exception:
        return None


if __name__ == "__main__":
    # python -m db.schema --provision-key goldman
    if len(sys.argv) == 3 and sys.argv[1] == "--provision-key":
        key = provision_service_key(sys.argv[2])
        print(f"Service key for tenant '{sys.argv[2]}' — save this now, it will not be shown again:")
        print(key)
    else:
        setup_phase3_schema()
