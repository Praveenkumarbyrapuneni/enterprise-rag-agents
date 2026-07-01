"""
db/schema.py — Phase 3 PostgreSQL schema: tenant registry + mock transactions.

Run once before starting the API:
    python -m db.schema

Safe to re-run: all creates use IF NOT EXISTS, inserts use ON CONFLICT DO NOTHING.
"""

import os
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
    setup_phase3_schema()
