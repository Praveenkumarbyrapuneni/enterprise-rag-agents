#!/usr/bin/env python3
"""
Local end-to-end query test — no LangSmith, no cloud.
Tests all three data paths: sql, hybrid, and rag.

Usage:
    python test_query.py                                    # run all test queries
    python test_query.py "your question" apple apple_user_001  # custom question
"""

import sys
from dotenv import load_dotenv

load_dotenv()

from agents.graph import run_query

SEP = "=" * 60

# ── Test queries covering all three data paths ────────────────────────────────

TEST_QUERIES = [
    # SQL path — live transaction data only
    {
        "question":    "What is my current balance?",
        "tenant_id":   "apple",
        "customer_id": "apple_user_001",
        "expect_source": "sql",
        "label": "SQL: balance lookup",
    },
    {
        "question":    "Show me my last 5 transactions",
        "tenant_id":   "goldman",
        "customer_id": "goldman_user_001",
        "expect_source": "sql",
        "label": "SQL: recent transactions",
    },
    {
        "question":    "Do I have any flagged or suspicious transactions?",
        "tenant_id":   "apple",
        "customer_id": "apple_user_001",
        "expect_source": "sql",
        "label": "SQL: flagged transactions",
    },
    # RAG path — document search only
    {
        "question":    "What was Apple's iPhone revenue in Q3 2024?",
        "tenant_id":   "apple",
        "customer_id": "apple_user_001",
        "expect_source": "rag",
        "label": "RAG: numerical from filing",
    },
    {
        "question":    "How did Goldman Sachs manage risk in 2025?",
        "tenant_id":   "goldman",
        "customer_id": "goldman_user_001",
        "expect_source": "rag",
        "label": "RAG: conceptual + HyDE",
    },
    # Hybrid path — SQL + document combined
    {
        "question":    "Why was I charged a foreign transaction fee on June 26?",
        "tenant_id":   "apple",
        "customer_id": "apple_user_001",
        "expect_source": "hybrid",
        "label": "HYBRID: transaction + fee policy",
    },
]


def run(question: str, tenant_id: str, customer_id: str, label: str = "", expect_source: str = "") -> None:
    print(f"\n{SEP}")
    if label:
        print(f"[{label}]")
    print(f"QUESTION   : {question}")
    print(f"TENANT     : {tenant_id}  |  CUSTOMER: {customer_id}")
    print(SEP)

    result = run_query(question, tenant_id=tenant_id, customer_id=customer_id)

    # Check data_source routing
    actual_source = result.get("data_source", "rag")
    routing_ok    = (not expect_source) or (actual_source == expect_source)
    routing_tag   = "✅" if routing_ok else f"⚠️  (expected {expect_source})"

    print(f"  query_type   : {result['query_type']}")
    print(f"  data_source  : {actual_source} {routing_tag}")
    print(f"  faithfulness : {result['faithfulness']:.2f}")
    print(f"  relevance    : {result['relevance']:.2f}")
    print(f"  retries      : {result['retry_count']}")

    if result.get("error"):
        print(f"  error        : {result['error']}")

    print(f"\nANSWER:\n{result['answer']}")

    if result["sources"]:
        print(f"\nSOURCES:")
        for s in result["sources"]:
            print(f"  - {s}")

    print()


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        # Custom question: python test_query.py "question" [tenant_id] [customer_id]
        question    = sys.argv[1]
        tenant_id   = sys.argv[2] if len(sys.argv) > 2 else "demo"
        customer_id = sys.argv[3] if len(sys.argv) > 3 else "demo_user_001"
        run(question, tenant_id, customer_id)
    else:
        for q in TEST_QUERIES:
            run(
                question      = q["question"],
                tenant_id     = q["tenant_id"],
                customer_id   = q["customer_id"],
                label         = q["label"],
                expect_source = q["expect_source"],
            )
