#!/usr/bin/env python3
"""
End-to-end pipeline demo: file → parse → chunk → embed → upload → query.

Usage:
    python demo.py --file path/to/document.pdf --query "your question"

Requires:
    - Docker running: docker compose up -d
    - .env file with AWS Bedrock credentials, DATABASE_URL, and QDRANT settings
"""

import argparse
import hashlib
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def run(file_path: str, query_text: str, tenant_id: str = "demo") -> None:
    from ingestion.chunker import chunk_document
    from ingestion.embedder import build_document_text, embed_chunks
    from ingestion.parser import parse_document
    from ingestion.qdrant_uploader import (
        bootstrap_schema,
        query_similar,
        upload_document,
    )

    sep = "=" * 60

    # ── Step 1: Parse ─────────────────────────────────────────────
    print(f"\n{sep}\nSTEP 1 — Parse: {file_path}\n{sep}")
    pages = parse_document(file_path)
    total_chars = sum(len(p["text"]) for p in pages)
    print(f"  {len(pages)} pages, {total_chars:,} chars extracted")

    # ── Step 2: Chunk ─────────────────────────────────────────────
    file_name = os.path.basename(file_path)
    file_hash = _hash_file(file_path)
    print(f"\n{sep}\nSTEP 2 — Chunk  (file_hash={file_hash[:12]}...)\n{sep}")
    result = chunk_document(pages, file_name, file_hash)
    chunks  = result["chunks"]
    parents = result["parents"]
    strategy = chunks[0]["metadata"]["strategy"] if chunks else "unknown"
    print(
        f"  {len(chunks)} chunks, {len(parents)} parents  strategy={strategy}"
    )
    if chunks:
        print(f"  Preview chunk 0: {chunks[0]['text'][:200]}...")

    # ── Step 3: Embed ─────────────────────────────────────────────
    print(f"\n{sep}\nSTEP 3 — Embed  (Cohere Embed v3 via Bedrock, 1024 dims)\n{sep}")
    document_text = build_document_text(pages)
    embedded = embed_chunks(chunks, document_text, file_name)
    print(f"  {len(embedded)} vectors, {len(embedded[0]['vector'])} dims each")

    # ── Step 4: Upload ────────────────────────────────────────────
    print(f"\n{sep}\nSTEP 4 — Upload to Qdrant + PostgreSQL\n{sep}")
    bootstrap_schema()
    upload_result = upload_document(embedded, parents, file_name, file_hash, tenant_id=tenant_id)
    if upload_result["skipped"]:
        print("  Document already indexed — skipped (idempotent)")
    else:
        print(
            f"  {upload_result['chunks_uploaded']} child chunks → Qdrant\n"
            f"  {upload_result['parents_stored']} parent chunks → PostgreSQL"
        )

    # ── Step 5: Query ─────────────────────────────────────────────
    print(f"\n{sep}\nSTEP 5 — Query: '{query_text}'\n{sep}")
    from agents.retriever import _embed_query

    query_vector = _embed_query(query_text)
    results = query_similar(query_vector, top_k=3, file_name_filter=file_name)

    for i, r in enumerate(results, 1):
        context = r["parent_text"] or r["text"]
        print(f"\n  Result {i}  score={r['score']:.3f}  page={r['page']}  "
              f"strategy={r['strategy']}")
        print(f"  {context[:400]}...")

    print(f"\n{sep}\nDemo complete.\n{sep}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enterprise RAG pipeline demo")
    parser.add_argument("--file",  required=True, help="Path to document")
    parser.add_argument("--query", required=True, help="Question to answer")
    parser.add_argument("--tenant-id", default="demo", help="Tenant ID to attach to uploaded chunks")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: file not found: {args.file}")
        sys.exit(1)

    run(args.file, args.query, tenant_id=args.tenant_id)
