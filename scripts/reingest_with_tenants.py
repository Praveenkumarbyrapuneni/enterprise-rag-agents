"""
scripts/reingest_with_tenants.py — Delete + re-ingest all SEC filings with tenant_id.

Run ONCE after Phase 3 setup:
    python -m scripts.reingest_with_tenants

What this does:
  1. Deletes the existing Qdrant collection (chunks have no tenant_id)
  2. Resets ingestion_status for these files (forces re-ingest)
  3. Re-creates Qdrant collection with tenant_id payload index
  4. Re-ingests all 3 SEC filings directly (no Celery needed) with correct tenant_id

After this script, every chunk in Qdrant has tenant_id in its payload.
Goldman analyst queries will ONLY return Goldman chunks. Same for Apple, JPMorgan.

WARNING: Do not run while Celery workers are actively ingesting — stop them first.
"""

import hashlib
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# ── Tenant map: filename → tenant_id ─────────────────────────────────────────

TENANT_MAP = {
    "Apple_2025-10-31.htm":         "apple",
    "Goldman_Sachs_2026-02-25.htm": "goldman",
    "JPMorgan_Chase_2026-02-13.htm":"jpmorgan",
}

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TESTS_DIR    = os.path.join(_PROJECT_ROOT, "tests")


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _reset_ingestion_status(file_names: list[str]) -> int:
    """Delete ingestion_status rows for these files so they re-ingest cleanly."""
    import psycopg2
    url = os.getenv("DATABASE_URL")
    if not url:
        print("WARNING: DATABASE_URL not set — skipping ingestion_status reset")
        return 0
    conn = psycopg2.connect(url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM ingestion_status WHERE file_name = ANY(%s)",
                (file_names,),
            )
            deleted = cur.rowcount
        conn.commit()
        return deleted
    finally:
        conn.close()


def _delete_qdrant_collection(collection_name: str) -> bool:
    """Delete the Qdrant collection. Returns True if it existed."""
    from qdrant_client import QdrantClient
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
        timeout=30.0,
    )
    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        client.delete_collection(collection_name)
        return True
    return False


def reingest() -> None:
    # Import pipeline components
    from ingestion.parser import parse_document
    from ingestion.chunker import chunk_document
    from ingestion.embedder import embed_chunks, build_document_text
    from ingestion.qdrant_uploader import (
        COLLECTION_NAME,
        bootstrap_schema,
        upload_document,
        _ensure_collection,  # we reset _collection_ready so it re-creates
    )
    import ingestion.qdrant_uploader as uploader_mod

    # ── Step 1: Delete existing Qdrant collection ─────────────────────────────
    print(f"\nStep 1: Deleting Qdrant collection '{COLLECTION_NAME}'...")
    deleted = _delete_qdrant_collection(COLLECTION_NAME)
    if deleted:
        print(f"  Deleted '{COLLECTION_NAME}'.")
        uploader_mod._collection_ready = False  # force re-create
    else:
        print(f"  Collection '{COLLECTION_NAME}' did not exist — nothing to delete.")

    # ── Step 2: Reset ingestion_status ───────────────────────────────────────
    print("\nStep 2: Resetting ingestion_status for target files...")
    n = _reset_ingestion_status(list(TENANT_MAP.keys()))
    print(f"  Cleared {n} ingestion_status record(s).")

    # ── Step 3: Bootstrap schema + create new collection ─────────────────────
    print("\nStep 3: Creating new Qdrant collection with tenant_id payload index...")
    bootstrap_schema()
    _ensure_collection()
    print(f"  Collection '{COLLECTION_NAME}' created with tenant_id index.")

    # ── Step 4: Re-ingest each file ───────────────────────────────────────────
    print("\nStep 4: Re-ingesting files with tenant_id...\n")

    for file_name, tenant_id in TENANT_MAP.items():
        path = os.path.join(_TESTS_DIR, file_name)
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found — skipping {file_name}")
            continue

        print(f"  [{tenant_id.upper()}] {file_name}")

        try:
            file_hash = _hash_file(path)

            # Parse
            pages = parse_document(path)
            print(f"    Parsed: {len(pages)} pages, {sum(len(p['text']) for p in pages):,} chars")

            # Chunk
            chunk_result = chunk_document(pages, file_name, file_hash)
            chunks  = chunk_result["chunks"]
            parents = chunk_result["parents"]
            strategy = chunks[0]["metadata"].get("strategy", "?") if chunks else "?"
            print(f"    Chunked: {len(chunks)} chunks, {len(parents)} parents (strategy={strategy})")

            # Inject tenant_id into every chunk and parent metadata
            for chunk in chunks:
                chunk.setdefault("metadata", {})["tenant_id"] = tenant_id
            for parent in parents:
                parent["tenant_id"] = tenant_id

            # Embed
            document_text = build_document_text(pages)
            embedded = embed_chunks(chunks, document_text, file_name)
            print(f"    Embedded: {len(embedded)} vectors ({len(embedded[0]['vector'])} dims each)")

            # Upload — pass tenant_id so it goes into Qdrant payload + shard key
            result = upload_document(
                embedded, parents, file_name, file_hash, tenant_id=tenant_id
            )

            if result["skipped"]:
                print(f"    SKIPPED (ingestion_status shows complete — was reset not applied?)")
            else:
                print(f"    ✅ Uploaded: {result['chunks_uploaded']} chunks → Qdrant")
                print(f"       Parents: {result['parents_stored']} → PostgreSQL")

        except Exception as e:
            print(f"    ❌ FAILED: {type(e).__name__}: {e}")
            print("       Continuing with next file...")

        print()

    # ── Step 5: Verify ────────────────────────────────────────────────────────
    print("Step 5: Verifying tenant isolation in Qdrant...")
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
        )
        for tenant_id in TENANT_MAP.values():
            result = client.count(
                collection_name=COLLECTION_NAME,
                count_filter=Filter(
                    must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
                ),
                exact=True,
            )
            print(f"  tenant_id={tenant_id}: {result.count} chunks")

        total = client.count(collection_name=COLLECTION_NAME, exact=True)
        print(f"  TOTAL: {total.count} chunks across all tenants")

    except Exception as e:
        print(f"  Verification failed: {e}")

    print("\n✅ Re-ingestion complete. Run python test_query.py to verify end-to-end.")


if __name__ == "__main__":
    reingest()
