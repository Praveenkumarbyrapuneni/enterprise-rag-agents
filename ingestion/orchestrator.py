"""
Ingestion Orchestrator — ties the full pipeline together.

Entry point: submit_document(file_path) → hash gate + Celery task
Pipeline:    ingest_document task → parse → chunk → embed → upload
Failures:    transient errors retry 3x with backoff; permanent failures → DLQ table

Production patterns (from Instagram, GitGuardian, Wolt engineering):
  acks_late + reject_on_worker_lost → task survives worker death
  prefetch_multiplier=1             → no task hoarding on long-running jobs
  soft_time_limit / time_limit      → kill runaway tasks (e.g. stuck OCR)
  exponential backoff + jitter      → safe retry, no thundering herd
  DLQ in PostgreSQL                 → queryable, permanent, auditable failure record
  JSON serializer (not pickle)      → prevents code execution from malicious messages
"""

import hashlib
import logging
import os
import random
import threading
import traceback as tb_module
from pathlib import Path
from typing import Dict, List, Optional

import psycopg2
from celery import Celery
from celery.exceptions import SoftTimeLimitExceeded

# ponytail: lazy imports inside the task — the submit process must not load
# parser/embedder/qdrant (and their heavy deps) just to queue a task

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Celery app
# ---------------------------------------------------------------------------

_BROKER = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

celery_app = Celery("rag_ingestion", broker=_BROKER, backend=_BACKEND)

celery_app.conf.update(
    # Task survives worker crash — not ack'd until the function returns.
    task_acks_late=True,
    # Worker killed by SIGKILL/OOM → task goes back to queue immediately.
    task_reject_on_worker_lost=True,
    # Each worker holds exactly one task. Long tasks cannot starve others.
    worker_prefetch_multiplier=1,
    # JSON is safe. Pickle allows arbitrary code execution from queue messages.
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    # visibility_timeout must exceed hard time_limit. 1 hour > 660s.
    # ponytail: prevents phantom retries when acks_late=True with Redis broker
    broker_transport_options={"visibility_timeout": 3600},
    task_track_started=True,
    task_send_sent_event=True,
    task_routes={
        "ingestion.orchestrator.ingest_document": {"queue": "ingestion"},
    },
)

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def _get_db_conn() -> psycopg2.extensions.connection:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL environment variable is not set")
    # connect_timeout=5: fail fast if Postgres is down — don't hang the submit loop
    return psycopg2.connect(url, connect_timeout=5)


# --- DLQ table setup: run DDL once per worker process, not on every failure ---
_dlq_ready = False
_dlq_ready_lock = threading.Lock()


def _ensure_dlq_once(conn: psycopg2.extensions.connection) -> None:
    """Create failed_documents table on first call per worker. Thread-safe."""
    global _dlq_ready
    if not _dlq_ready:
        with _dlq_ready_lock:
            if not _dlq_ready:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS failed_documents (
                            id              SERIAL PRIMARY KEY,
                            file_path       TEXT NOT NULL,
                            file_hash       TEXT,
                            celery_task_id  TEXT,
                            error_message   TEXT,
                            error_traceback TEXT,
                            retry_count     INTEGER DEFAULT 0,
                            failed_at       TIMESTAMPTZ DEFAULT NOW()
                        )
                    """)
                conn.commit()
                _dlq_ready = True


def _record_dlq(
    file_path: str,
    file_hash: str,
    task_id: str,
    exc: Exception,
    traceback_str: str,
    retry_count: int,
) -> None:
    """
    Write a permanently failed document to the dead letter table.

    Never raises — if the DLQ write itself fails, we log it and move on.
    Losing the DLQ record is bad but better than crashing the error handler.
    Uses a single connection for both table creation and insert (no leaks).

    Args:
        file_path:     Path to the original document.
        file_hash:     SHA-256 of the file.
        task_id:       Celery task ID for cross-referencing with Flower/logs.
        exc:           The exception that killed the task.
        traceback_str: Full Python traceback string.
        retry_count:   How many retries were attempted before giving up.
    """
    conn = None
    try:
        conn = _get_db_conn()
        _ensure_dlq_once(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO failed_documents
                    (file_path, file_hash, celery_task_id, error_message,
                     error_traceback, retry_count)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (file_path, file_hash, task_id, str(exc), traceback_str, retry_count),
            )
        conn.commit()
        logger.error(
            f"[DLQ] Recorded: {Path(file_path).name} after {retry_count} retries "
            f"— {type(exc).__name__}: {exc}"
        )
    except Exception as db_exc:
        logger.error(f"[DLQ] Failed to record to failed_documents: {db_exc}")
    finally:
        if conn is not None:
            conn.close()


# ---------------------------------------------------------------------------
# File hashing
# ---------------------------------------------------------------------------


def _hash_file(file_path: str) -> str:
    """
    Compute SHA-256 of a file using 64KB streaming blocks.

    Streaming avoids loading the full file into memory — safe for 500MB PDFs.

    Args:
        file_path: Path to the file.
    Returns:
        64-character hex string (SHA-256 digest).
    """
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _is_already_complete(file_hash: str) -> bool:
    """
    Pre-flight check: has this file hash been fully processed before?

    This is a best-effort gate — if the DB call fails, we proceed anyway.
    The authoritative idempotency check lives inside upload_document().

    Args:
        file_hash: SHA-256 of the file.
    Returns:
        True if status=complete exists in document_hashes.
    """
    conn = None
    try:
        conn = _get_db_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status FROM document_hashes WHERE file_hash = %s",
                (file_hash,),
            )
            row = cur.fetchone()
        return row is not None and row[0] == "complete"
    except Exception as e:
        logger.warning(f"Hash pre-check failed ({e}) — proceeding with ingestion")
        return False
    finally:
        if conn is not None:
            conn.close()


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def submit_document(file_path: str) -> Optional[str]:
    """
    Hash-gate + queue a single document for ingestion.

    Does NOT run the pipeline — that happens inside a Celery worker.
    Returns immediately after queuing so callers can submit hundreds of
    documents without waiting for any of them to complete.

    Args:
        file_path: Absolute path to the document.
    Returns:
        Celery task ID string, or None if document was already complete (skipped).
    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError:        If file_path is a directory, not a file.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not Path(file_path).is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    file_hash = _hash_file(file_path)

    if _is_already_complete(file_hash):
        logger.info(
            f"[{Path(file_path).name}] Skipped — already complete (hash={file_hash[:8]}...)"
        )
        return None

    result = ingest_document.apply_async(
        args=[file_path, file_hash],
        queue="ingestion",
    )
    logger.info(
        f"[{Path(file_path).name}] Queued — task_id={result.id} hash={file_hash[:8]}..."
    )
    return result.id


def submit_batch(file_paths: List[str]) -> Dict[str, Optional[str]]:
    """
    Submit multiple documents. Returns mapping of file_path → task_id (or None if skipped).

    Args:
        file_paths: List of absolute file paths.
    Returns:
        Dict mapping each path to its task ID, or None if already complete.
    """
    return {fp: submit_document(fp) for fp in file_paths}


# ---------------------------------------------------------------------------
# Celery task: full pipeline
# ---------------------------------------------------------------------------

# Transient errors worth retrying. Network blips and timeouts usually resolve
# within seconds. ValueError and RuntimeError indicate bad data or logic errors
# — retrying the same bad input produces the same error, so we skip to DLQ.
_TRANSIENT = (ConnectionError, TimeoutError, OSError)


@celery_app.task(
    bind=True,
    name="ingestion.orchestrator.ingest_document",
    max_retries=3,
    acks_late=True,
    reject_on_worker_lost=True,
    soft_time_limit=600,  # 10 min: raises SoftTimeLimitExceeded → graceful exit
    time_limit=660,       # 11 min: hard kill after soft window expires
)
def ingest_document(self, file_path: str, file_hash: str) -> Dict:
    """
    Full ingestion pipeline: parse → chunk → embed → upload.

    Retries up to 3 times on transient errors (ConnectionError, TimeoutError, OSError)
    with exponential backoff + jitter. Non-transient errors skip retries entirely.
    Both paths record permanently failed tasks to the failed_documents DLQ table.

    Args:
        file_path: Absolute path to the document (passed through from submit_document).
        file_hash: SHA-256 computed by submit_document (avoids re-hashing inside task).
    Returns:
        {"status": "complete", "file_name": str, "chunks": int, "parents": int}
    """
    file_name = Path(file_path).name
    attempt = self.request.retries + 1
    logger.info(
        f"[{file_name}] INGEST START "
        f"task_id={self.request.id} attempt={attempt}/{self.max_retries + 1}"
    )

    try:
        from ingestion.chunker import chunk_document
        from ingestion.embedder import build_document_text, embed_chunks
        from ingestion.parser import parse_document
        from ingestion.qdrant_uploader import upload_document

        logger.info(f"[{file_name}] 1/4 Parsing")
        pages = parse_document(file_path)
        if not pages:
            raise ValueError("Parser returned 0 pages — file may be empty or unreadable")

        logger.info(f"[{file_name}] 2/4 Chunking")
        chunk_result = chunk_document(pages, file_name, file_hash)
        chunks = chunk_result["chunks"]
        parents = chunk_result.get("parents", [])
        if not chunks:
            raise ValueError("Chunker returned 0 chunks — document may contain only whitespace")

        logger.info(f"[{file_name}] 3/4 Embedding {len(chunks)} chunks")
        document_text = build_document_text(pages)
        embedded = embed_chunks(chunks, document_text, file_name)

        logger.info(f"[{file_name}] 4/4 Uploading to Qdrant")
        upload_result = upload_document(embedded, parents, file_name, file_hash)

        if upload_result.get("skipped"):
            logger.info(f"[{file_name}] INGEST SKIPPED — already complete (idempotent check inside uploader)")
        else:
            logger.info(
                f"[{file_name}] INGEST SUCCESS "
                f"chunks={upload_result['chunks_uploaded']} parents={upload_result['parents_stored']}"
            )

        return {
            "status": "complete",
            "file_name": file_name,
            "chunks": upload_result.get("chunks_uploaded", len(chunks)),
            "parents": upload_result.get("parents_stored", len(parents)),
        }

    except SoftTimeLimitExceeded:
        logger.error(f"[{file_name}] SOFT TIME LIMIT — task exceeded 10 minutes, routing to DLQ")
        _record_dlq(
            file_path, file_hash, self.request.id,
            SoftTimeLimitExceeded("10-minute soft limit exceeded"),
            "", self.request.retries,
        )
        raise

    except _TRANSIENT as exc:
        # Transient error — retry with exponential backoff + jitter.
        # Cap at 60s so the queue doesn't stall for minutes between retries.
        retry_in = min(2 ** self.request.retries + random.uniform(0, 1), 60)
        logger.warning(
            f"[{file_name}] Transient error on attempt {attempt}: "
            f"{type(exc).__name__}: {exc} — retrying in {retry_in:.1f}s"
        )
        try:
            raise self.retry(exc=exc, countdown=retry_in)
        except self.MaxRetriesExceededError:
            traceback_str = tb_module.format_exc()
            logger.error(f"[{file_name}] Max retries exhausted — routing to DLQ")
            _record_dlq(file_path, file_hash, self.request.id, exc, traceback_str, self.request.retries)
            raise

    except Exception as exc:
        # Non-transient error (bad data, logic bug, permanent failure).
        # Retrying the same input produces the same error — go straight to DLQ.
        traceback_str = tb_module.format_exc()
        logger.error(
            f"[{file_name}] Non-transient failure on attempt {attempt} — routing to DLQ. "
            f"{type(exc).__name__}: {exc}"
        )
        _record_dlq(file_path, file_hash, self.request.id, exc, traceback_str, self.request.retries)
        raise


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import tempfile

    print("Orchestrator self-check...")

    # Test 1: hash is deterministic for the same content
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        f.write(b"enterprise rag system test content")
        tmp1 = f.name

    h1 = _hash_file(tmp1)
    h2 = _hash_file(tmp1)
    assert h1 == h2, "Hash must be deterministic"
    assert len(h1) == 64, "SHA-256 must produce a 64-character hex string"
    print(f"  hash determinism: OK ({h1[:8]}...)")

    # Test 2: different content produces different hash
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        f.write(b"completely different content")
        tmp2 = f.name

    assert _hash_file(tmp1) != _hash_file(tmp2), "Different files must have different hashes"
    print("  hash uniqueness: OK")

    # Test 3: missing file raises FileNotFoundError before any DB access
    try:
        submit_document("/nonexistent/path/file.pdf")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        print("  missing file check: OK")

    # Test 4: retry delay formula stays within bounds
    for attempt in range(4):
        delay = min(2 ** attempt + random.uniform(0, 1), 60)
        assert 0 < delay <= 60, f"Retry delay out of bounds: {delay}"
    print("  retry delay formula: OK")

    os.unlink(tmp1)
    os.unlink(tmp2)
    print("Self-check passed.")
