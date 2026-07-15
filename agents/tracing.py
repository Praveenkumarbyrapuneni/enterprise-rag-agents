"""
agents/tracing.py — OpenTelemetry tracing via self-hosted Arize Phoenix.

Phoenix runs locally in Docker (docker-compose.yml, service "phoenix") — a single
container, SQLite-backed, zero data leaves the machine. Same rule as CloudWatch
logging (see NEXT_VERSION.md #8): no SaaS tracing (LangSmith etc.) for a financial
institution client.

register(auto_instrument=True) activates whichever OpenInference instrumentors are
installed — langchain (covers LangGraph, since LangGraph runs on the same tracer)
and bedrock are both listed in requirements.txt, so both activate automatically.
Safe to call with no Phoenix container running: spans just queue and drop silently,
nothing in the request path depends on the collector being reachable.

Usage:
    from .tracing import get_tracer
    tracer = get_tracer(__name__)

    with tracer.start_as_current_span("parse_document") as span:
        span.set_attribute("file_name", file_name)
        ...
"""

import os
import threading

from phoenix.otel import register

_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:4317")
_PROJECT  = os.getenv("PHOENIX_PROJECT_NAME", "enterprise-rag")

_provider = None
_lock = threading.Lock()


def _setup():
    global _provider
    if _provider is None:
        with _lock:
            if _provider is None:
                _provider = register(
                    project_name=_PROJECT,
                    endpoint=_ENDPOINT,
                    auto_instrument=True,
                    batch=True,
                )
    return _provider


def get_tracer(name: str):
    """Return a named OTel tracer. Call once per module at import time."""
    return _setup().get_tracer(name)


# ── Self-check ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # No Phoenix container required — register() must not raise even if the
    # collector at _ENDPOINT is unreachable; spans just fail to export silently.
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("self_check_span") as span:
        span.set_attribute("check", "tracing.py")
    print(f"tracing.py self-check passed — tracer created, span emitted, endpoint={_ENDPOINT}")
