"""
agents/logger.py — Centralised logging for all RAG agents.

Phase A (laptop): logs go to logs/rag.log + terminal.
Phase B (AWS ECS): set LOG_OUTPUT=cloudwatch in environment.
  ECS automatically ships stdout/stderr to CloudWatch Logs when
  awslogs log driver is configured in the task definition.
  No code change needed — just set LOG_OUTPUT=cloudwatch and the
  handler switches from file to stdout, which ECS captures.

Usage (every agent file):
    from agents.logger import get_logger
    logger = get_logger(__name__)

Log file: logs/rag.log
  - Rotates at midnight daily
  - Keeps 30 days of history
  - Format: 2026-06-30 14:23:01 | ERROR | agents.retriever | message
"""

import logging
import logging.handlers
import os
from pathlib import Path

_INITIALIZED = False

# Phase B switch: set LOG_OUTPUT=cloudwatch in ECS task definition env vars.
# ECS + awslogs driver captures stdout automatically — no SDK needed.
_LOG_OUTPUT = os.getenv("LOG_OUTPUT", "file").lower()

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_DIR   = Path(__file__).parent.parent / "logs"
_LOG_FILE  = _LOG_DIR / "rag.log"
_FORMAT    = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATEFMT   = "%Y-%m-%d %H:%M:%S"


def _setup() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return

    level = getattr(logging, _LOG_LEVEL, logging.INFO)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    root = logging.getLogger()
    root.setLevel(level)

    # Always log to terminal
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if _LOG_OUTPUT == "file":
        # Phase A: write to logs/rag.log, rotate daily, keep 30 days
        _LOG_DIR.mkdir(exist_ok=True)
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=_LOG_FILE,
            when="midnight",
            backupCount=30,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Phase B (cloudwatch): LOG_OUTPUT=cloudwatch → stdout only.
    # ECS task definition must include:
    #   "logConfiguration": {
    #     "logDriver": "awslogs",
    #     "options": {
    #       "awslogs-group": "/rag-agents/production",
    #       "awslogs-region": "us-east-1",
    #       "awslogs-stream-prefix": "rag"
    #     }
    #   }
    # That's it. ECS ships stdout to CloudWatch automatically.

    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Call once per module at import time."""
    _setup()
    return logging.getLogger(name)
