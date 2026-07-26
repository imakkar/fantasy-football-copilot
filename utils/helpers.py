"""
Shared utility functions used across the pipeline.

These are small, reusable helpers with no project-specific state, kept separate from
the pipeline logic in `src/` to maintain a clean separation of concerns:

  * format_context    - render retrieved passages into a numbered context block
  * sanitize_metadata - coerce metadata into ChromaDB-compatible primitive types
  * Timer             - context manager for measuring elapsed wall-clock time
  * get_logger        - a minimally configured stdlib logger
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List


def format_context(hits: List[Dict]) -> str:
    """Render a list of retrieved passages into a numbered context block."""
    return "\n".join(f"[{i}] {hit['text']}" for i, hit in enumerate(hits, 1))


def sanitize_metadata(md: Dict) -> Dict:
    """Coerce metadata values into types ChromaDB accepts (str/int/float/bool)."""
    clean = {}
    for key, value in md.items():
        clean[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
    return clean


class Timer:
    """Context manager that records elapsed wall-clock seconds.

    Usage:
        with Timer() as t:
            do_work()
        print(t.seconds)
    """

    def __enter__(self) -> "Timer":
        self._start = time.time()
        self.seconds = 0.0
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = round(time.time() - self._start, 2)


def get_logger(name: str = "copilot") -> logging.Logger:
    """Return a simple, consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
