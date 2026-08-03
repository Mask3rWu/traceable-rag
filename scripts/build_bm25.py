"""Command-line wrapper for the persistent BM25 indexer."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25 import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
