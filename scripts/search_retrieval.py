"""Inspect dense, BM25 or RRF results for one query."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.catalog import ChunkCatalog  # noqa: E402
from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.retrieval.dense import DenseRetriever  # noqa: E402
from src.retrieval.service import RetrievalService  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect retrieval results")
    parser.add_argument("query")
    parser.add_argument("--method", choices=("dense", "bm25", "rrf"), default="rrf")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--text-chars", type=int, default=300)
    args = parser.parse_args()

    if args.limit <= 0 or args.text_chars <= 0:
        raise SystemExit("--limit and --text-chars must be greater than zero")
    if args.method == "dense":
        results = DenseRetriever().search(args.query, limit=args.limit)
    elif args.method == "bm25":
        results = BM25Retriever().search(args.query, limit=args.limit)
    else:
        results = RetrievalService().search(args.query, limit=args.limit)

    catalog = ChunkCatalog.load()
    for result in results:
        catalog.validate_result(result)
        chunk = catalog.source(result.chunk_id).chunk
        score = result.fusion_score
        if args.method == "dense":
            score = result.dense_score
        elif args.method == "bm25":
            score = result.bm25_score
        print(
            f"\n[{result.final_rank}] {result.chunk_id} score={score:.6f} "
            f"pages={chunk.page_start}-{chunk.page_end}"
        )
        print(" > ".join(chunk.heading_path) or "(no heading)")
        print(chunk.text[: args.text_chars].replace("\n", " "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
