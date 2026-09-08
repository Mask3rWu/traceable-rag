"""READ-ONLY offline RRF weight sweep over a real runtime-eval batch.

Runs the research agent's actual retrieval queries (grown from a batch's
backed-up ``runs/``) through the dense and BM25 retrievers ONCE each, then
sweeps ``dense_weight``/``bm25_weight`` and reports, per weight combination, how
much of each query's weak gold (the evidence chunks that query actually
surfaced in the run) stays within the production top-K fused results.

This is decision evidence only: it never mutates retrieval configuration,
indexes, or stored runs. It is not part of the batch report flow — run it on
demand when a weight decision is needed.

Usage::

    python scripts/analyze_retrieval_sweep.py --batch-dir eval/runtime/batches/b_20260817_0028
    python scripts/analyze_retrieval_sweep.py --batch-dir ... --max-queries 40 --output sweep.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.bm25 import BM25Retriever  # noqa: E402
from src.retrieval.contracts import SearchResult  # noqa: E402
from src.retrieval.dense import DenseRetriever  # noqa: E402
from src.retrieval.fusion import reciprocal_rank_fusion  # noqa: E402
from src.research.retrieval_quality import DEFAULT_DEPTH  # noqa: E402


def extract_queries_and_gold(runs_dir: Path) -> dict[str, set[str]]:
    """Map distinct real query -> weak-gold chunk_ids (evidence it surfaced).

    Gold for a query is the union of ``chunk_id`` across every evidence whose
    retrieval trace references that query in any run. Queries with no gold fall
    out naturally as empty sets and are excluded by the caller.
    """
    gold: dict[str, set[str]] = {}
    for path in sorted(runs_dir.glob("*.json")):
        try:
            run = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for evidence in run.get("evidence") or []:
            chunk_id = evidence.get("chunk_id")
            if not chunk_id:
                continue
            for trace in evidence.get("retrieval") or []:
                query = trace.get("query")
                if not query:
                    continue
                gold.setdefault(query, set()).add(chunk_id)
    return gold


def hit_at_depth(
    fused: Iterable[SearchResult],
    *,
    gold: set[str],
    depth: int,
) -> bool:
    """Whether any weak-gold chunk is present in the top-``depth`` fused results."""
    seen = 0
    for result in fused:
        if result.chunk_id in gold:
            return True
        seen += 1
        if seen >= depth:
            break
    return False


def parse_weights(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise SystemExit("--weights must be a comma list of positive numbers")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep RRF weights over a real batch's retrieval quality (read-only)"
    )
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--max-queries", type=int, help="Cap on distinct queries used")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="Production fused top-K")
    parser.add_argument("--candidate-limit", type=int, default=50, help="Per-route candidates to retrieve")
    parser.add_argument("--rank-constant", type=int, default=60)
    parser.add_argument("--weights", default="0.5,1.0,1.5,2.0")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path (text table always printed)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.batch_dir.is_dir():
        raise SystemExit(f"batch dir not found: {args.batch_dir}")
    runs_dir = args.batch_dir / "runs"
    if not runs_dir.is_dir():
        raise SystemExit(f"no runs/ dir under {args.batch_dir}")

    gold = extract_queries_and_gold(runs_dir)
    queries = [q for q, g in gold.items() if g]
    if args.max_queries:
        queries = queries[: args.max_queries]
    if not queries:
        raise SystemExit("no real queries with weak gold extracted")
    print(f"queries: {len(queries)} (of {sum(bool(g) for g in gold.values())} with gold) | "
          f"depth: {args.depth} | candidate-limit: {args.candidate_limit} | "
          f"gold granularity: chunk")

    # Single retrieval pass per route — the weight grid below reuses these.
    dense = DenseRetriever()
    bm25 = BM25Retriever()
    try:
        dense_results = dense.search_many(queries, limit=args.candidate_limit)
        bm25_results = bm25.search_many(queries, limit=args.candidate_limit)
    except Exception as exc:  # pragma: no cover - env-dependent
        raise SystemExit(
            f"retrieval failed (is pgvector/embedding reachable?): {exc}"
        ) from exc

    weights = parse_weights(args.weights)
    combos = [
        (d, b) for d in weights for b in weights
    ]  # include baseline (1.0, 1.0); RRF requires both > 0
    rows: list[dict] = []
    for d_weight, b_weight in combos:
        hits = 0
        for query, dense_items, bm25_items in zip(queries, dense_results, bm25_results, strict=True):
            fused = reciprocal_rank_fusion(
                dense_items,
                bm25_items,
                rank_constant=args.rank_constant,
                dense_weight=d_weight,
                bm25_weight=b_weight,
                limit=args.depth,
            )
            if hit_at_depth(fused, gold=gold[query], depth=args.depth):
                hits += 1
        rate = hits / len(queries)
        rows.append(
            {
                "dense_weight": d_weight,
                "bm25_weight": b_weight,
                "hit_rate": round(rate, 4),
                "hits": hits,
                "queries": len(queries),
            }
        )
    rows.sort(key=lambda row: (-row["hit_rate"], row["dense_weight"], row["bm25_weight"]))

    print(f"\nweight sweep (weak-gold hit@{args.depth}); sorted by hit_rate desc (read-only, no config change)")
    print(f"{'dense_w':>8} {'bm25_w':>8}  {'hit_rate':>8}  {'hits/q':>8}")
    for row in rows:
        print(f"{row['dense_weight']:>8} {row['bm25_weight']:>8}  {row['hit_rate']:>8.4f}  "
              f"{row['hits']}/{row['queries']}")
    baseline = next((r for r in rows if r["dense_weight"] == 1.0 and r["bm25_weight"] == 1.0), None)
    if baseline:
        better = [r for r in rows if r["hit_rate"] > baseline["hit_rate"]]
        print(f"\nbaseline (1.0/1.0) hit_rate = {baseline['hit_rate']:.4f}; "
              f"{len(better)} / {len(rows)} combinations beat it")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "batch_dir": str(args.batch_dir),
            "depth": args.depth,
            "candidate_limit": args.candidate_limit,
            "rank_constant": args.rank_constant,
            "queries": len(queries),
            "gold_granularity": "chunk",
            "note": "decision evidence only; does not modify retrieval configuration",
            "rows": rows,
        }
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"report written to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())