"""Retrieval evaluation against block-level evidence annotations."""
from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from src.paths import PROJECT_ROOT
from src.retrieval.bm25 import BM25Retriever, DEFAULT_INDEX_ROOT
from src.retrieval.catalog import ChunkCatalog
from src.retrieval.contracts import SearchResult
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion


DEFAULT_EVAL_ROOT = PROJECT_ROOT / "eval" / "v1"


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    document_id: str
    question_id: str
    question: str
    evidence_block_ids: frozenset[str]
    source_granularity: str
    answerability: str = "answerable"


@dataclass(frozen=True)
class MetricSummary:
    k: int
    evidence_recall: float
    hit_rate: float
    complete_recall: float
    mrr: float


def load_evaluation_cases(
    root: Path = DEFAULT_EVAL_ROOT, *, documents: set[str] | None = None
) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    for path in sorted(root.glob("*.jsonl"), key=lambda item: item.name):
        if documents and path.stem not in documents:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = row["question_id"]
            cases.append(
                EvaluationCase(
                    case_id=f"{path.stem}:{question_id}",
                    document_id=path.stem,
                    question_id=question_id,
                    question=row["question"],
                    evidence_block_ids=frozenset(row["evidence_block_ids"]),
                    source_granularity=row["source_granularity"],
                    answerability=row.get("answerability", "answerable"),
                )
            )
    return cases


def evaluate_rankings(
    cases: Sequence[EvaluationCase],
    rankings: Sequence[Sequence[SearchResult]],
    catalog: ChunkCatalog,
    *,
    ks: Sequence[int],
) -> list[MetricSummary]:
    if len(cases) != len(rankings):
        raise ValueError("Evaluation cases and rankings must have the same length")
    if not cases:
        raise ValueError("No evaluation cases selected")
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("Evaluation cutoffs must be positive")
    scored_pairs = [
        (case, results)
        for case, results in zip(cases, rankings, strict=True)
        if case.answerability == "answerable"
    ]
    if not scored_pairs:
        raise ValueError("No answerable evaluation cases selected")
    missing_gold = set().union(*(case.evidence_block_ids for case, _ in scored_pairs)) - (
        catalog.block_id_universe
    )
    if missing_gold:
        raise RuntimeError(
            f"Evaluation evidence is absent from current chunks: {sorted(missing_gold)[:5]}"
        )
    for _, results in scored_pairs:
        for result in results:
            catalog.validate_result(result)

    summaries: list[MetricSummary] = []
    for k in sorted(set(ks)):
        recalls: list[float] = []
        hits = 0
        complete = 0
        reciprocal_ranks: list[float] = []
        for case, results in scored_pairs:
            result_blocks = [catalog.block_ids(result.chunk_id) for result in results[:k]]
            retrieved = set().union(*result_blocks) if result_blocks else set()
            matched = case.evidence_block_ids & retrieved
            recalls.append(len(matched) / len(case.evidence_block_ids))
            hits += bool(matched)
            complete += case.evidence_block_ids <= retrieved
            first_relevant = next(
                (
                    rank
                    for rank, block_ids in enumerate(result_blocks, start=1)
                    if case.evidence_block_ids & block_ids
                ),
                None,
            )
            reciprocal_ranks.append(1 / first_relevant if first_relevant else 0.0)
        count = len(scored_pairs)
        summaries.append(
            MetricSummary(
                k=k,
                evidence_recall=sum(recalls) / count,
                hit_rate=hits / count,
                complete_recall=complete / count,
                mrr=sum(reciprocal_ranks) / count,
            )
        )
    return summaries


def _timed(callable_):
    started = time.perf_counter()
    result = callable_()
    return result, time.perf_counter() - started


def _print_metrics(
    reports: dict[str, list[MetricSummary]], timings: dict[str, float]
) -> None:
    print("method  k   evidence_recall  hit_rate  complete_recall  mrr")
    for method, summaries in reports.items():
        for summary in summaries:
            print(
                f"{method:<7} {summary.k:>3} "
                f"{summary.evidence_recall:>16.4f} "
                f"{summary.hit_rate:>9.4f} "
                f"{summary.complete_recall:>16.4f} "
                f"{summary.mrr:>6.4f}"
            )
        print(f"{method} wall time: {timings[method]:.2f}s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare dense, BM25 and RRF retrieval on an evaluation set"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("dense", "bm25", "rrf"),
        default=("dense", "bm25", "rrf"),
    )
    parser.add_argument("--k", nargs="+", type=int, default=(5, 10, 20, 50))
    parser.add_argument("--candidate-limit", type=int, default=50)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--document", action="append", dest="documents")
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--bm25-index", type=Path, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rank-constant", type=int, default=60)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    methods = list(dict.fromkeys(args.methods))
    if args.candidate_limit < max(args.k):
        raise SystemExit("--candidate-limit must be at least the largest --k value")
    cases = load_evaluation_cases(
        args.eval_root, documents=set(args.documents) if args.documents else None
    )
    if args.max_questions is not None:
        if args.max_questions <= 0:
            raise SystemExit("--max-questions must be greater than zero")
        if args.max_questions < len(cases):
            cases = random.Random(args.seed).sample(cases, args.max_questions)
            cases.sort(key=lambda item: item.case_id)
    if not cases:
        raise SystemExit("No evaluation questions selected")

    catalog = ChunkCatalog.load()
    queries = [case.question for case in cases]
    need_dense = "dense" in methods or "rrf" in methods
    need_bm25 = "bm25" in methods or "rrf" in methods
    rankings: dict[str, list[list[SearchResult]]] = {}
    timings: dict[str, float] = {}

    if need_dense:
        dense = DenseRetriever()
        dense_results, dense_time = _timed(
            lambda: dense.search_many(queries, limit=args.candidate_limit)
        )
        if "dense" in methods:
            rankings["dense"] = dense_results
            timings["dense"] = dense_time
    else:
        dense_results = []

    if need_bm25:
        bm25 = BM25Retriever(args.bm25_index)
        bm25_results, bm25_time = _timed(
            lambda: bm25.search_many(queries, limit=args.candidate_limit)
        )
        if "bm25" in methods:
            rankings["bm25"] = bm25_results
            timings["bm25"] = bm25_time
    else:
        bm25_results = []

    if "rrf" in methods:
        started = time.perf_counter()
        rrf_results = [
            reciprocal_rank_fusion(
                dense_items,
                bm25_items,
                rank_constant=args.rank_constant,
                dense_weight=args.dense_weight,
                bm25_weight=args.bm25_weight,
                limit=args.candidate_limit,
            )
            for dense_items, bm25_items in zip(
                dense_results, bm25_results, strict=True
            )
        ]
        rankings["rrf"] = rrf_results
        timings["rrf"] = dense_time + bm25_time + (time.perf_counter() - started)

    reports = {
        method: evaluate_rankings(cases, rankings[method], catalog, ks=args.k)
        for method in methods
    }
    answerable_count = sum(case.answerability == "answerable" for case in cases)
    print(
        f"Questions: {len(cases)} (answerable: {answerable_count}; "
        f"unanswerable: {len(cases) - answerable_count}); chunks: {len(catalog)}"
    )
    _print_metrics(reports, timings)
    if args.output:
        payload = {
            "questions": len(cases),
            "chunks": len(catalog),
            "candidate_limit": args.candidate_limit,
            "timings_seconds": timings,
            "metrics": {
                method: [asdict(summary) for summary in summaries]
                for method, summaries in reports.items()
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Report written to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
