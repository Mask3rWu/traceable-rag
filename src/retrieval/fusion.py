"""Rank-based fusion for dense and BM25 candidates."""
from __future__ import annotations

from collections.abc import Sequence

from src.retrieval.contracts import SearchResult


def reciprocal_rank_fusion(
    dense_results: Sequence[SearchResult],
    bm25_results: Sequence[SearchResult],
    *,
    rank_constant: int = 60,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
    limit: int | None = None,
) -> list[SearchResult]:
    if rank_constant < 0:
        raise ValueError("RRF rank_constant must not be negative")
    if dense_weight <= 0 or bm25_weight <= 0:
        raise ValueError("RRF weights must be greater than zero")
    if limit is not None and limit <= 0:
        raise ValueError("RRF limit must be greater than zero")

    combined: dict[str, SearchResult] = {}
    fusion_scores: dict[str, float] = {}
    for channel, results, weight in (
        ("dense", dense_results, dense_weight),
        ("bm25", bm25_results, bm25_weight),
    ):
        for position, result in enumerate(results, start=1):
            rank = result.dense_rank if channel == "dense" else result.bm25_rank
            rank = rank or position
            existing = combined.get(result.chunk_id)
            if existing is not None and existing.content_hash != result.content_hash:
                raise RuntimeError(
                    f"Index version mismatch for chunk {result.chunk_id}: "
                    f"{existing.content_hash} != {result.content_hash}"
                )
            updates = {
                "dense_rank": result.dense_rank,
                "dense_score": result.dense_score,
            } if channel == "dense" else {
                "bm25_rank": result.bm25_rank,
                "bm25_score": result.bm25_score,
            }
            base = existing or result
            combined[result.chunk_id] = base.model_copy(update=updates)
            fusion_scores[result.chunk_id] = fusion_scores.get(result.chunk_id, 0.0) + (
                weight / (rank_constant + rank)
            )

    ordered = sorted(
        combined.values(),
        key=lambda item: (-fusion_scores[item.chunk_id], item.chunk_id),
    )
    if limit is not None:
        ordered = ordered[:limit]
    return [
        item.model_copy(
            update={
                "fusion_score": fusion_scores[item.chunk_id],
                "final_rank": rank,
            }
        )
        for rank, item in enumerate(ordered, start=1)
    ]
