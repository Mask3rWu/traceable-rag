"""Reranker contract; model-backed implementations are intentionally deferred."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from src.retrieval.contracts import SearchResult


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: Sequence[SearchResult], *, limit: int
    ) -> list[SearchResult]: ...


class NoopReranker:
    """Preserve fused order while satisfying the future reranker interface."""

    def rerank(
        self, query: str, candidates: Sequence[SearchResult], *, limit: int
    ) -> list[SearchResult]:
        del query
        if limit <= 0:
            raise ValueError("Reranker limit must be greater than zero")
        return [
            item.model_copy(update={"final_rank": rank})
            for rank, item in enumerate(candidates[:limit], start=1)
        ]
