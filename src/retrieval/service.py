"""Composable dense, BM25 and hybrid retrieval service."""
from __future__ import annotations

from dataclasses import dataclass

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.contracts import SearchResult
from src.retrieval.dense import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import NoopReranker, Reranker


@dataclass(frozen=True)
class RetrievalOptions:
    dense_limit: int = 50
    bm25_limit: int = 50
    fusion_limit: int = 50
    rank_constant: int = 60
    dense_weight: float = 1.0
    bm25_weight: float = 1.0

    def __post_init__(self) -> None:
        limits = (self.dense_limit, self.bm25_limit, self.fusion_limit)
        if any(limit <= 0 for limit in limits):
            raise ValueError("Retrieval limits must be greater than zero")
        if self.rank_constant < 0:
            raise ValueError("RRF rank_constant must not be negative")
        if self.dense_weight <= 0 or self.bm25_weight <= 0:
            raise ValueError("RRF weights must be greater than zero")


class RetrievalService:
    def __init__(
        self,
        *,
        dense: DenseRetriever | None = None,
        bm25: BM25Retriever | None = None,
        reranker: Reranker | None = None,
        options: RetrievalOptions | None = None,
    ) -> None:
        self.dense = dense or DenseRetriever()
        self.bm25 = bm25 or BM25Retriever()
        self.reranker = reranker or NoopReranker()
        self.options = options or RetrievalOptions()

    def search_dense(self, query: str, *, limit: int | None = None) -> list[SearchResult]:
        resolved = self.options.dense_limit if limit is None else limit
        return self.dense.search(query, limit=resolved)

    def search_bm25(self, query: str, *, limit: int | None = None) -> list[SearchResult]:
        resolved = self.options.bm25_limit if limit is None else limit
        return self.bm25.search(query, limit=resolved)

    def search(self, query: str, *, limit: int | None = None) -> list[SearchResult]:
        final_limit = self.options.fusion_limit if limit is None else limit
        if final_limit <= 0:
            raise ValueError("Retrieval limit must be greater than zero")
        fused = reciprocal_rank_fusion(
            self.search_dense(query),
            self.search_bm25(query),
            rank_constant=self.options.rank_constant,
            dense_weight=self.options.dense_weight,
            bm25_weight=self.options.bm25_weight,
            limit=max(self.options.fusion_limit, final_limit),
        )
        return self.reranker.rerank(query, fused, limit=final_limit)
