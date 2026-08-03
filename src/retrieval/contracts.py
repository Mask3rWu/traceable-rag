"""Typed contracts shared by retrieval stages."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SearchResult(BaseModel):
    """A lightweight chunk reference with scores produced by retrieval stages."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    content_hash: str
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    final_rank: int | None = None
