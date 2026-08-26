"""Persisted contracts shared by the research agent."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RetrievalTrace(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    final_rank: int
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    fusion_score: float | None = None


class EvidenceVisual(BaseModel):
    model_config = ConfigDict(frozen=True)

    block_id: str
    block_type: str
    page: int
    relation: str
    image_crop: str | None = None
    description: str | None = None
    status: str


class Evidence(BaseModel):
    """A source excerpt whose provenance can be checked without a UI."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str
    chunk_id: str
    content_hash: str
    document_id: str
    source_file: str
    page_start: int
    page_end: int
    section_path: list[str] = Field(default_factory=list)
    block_ids: list[str]
    quote: str
    quote_truncated: bool = False
    visual_assets: list[EvidenceVisual] = Field(default_factory=list)
    retrieval: list[RetrievalTrace] = Field(default_factory=list)


class Conflict(BaseModel):
    model_config = ConfigDict(frozen=True)

    conflict_id: str
    description: str = Field(min_length=1)
    claim_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    status: Literal["open", "resolved"] = "open"
    resolution: str | None = None


