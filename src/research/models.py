"""Persisted contracts for traceable research runs."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


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


class Citation(BaseModel):
    """A source anchor pointing at an evidence excerpt."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str


class Claim(BaseModel):
    model_config = ConfigDict(frozen=True)

    claim_id: str
    text: str = Field(min_length=1)
    conclusion_type: Literal["direct", "synthesized", "normative", "hypothesis"]
    citations: list[Citation] = Field(default_factory=list)


class Conflict(BaseModel):
    model_config = ConfigDict(frozen=True)

    conflict_id: str
    description: str = Field(min_length=1)
    claim_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    status: Literal["open", "resolved"] = "open"
    resolution: str | None = None


class ToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    call_id: str
    tool: str
    arguments: dict
    result: dict
    created_at: datetime = Field(default_factory=utc_now)


class ResearchRun(BaseModel):
    """Complete, resumable record of a single research question."""

    run_id: str = Field(default_factory=lambda: uuid4().hex)
    question: str = Field(min_length=1)
    status: Literal[
        "created",
        "planning",
        "retrieving",
        "synthesizing",
        "verifying",
        "completed",
        "failed",
    ] = "created"
    queries: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    summary: str = ""
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def touch(self) -> None:
        self.updated_at = utc_now()
