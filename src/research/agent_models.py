"""Persisted contracts for routed ReAct agent runs."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from src.research.models import Claim, Conflict, Evidence


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RouteDecision(BaseModel):
    mode: Literal["fast", "supervisor"]
    reason: str = Field(min_length=1)


class ResearchPacket(BaseModel):
    task: str = Field(min_length=1)
    status: Literal["sufficient", "insufficient"]
    summary: str = Field(min_length=1)
    claims: list[Claim] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class AgentAnswer(BaseModel):
    content: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class AgentRun(BaseModel):
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    request: str = Field(min_length=1)
    route: RouteDecision
    answer: AgentAnswer
    evidence: list[Evidence] = Field(default_factory=list)
    worker_packets: list[ResearchPacket] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
