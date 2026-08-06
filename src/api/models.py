"""Public API contracts."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.research.agent_models import AgentRun


RunStatus = Literal[
    "queued",
    "running",
    "cancel_requested",
    "cancelled",
    "completed",
    "incomplete",
    "failed",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RunCreate(BaseModel):
    request: str = Field(min_length=1, max_length=20_000)


class RunResume(BaseModel):
    start_chapter: str | None = Field(default=None, min_length=1, max_length=120)


class RunEvent(BaseModel):
    sequence: int
    type: str
    created_at: datetime = Field(default_factory=utc_now)
    data: dict[str, Any] = Field(default_factory=dict)


class RunSummary(BaseModel):
    run_id: str
    request: str
    status: RunStatus
    route: Literal["fast", "supervisor"] | None = None
    route_reason: str | None = None
    trace_id: str | None = None
    evidence_count: int = 0
    worker_count: int = 0
    created_at: datetime
    updated_at: datetime
    error: str | None = None


class RunDetail(RunSummary):
    result: AgentRun | None = None


class RunList(BaseModel):
    items: list[RunSummary]
