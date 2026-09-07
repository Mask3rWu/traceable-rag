"""Public API contracts."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from src.research.agent_models import AgentRun


RunStatus = Literal[
    "queued",
    "running",
    "cancel_requested",
    "cancelled",
    "completed",
    "incomplete",
    "failed",
    "routed_away",
]


class DegradedStatus(BaseModel):
    """Degradation of a completed run, as a modifier not a peer terminal state.

    ``layers`` lists which component degraded (retrieval / model / task). It
    may only accompany ``status=completed``.
    """

    layers: list[Literal["retrieval", "model", "task"]] = Field(default_factory=list)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RunCreate(BaseModel):
    request: str = Field(min_length=1, max_length=20_000)
    expected_route: Literal["fast", "supervisor"] | None = Field(
        default=None,
        description=(
            "Route policy for the run. If 'fast', the run is interrupted "
            "(status=routed_away) should the router decide to run it as a "
            "multi-agent (supervisor) task."
        ),
    )


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
    degraded: DegradedStatus | None = None
    wasted_tokens: int = 0
    spurious_tool_calls: int = 0

    @model_validator(mode="after")
    def degraded_implies_completed(self) -> "RunSummary":
        if self.degraded is not None and self.status != "completed":
            raise ValueError("degraded may only accompany status=completed")
        return self


class RunDetail(RunSummary):
    result: AgentRun | None = None
    metrics: dict | None = Field(
        default=None,
        description="Raw RuntimeMetrics snapshot for the run, if the API collected it.",
    )


class RunList(BaseModel):
    items: list[RunSummary]
