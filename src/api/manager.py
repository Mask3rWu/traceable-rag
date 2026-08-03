"""Threaded run manager used by the HTTP layer."""
from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable
from uuid import uuid4

from src.api.events import AgentEventCallback
from src.api.models import RunDetail, RunEvent, RunStatus, RunSummary, utc_now
from src.research.agent_models import AgentRun
from src.research.agent_store import AgentRunStore
from src.research.service import RoutedResearchAgent, build_research_agent


TERMINAL_STATUSES: set[RunStatus] = {
    "cancelled",
    "completed",
    "incomplete",
    "failed",
}


@dataclass
class ManagedRun:
    run_id: str
    request: str
    status: RunStatus = "queued"
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    trace_id: str | None = None
    result: AgentRun | None = None
    error: str | None = None
    events: list[RunEvent] = field(default_factory=list)
    future: Future | None = None
    condition: threading.Condition = field(default_factory=threading.Condition)


class RunManager:
    def __init__(
        self,
        *,
        store: AgentRunStore | None = None,
        agent_factory: Callable[[], RoutedResearchAgent] = build_research_agent,
        max_concurrent_runs: int = 2,
    ) -> None:
        if max_concurrent_runs <= 0:
            raise ValueError("max_concurrent_runs must be greater than zero")
        self.store = store or AgentRunStore()
        self.agent_factory = agent_factory
        self.executor = ThreadPoolExecutor(
            max_workers=max_concurrent_runs, thread_name_prefix="research-run"
        )
        self._runs: dict[str, ManagedRun] = {}
        self._lock = threading.RLock()

    def create(self, request: str) -> RunSummary:
        normalized = request.strip()
        if not normalized:
            raise ValueError("request must not be blank")
        managed = ManagedRun(run_id=uuid4().hex, request=normalized)
        with self._lock:
            self._runs[managed.run_id] = managed
        self._emit(managed, "queued", {"request": normalized[:240]})
        managed.future = self.executor.submit(self._execute, managed)
        return self._summary(managed)

    def _execute(self, managed: ManagedRun) -> None:
        with managed.condition:
            if managed.status == "cancelled":
                return
            managed.status = "running"
            managed.updated_at = utc_now()
        self._emit(managed, "running", {})
        try:
            agent = self.agent_factory()
            if agent.langfuse is not None:
                managed.trace_id = agent.langfuse.create_trace_id(seed=managed.run_id)
            callback = AgentEventCallback(
                lambda event_type, data: self._emit(managed, event_type, data)
            )
            result, _ = agent.run(
                managed.request,
                run_id=managed.run_id,
                trace_id=managed.trace_id,
                callbacks=[callback],
            )
            with managed.condition:
                managed.result = result
                managed.trace_id = result.trace_id
                managed.status = result.outcome
                managed.updated_at = utc_now()
            self._emit(
                managed,
                result.outcome,
                {
                    "route": result.route.mode,
                    "evidence_count": len(result.evidence),
                    "worker_count": len(result.worker_packets),
                    "trace_id": result.trace_id,
                },
            )
        except Exception as exc:
            with managed.condition:
                managed.status = "failed"
                managed.error = f"{type(exc).__name__}: {exc}"
                managed.updated_at = utc_now()
            self._emit(managed, "failed", {"error": managed.error})

    def _emit(self, managed: ManagedRun, event_type: str, data: dict) -> None:
        with managed.condition:
            event = RunEvent(
                sequence=len(managed.events) + 1, type=event_type, data=data
            )
            managed.events.append(event)
            managed.updated_at = event.created_at
            managed.condition.notify_all()

    def cancel(self, run_id: str) -> RunSummary:
        managed = self._managed(run_id)
        with managed.condition:
            if managed.status in TERMINAL_STATUSES:
                return self._summary(managed)
            if managed.future is not None and managed.future.cancel():
                managed.status = "cancelled"
                event_type = "cancelled"
            else:
                managed.status = "cancel_requested"
                event_type = "cancel_requested"
            managed.updated_at = utc_now()
        self._emit(
            managed,
            event_type,
            {
                "interruptible": event_type == "cancelled",
                "message": (
                    "Queued run cancelled"
                    if event_type == "cancelled"
                    else "Cancellation requested; the active provider call cannot be interrupted"
                ),
            },
        )
        return self._summary(managed)

    def events_after(self, run_id: str, sequence: int) -> tuple[list[RunEvent], bool]:
        managed = self._managed(run_id)
        with managed.condition:
            events = [item for item in managed.events if item.sequence > sequence]
            terminal = managed.status in TERMINAL_STATUSES
        return events, terminal

    def wait_for_events(self, run_id: str, sequence: int, timeout: float = 15) -> None:
        managed = self._managed(run_id)
        with managed.condition:
            if not any(item.sequence > sequence for item in managed.events):
                managed.condition.wait(timeout=timeout)

    def get(self, run_id: str) -> RunDetail:
        with self._lock:
            managed = self._runs.get(run_id)
        if managed is not None:
            summary = self._summary(managed)
            return RunDetail(**summary.model_dump(), result=managed.result)
        persisted = self.store.load(run_id)
        return self._persisted_detail(persisted)

    def list(self, *, limit: int = 50) -> list[RunSummary]:
        with self._lock:
            active = sorted(
                self._runs.values(), key=lambda item: item.created_at, reverse=True
            )
        items = [self._summary(item) for item in active]
        known = {item.run_id for item in items}
        for persisted in self.store.list(limit=limit):
            if persisted.run_id not in known:
                items.append(self._persisted_detail(persisted))
        return sorted(items, key=lambda item: item.created_at, reverse=True)[:limit]

    def _managed(self, run_id: str) -> ManagedRun:
        with self._lock:
            try:
                return self._runs[run_id]
            except KeyError as exc:
                raise FileNotFoundError(f"Agent run not found: {run_id}") from exc

    @staticmethod
    def _summary(managed: ManagedRun) -> RunSummary:
        result = managed.result
        return RunSummary(
            run_id=managed.run_id,
            request=managed.request,
            status=managed.status,
            route=result.route.mode if result else None,
            route_reason=result.route.reason if result else None,
            trace_id=managed.trace_id,
            evidence_count=len(result.evidence) if result else 0,
            worker_count=len(result.worker_packets) if result else 0,
            created_at=managed.created_at,
            updated_at=managed.updated_at,
            error=managed.error,
        )

    @staticmethod
    def _persisted_detail(run: AgentRun) -> RunDetail:
        return RunDetail(
            run_id=run.run_id,
            request=run.request,
            status=run.outcome,
            route=run.route.mode,
            route_reason=run.route.reason,
            trace_id=run.trace_id,
            evidence_count=len(run.evidence),
            worker_count=len(run.worker_packets),
            created_at=run.created_at,
            updated_at=run.created_at,
            result=run,
        )
