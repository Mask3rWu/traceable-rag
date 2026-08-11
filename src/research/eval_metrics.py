"""Runtime telemetry collector for end-to-end agent evaluation.

Self-contained and offline: the collected metrics are emitted as plain JSON by
the eval runner, never read back from Langfuse or any external store. This is
the single source of truth for one agent run's model-call / tool-call /
schema-validation / retrieval metrics.

Two feeders write into a :class:`RuntimeMetrics`:

* :class:`EvalCallbackHandler` — an optional LangChain ``BaseCallbackHandler``
  that turns model/tool callbacks into metric records. It is attached by the
  eval runner only; production run paths never need it.
* explicit schema hooks in ``AgentRuntime`` (guarded, off by default) that
  record structured-output validation outcomes in the places the runtime
  validates model-emitted artifacts.

Cost is computed from reported token usage against a pricing table keyed by
model name (``eval/runtime/pricing.yaml``). If a model has no entry, cost is
reported as ``None`` rather than a fabricated zero.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


def _as_usage(raw: Any) -> dict[str, int]:
    """Normalize LangChain token-usage payloads (dict or object) to ints."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raw = getattr(raw, "model_dump", lambda: getattr(raw, "__dict__", {}))()
    return {
        key: int(value)
        for key, value in raw.items()
        if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
        and isinstance(value, (int, float))
    }


class RuntimeMetrics:
    """Thread-safe accumulator for one agent run's evaluation telemetry."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._model_calls: list[dict[str, Any]] = []
        self._tool_calls: list[dict[str, Any]] = []
        self._schema: list[dict[str, Any]] = []

    # -- recorders -----------------------------------------------------------

    def record_model_call(
        self,
        *,
        model: str | None,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float | None,
        ok: bool,
        error_hint: str | None = None,
    ) -> None:
        with self._lock:
            self._model_calls.append(
                {
                    "model": model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "latency_ms": latency_ms,
                    "ok": ok,
                    "error_hint": error_hint,
                }
            )

    def record_tool_call(
        self,
        *,
        tool: str | None,
        latency_ms: float,
        ok: bool,
        error: str | None = None,
    ) -> None:
        with self._lock:
            self._tool_calls.append(
                {
                    "tool": tool,
                    "latency_ms": latency_ms,
                    "ok": ok,
                    "error": error,
                }
            )

    def record_schema_validation(
        self, *, stage: str, ok: bool, reason: str | None = None
    ) -> None:
        with self._lock:
            self._schema.append(
                {"stage": stage, "ok": ok, "reason": reason}
            )

    # -- summaries -----------------------------------------------------------

    def model_calls_summary(self, pricing: dict[str, dict] | None = None) -> dict:
        pricing = pricing or {}
        with self._lock:
            calls = list(self._model_calls)
        if not calls:
            return {
                "count": 0,
                "total_cost_usd": None,
                "avg_latency_ms": None,
                "failures": [],
            }
        total_cost = 0.0
        has_cost = True
        latencies = [c["latency_ms"] for c in calls if c["latency_ms"] is not None]
        for call in calls:
            price = pricing.get(call["model"] or "")
            if price is None:
                has_cost = False
                continue
            cost = (
                call["prompt_tokens"] * price["input"]
                + call["completion_tokens"] * price["output"]
            ) / 1_000_000
            total_cost += cost
        return {
            "count": len(calls),
            "total_cost_usd": round(total_cost, 6) if has_cost else None,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1)
            if latencies
            else None,
            "failures": [
                {"model": c["model"], "reason": c["error_hint"]}
                for c in calls
                if not c["ok"] and c["error_hint"]
            ],
        }

    def tool_calls_summary(self) -> dict:
        with self._lock:
            calls = list(self._tool_calls)
        by_tool: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "success": 0, "failed": 0, "total_latency_ms": 0}
        )
        failures: list[dict[str, str]] = []
        for call in calls:
            key = call["tool"] or "(unknown)"
            bucket = by_tool[key]
            bucket["count"] += 1
            bucket["success"] += 1 if call["ok"] else 0
            bucket["failed"] += 0 if call["ok"] else 1
            bucket["total_latency_ms"] += call["latency_ms"] or 0
            if not call["ok"] and call["error"]:
                failures.append({"tool": key, "reason": call["error"]})
        return {
            "count": len(calls),
            "by_tool": dict(by_tool),
            "failures": failures,
        }

    def schema_validation_summary(self) -> dict:
        with self._lock:
            entries = list(self._schema)
        passed = sum(1 for entry in entries if entry["ok"])
        return {
            "attempts": len(entries),
            "passed": passed,
            "failed": len(entries) - passed,
            "failures": [
                {"stage": entry["stage"], "reason": entry["reason"]}
                for entry in entries
                if not entry["ok"] and entry["reason"]
            ],
        }

    def search_latencies_ms(self) -> list[float]:
        with self._lock:
            return [
                round(call["latency_ms"], 1)
                for call in self._tool_calls
                if call["tool"] == "search_knowledge"
                and call["latency_ms"] is not None
            ]

    def search_count(self) -> int:
        with self._lock:
            return sum(
                1
                for call in self._tool_calls
                if call["tool"] == "search_knowledge"
            )

    # -- persistence ---------------------------------------------------------

    def snapshot(self) -> dict:
        """Return a JSON-serializable snapshot of the raw telemetry.

        Deliberately omits pricing/cost so any persistence layer (e.g. the API)
        can store it without knowing the pricing table; the eval runner applies
        pricing when building the report via :meth:`from_snapshot`.
        """
        with self._lock:
            return {
                "model_calls": list(self._model_calls),
                "tool_calls": list(self._tool_calls),
                "schema": list(self._schema),
            }

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> "RuntimeMetrics":
        """Rebuild a :class:`RuntimeMetrics` from a persisted snapshot.

        Lets the eval runner reconstruct cost/aggregate summaries from a
        snapshot the API wrote, without re-running the agent.
        """
        metrics = cls()
        metrics._model_calls = list(snapshot.get("model_calls") or [])
        metrics._tool_calls = list(snapshot.get("tool_calls") or [])
        metrics._schema = list(snapshot.get("schema") or [])
        return metrics


class EvalCallbackHandler(BaseCallbackHandler):
    """Convert LangChain model/tool callbacks into :class:`RuntimeMetrics`.

    Attached by the eval runner via ``agent.run(..., callbacks=[...])``. It
    records latency, token usage, and success/failure per model call and per
    tool call. ``raise_error`` stays False so evaluation never alters agent
    behavior.
    """

    raise_error = False

    def __init__(self, metrics: RuntimeMetrics) -> None:
        super().__init__()
        self.metrics = metrics
        self._llm_start: dict[str, tuple[str | None, float]] = {}
        self._tool_start: dict[str, tuple[str | None, float]] = {}

    @staticmethod
    def _model_name(serialized: dict, kwargs: dict) -> str | None:
        params = kwargs.get("invocation_params") or {}
        if isinstance(params, dict) and params.get("model"):
            return str(params["model"])
        if isinstance(serialized, dict) and serialized.get("model"):
            return str(serialized["model"])
        return None

    @staticmethod
    def _tool_name(serialized: dict, kwargs: dict) -> str | None:
        name = kwargs.get("name")
        if name:
            return str(name)
        if isinstance(serialized, dict) and serialized.get("name"):
            return str(serialized["name"])
        return None

    # -- LLM -----------------------------------------------------------------

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs) -> None:
        self._llm_start[run_id] = (
            self._model_name(serialized, kwargs),
            time.perf_counter(),
        )

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        start = self._llm_start.pop(run_id, None)
        latency_ms = None
        if start is not None:
            latency_ms = (time.perf_counter() - start[1]) * 1000
        usage = {}
        llm_output = getattr(response, "llm_output", None) or {}
        if isinstance(llm_output, dict):
            usage = _as_usage(llm_output.get("token_usage"))
        self.metrics.record_model_call(
            model=start[0] if start else None,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
            ok=True,
        )

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        start = self._llm_start.pop(run_id, None)
        latency_ms = None
        if start is not None:
            latency_ms = (time.perf_counter() - start[1]) * 1000
        self.metrics.record_model_call(
            model=start[0] if start else None,
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=latency_ms,
            ok=False,
            error_hint=f"{type(error).__name__}: {error}",
        )

    # -- tools ---------------------------------------------------------------

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs) -> None:
        self._tool_start[run_id] = (
            self._tool_name(serialized, kwargs),
            time.perf_counter(),
        )

    def on_tool_end(self, output, *, run_id, **kwargs) -> None:
        start = self._tool_start.pop(run_id, None)
        if start is None:
            return
        latency_ms = (time.perf_counter() - start[1]) * 1000
        self.metrics.record_tool_call(
            tool=start[0], latency_ms=latency_ms, ok=True
        )

    def on_tool_error(self, error, *, run_id, **kwargs) -> None:
        start = self._tool_start.pop(run_id, None)
        latency_ms = None
        if start is not None:
            latency_ms = (time.perf_counter() - start[1]) * 1000
        self.metrics.record_tool_call(
            tool=start[0] if start else None,
            latency_ms=latency_ms,
            ok=False,
            error=f"{type(error).__name__}: {error}",
        )