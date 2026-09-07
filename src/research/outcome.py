"""Run outcome classification: component level x result x reason.

Pure rules that map exceptions and soft control signals to a normalized
outcome triple, with no I/O. The run manager (terminal state), the persisted
event trace, and the metrics aggregator all consume this single source so the
result taxonomy never drifts between the trace and the aggregated report.

``level`` identifies the component that produced the outcome (tool / model /
task); ``result`` is its coarse outcome (ok / degraded / failed); ``reason``
tells consumers why a degraded or failed outcome happened. Backoff policy
binds to ``reason``, not to ``level``: ``retryable-infra`` from a model call
and from a tool call are the same reason and share one retry path.
"""
from __future__ import annotations

from typing import Literal, NamedTuple

Level = Literal["tool", "model", "task"]
Result = Literal["ok", "degraded", "failed"]
Reason = Literal["retryable-infra", "permanent-config", "logic", "control"]


class SoftSignal(NamedTuple):
    """A control outcome that is not an exception.

    ``kind`` is ``"cancel"`` (a client interrupted the run) or ``"budget"``
    (a step budget was reached). These are deliberate, non-crash terminal
    signals: callers must translate them to their terminal (e.g. ``cancelled``)
    and must never put them on a retry path.
    """

    kind: Literal["cancel", "budget"]


class Outcome(NamedTuple):
    level: Level
    result: Result
    reason: Reason


def _status_code(exc: BaseException) -> int | None:
    """Extract an HTTP-ish status code from a provider/library exception."""
    for attr in ("status_code", "status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def _classify_reason(exc: BaseException) -> Reason:
    code = _status_code(exc)
    if code is not None:
        if code in (401, 403, 404):
            return "permanent-config"
        if code >= 500:
            return "retryable-infra"
    if isinstance(exc, (ConnectionError, TimeoutError, InterruptedError, OSError)):
        return "retryable-infra"
    name = type(exc).__name__
    if any(token in name for token in ("Operational", "Connection", "Timeout")):
        return "retryable-infra"
    if any(
        token in name
        for token in (
            "Authentication",
            "PermissionDenied",
            "NotFound",
            "Configuration",
            "NotImplemented",
        )
    ):
        return "permanent-config"
    if isinstance(exc, (ValueError, KeyError, TypeError, AssertionError, AttributeError)):
        return "logic"
    # Unknown/unrecognized exceptions default toward retry so a transient
    # hiccup that upstream did not label precisely is not mis-tagged as a
    # permanent configuration error.
    return "retryable-infra"


def classify(level: Level, cause: BaseException | SoftSignal) -> Outcome:
    """Normalize an exception or control signal to a level x result x reason triple.

    A bare exception is always ``result=failed``; the reason follows from the
    exception's nature (service status code, builtin classes, name heuristics).
    A :class:`SoftSignal` maps to ``reason=control`` — not a failure, never a
    retry candidate.
    """
    if isinstance(cause, SoftSignal):
        return Outcome(level, "failed", "control")
    return Outcome(level, "failed", _classify_reason(cause))


def degraded(level: Level) -> Outcome:
    """A component completed on a degraded path (e.g. listed dense->BM25 fallback)."""
    return Outcome(level, "degraded", "retryable-infra")