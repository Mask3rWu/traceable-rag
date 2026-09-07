"""Unit tests for run outcome classification and the degraded run-status modifier."""
from __future__ import annotations

import unittest

from src.api.models import DegradedStatus, RunSummary, utc_now
from src.research.outcome import SoftSignal, classify, degraded


class _HttpError(Exception):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(str(status_code))


class ClassifyTest(unittest.TestCase):
    def test_tool_transient_infrastructure(self):
        outcome = classify("tool", ConnectionError("db down"))
        self.assertEqual(outcome, ("tool", "failed", "retryable-infra"))

    def test_model_http_500_is_retryable_infra(self):
        # A model API 500 shares the same reason as a tool DB outage.
        outcome = classify("model", _HttpError(500))
        self.assertEqual(outcome, ("model", "failed", "retryable-infra"))

    def test_permanent_config_auth(self):
        outcome = classify("tool", _HttpError(401))
        self.assertEqual(outcome, ("tool", "failed", "permanent-config"))

    def test_logic_unknown_evidence_id(self):
        outcome = classify("tool", ValueError("unknown evidence id"))
        self.assertEqual(outcome.reason, "logic")
        self.assertEqual(outcome.result, "failed")

    def test_control_signal_is_not_a_retryable_failure(self):
        outcome = classify("task", SoftSignal("cancel"))
        self.assertEqual(outcome.level, "task")
        self.assertEqual(outcome.reason, "control")
        self.assertNotEqual(outcome.reason, "retryable-infra")

    def test_degraded_helper(self):
        self.assertEqual(degraded("tool"), ("tool", "degraded", "retryable-infra"))


class DegradedStatusTest(unittest.TestCase):
    def test_degraded_allowed_with_completed(self):
        summary = RunSummary(
            run_id="run1",
            request="r",
            status="completed",
            created_at=utc_now(),
            updated_at=utc_now(),
            degraded=DegradedStatus(layers=["retrieval"]),
        )
        self.assertEqual(summary.degraded.layers, ["retrieval"])

    def test_degraded_rejected_with_failed(self):
        with self.assertRaises(ValueError):
            RunSummary(
                run_id="run1",
                request="r",
                status="failed",
                created_at=utc_now(),
                updated_at=utc_now(),
                degraded=DegradedStatus(layers=["retrieval"]),
            )

    def test_degraded_defaults_to_none(self):
        summary = RunSummary(
            run_id="run1",
            request="r",
            status="completed",
            created_at=utc_now(),
            updated_at=utc_now(),
        )
        self.assertIsNone(summary.degraded)


if __name__ == "__main__":
    unittest.main()