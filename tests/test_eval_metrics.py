from __future__ import annotations

import unittest

from src.research.eval_metrics import EvalCallbackHandler, RuntimeMetrics


class RuntimeMetricsTest(unittest.TestCase):
    def test_model_calls_summary_computes_cost_from_pricing(self):
        metrics = RuntimeMetrics()
        metrics.record_model_call(
            model="m", prompt_tokens=1_000_000, completion_tokens=0,
            latency_ms=100.0, ok=True,
        )
        metrics.record_model_call(
            model="m", prompt_tokens=0, completion_tokens=500_000,
            latency_ms=200.0, ok=True,
        )
        summary = metrics.model_calls_summary(pricing={"m": {"input": 1.0, "output": 2.0}})
        self.assertEqual(summary["count"], 2)
        # 1M in * $1/M + 0.5M out * $2/M = 1 + 1 = 2
        self.assertEqual(summary["total_cost_usd"], 2.0)
        self.assertEqual(summary["avg_latency_ms"], 150.0)
        self.assertEqual(summary["failures"], [])

    def test_model_call_cost_is_none_for_unknown_model(self):
        metrics = RuntimeMetrics()
        metrics.record_model_call(
            model="mystery", prompt_tokens=1000, completion_tokens=0,
            latency_ms=10.0, ok=True,
        )
        summary = metrics.model_calls_summary(pricing={"m": {"input": 1.0, "output": 2.0}})
        self.assertIsNone(summary["total_cost_usd"])
        self.assertEqual(summary["count"], 1)

    def test_model_call_cost_splits_cache_hit_and_miss(self):
        metrics = RuntimeMetrics()
        metrics.record_model_call(
            model="m", prompt_tokens=2_000_000, completion_tokens=500_000,
            prompt_cache_hit_tokens=1_500_000, prompt_cache_miss_tokens=500_000,
            latency_ms=100.0, ok=True,
        )
        pricing = {"m": {"input": 1.0, "input_cache_hit": 0.1, "output": 2.0}}
        summary = metrics.model_calls_summary(pricing=pricing)
        # 1.5M hit * 0.1 + 0.5M miss * 1.0 + 0.5M out * 2.0 = 0.15 + 0.5 + 1.0
        self.assertAlmostEqual(summary["total_cost_usd"], 1.65)

    def test_model_call_cost_falls_back_to_input_without_cache_tier(self):
        metrics = RuntimeMetrics()
        metrics.record_model_call(
            model="m", prompt_tokens=1_000_000, completion_tokens=0,
            prompt_cache_hit_tokens=500_000, prompt_cache_miss_tokens=500_000,
            latency_ms=1.0, ok=True,
        )
        pricing = {"m": {"input": 1.0, "output": 2.0}}
        summary = metrics.model_calls_summary(pricing=pricing)
        self.assertEqual(summary["total_cost_usd"], 1.0)

    def test_model_failures_are_collected(self):
        metrics = RuntimeMetrics()
        metrics.record_model_call(
            model="m", prompt_tokens=0, completion_tokens=0,
            latency_ms=10.0, ok=False, error_hint="ValueError: bad json",
        )
        summary = metrics.model_calls_summary(pricing={"m": {"input": 1.0, "output": 2.0}})
        self.assertEqual(summary["count"], 1)
        self.assertEqual(summary["failures"], [{"model": "m", "reason": "ValueError: bad json"}])

    def test_tool_calls_summary_groups_by_tool_and_counts_failures(self):
        metrics = RuntimeMetrics()
        metrics.record_tool_call(tool="search_knowledge", latency_ms=100.0, ok=True)
        metrics.record_tool_call(tool="search_knowledge", latency_ms=50.0, ok=True)
        metrics.record_tool_call(tool="submit_answer", latency_ms=5.0, ok=False, error="boom")
        summary = metrics.tool_calls_summary()
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["by_tool"]["search_knowledge"]["count"], 2)
        self.assertEqual(summary["by_tool"]["search_knowledge"]["success"], 2)
        self.assertEqual(summary["by_tool"]["submit_answer"]["failed"], 1)
        self.assertEqual(summary["failures"], [{"tool": "submit_answer", "reason": "boom"}])

    def test_schema_validation_summary_counts_passes_and_failures(self):
        metrics = RuntimeMetrics()
        metrics.record_schema_validation(stage="plan", ok=True)
        metrics.record_schema_validation(stage="submit_chapter", ok=False, reason="budget")
        summary = metrics.schema_validation_summary()
        self.assertEqual(summary, {"attempts": 2, "passed": 1, "failed": 1,
                                   "failures": [{"stage": "submit_chapter", "reason": "budget"}]})

    def test_search_helpers_filter_search_knowledge_calls(self):
        metrics = RuntimeMetrics()
        metrics.record_tool_call(tool="search_knowledge", latency_ms=300.0, ok=True)
        metrics.record_tool_call(tool="read_evidence", latency_ms=1.0, ok=True)
        self.assertEqual(metrics.search_count(), 1)
        self.assertEqual(metrics.search_latencies_ms(), [300.0])

    def test_phase_summary_groups_by_phase_and_computes_cost(self):
        metrics = RuntimeMetrics()
        metrics.record_model_call(
            model="m", prompt_tokens=1_000_000, completion_tokens=0,
            latency_ms=100.0, ok=True, phase="router",
        )
        metrics.record_model_call(
            model="m", prompt_tokens=0, completion_tokens=500_000,
            latency_ms=200.0, ok=True, phase="worker",
        )
        metrics.record_model_call(
            model="m", prompt_tokens=1_000, completion_tokens=0,
            latency_ms=10.0, ok=True,
        )
        summary = metrics.phase_summary(pricing={"m": {"input": 1.0, "output": 2.0}})
        self.assertEqual(summary["router"]["model_calls"], 1)
        self.assertEqual(summary["router"]["cost_usd"], 1.0)
        self.assertEqual(summary["worker"]["cost_usd"], 1.0)
        self.assertEqual(summary["worker"]["total_latency_ms"], 200.0)
        self.assertEqual(summary["unknown"]["model_calls"], 1)

    def test_phase_summary_cost_none_for_unknown_model(self):
        metrics = RuntimeMetrics()
        metrics.record_model_call(
            model="mystery", prompt_tokens=100, completion_tokens=0,
            latency_ms=1.0, ok=True, phase="router",
        )
        summary = metrics.phase_summary(pricing={"m": {"input": 1, "output": 2}})
        self.assertIsNone(summary["router"]["cost_usd"])
        self.assertEqual(summary["router"]["failed"], 0)

    def test_phase_summary_tracks_failures_and_empty(self):
        metrics = RuntimeMetrics()
        self.assertEqual(metrics.phase_summary(), {})
        metrics.record_model_call(
            model="m", prompt_tokens=0, completion_tokens=0,
            latency_ms=None, ok=False, phase="worker", error_hint="boom",
        )
        summary = metrics.phase_summary(pricing={"m": {"input": 1, "output": 2}})
        self.assertEqual(summary["worker"]["failed"], 1)


class EvalCallbackHandlerTest(unittest.TestCase):
    def test_llm_end_records_usage_and_latency(self):
        metrics = RuntimeMetrics()
        handler = EvalCallbackHandler(metrics)

        class _Usage:
            prompt_tokens = 10
            completion_tokens = 5

        class _Response:
            llm_output = {"token_usage": _Usage()}

        handler.on_llm_start({}, [], run_id="r", invocation_params={"model": "m"})
        handler.on_llm_end(_Response(), run_id="r")
        summary = metrics.model_calls_summary(pricing={"m": {"input": 1.0, "output": 2.0}})
        self.assertEqual(summary["count"], 1)
        self.assertIsNotNone(summary["avg_latency_ms"])
        self.assertIsNotNone(summary["total_cost_usd"])

    def test_llm_end_records_cache_token_split(self):
        metrics = RuntimeMetrics()
        handler = EvalCallbackHandler(metrics)

        class _Response:
            llm_output = {
                "token_usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 200,
                    "prompt_cache_hit_tokens": 400,
                    "prompt_cache_miss_tokens": 600,
                }
            }

        handler.on_llm_start({}, [], run_id="r", invocation_params={"model": "m"})
        handler.on_llm_end(_Response(), run_id="r")
        pricing = {"m": {"input": 1.0, "input_cache_hit": 0.1, "output": 2.0}}
        summary = metrics.model_calls_summary(pricing=pricing)
        # (400*0.1 + 600*1.0 + 200*2.0) / 1M = 1040 / 1M
        self.assertAlmostEqual(summary["total_cost_usd"], 0.00104)

    def test_llm_error_records_failure(self):
        metrics = RuntimeMetrics()
        handler = EvalCallbackHandler(metrics)
        handler.on_llm_start({}, [], run_id="r", invocation_params={"model": "m"})
        handler.on_llm_error(ValueError("timeout"), run_id="r")
        summary = metrics.model_calls_summary(pricing={"m": {"input": 1.0, "output": 2.0}})
        self.assertEqual(summary["count"], 1)
        self.assertIn("ValueError: timeout", summary["failures"][0]["reason"])

    def test_tool_end_records_success(self):
        metrics = RuntimeMetrics()
        handler = EvalCallbackHandler(metrics)
        handler.on_tool_start({}, "", run_id="r", name="search_knowledge")
        handler.on_tool_end("{}", run_id="r")
        summary = metrics.tool_calls_summary()
        self.assertEqual(summary["by_tool"]["search_knowledge"]["success"], 1)

    def test_tool_error_records_failure(self):
        metrics = RuntimeMetrics()
        handler = EvalCallbackHandler(metrics)
        handler.on_tool_start({}, "", run_id="r", name="submit_chapter")
        handler.on_tool_error(RuntimeError("bad"), run_id="r")
        summary = metrics.tool_calls_summary()
        self.assertEqual(summary["by_tool"]["submit_chapter"]["failed"], 1)
        self.assertIn("RuntimeError: bad", summary["failures"][0]["reason"])

    def test_llm_end_records_phase_from_config_tags(self):
        metrics = RuntimeMetrics()
        handler = EvalCallbackHandler(metrics)

        class _Response:
            llm_output = {}

        handler.on_llm_start(
            {}, [], run_id="r",
            invocation_params={"model": "m"}, tags=["x", "phase:worker"],
        )
        handler.on_llm_end(_Response(), run_id="r")
        phases = metrics.phase_summary(pricing={"m": {"input": 1.0, "output": 2.0}})
        self.assertEqual(phases["worker"]["model_calls"], 1)
        self.assertNotIn("unknown", phases)


if __name__ == "__main__":
    unittest.main()