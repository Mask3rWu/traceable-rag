from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.manager import RunManager
from src.research.agent_models import AgentAnswer, AgentRun, RouteDecision
from src.research.agent_store import AgentRunStore
from src.research.graph import RoutePolicyError


class _FakeAgent:
    langfuse = None

    def __init__(self, store: AgentRunStore) -> None:
        self.store = store

    def run(self, request, *, run_id=None, trace_id=None, callbacks=None):
        for callback in callbacks or []:
            if hasattr(callback, "emit"):
                callback.emit("route_selected", {"mode": "fast", "reason": "test"})
        run = AgentRun(
            run_id=run_id,
            request=request,
            route=RouteDecision(mode="fast", reason="test route"),
            answer=AgentAnswer(content="test answer"),
            trace_id=trace_id,
        )
        return run, self.store.save(run)


class ApiTest(unittest.TestCase):
    def test_create_stream_and_load_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AgentRunStore(Path(tmp))
            manager = RunManager(
                store=store,
                agent_factory=lambda: _FakeAgent(store),
                max_concurrent_runs=1,
            )
            with TestClient(create_app(manager)) as client:
                created = client.post("/api/runs", json={"request": "测试问题"})
                self.assertEqual(created.status_code, 202)
                run_id = created.json()["run_id"]

                deadline = time.monotonic() + 3
                detail = None
                while time.monotonic() < deadline:
                    detail = client.get(f"/api/runs/{run_id}").json()
                    if detail["status"] == "completed":
                        break
                    time.sleep(0.01)

                self.assertIsNotNone(detail)
                self.assertEqual(detail["status"], "completed")
                self.assertEqual(detail["result"]["run_id"], run_id)
                self.assertEqual(store.load(run_id).run_id, run_id)

                events = client.get(f"/api/runs/{run_id}/events")
                self.assertEqual(events.status_code, 200)
                self.assertIn("event: queued", events.text)
                self.assertIn("event: completed", events.text)

                runs = client.get("/api/runs").json()["items"]
                self.assertEqual(runs[0]["run_id"], run_id)

    def test_missing_run_returns_404(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AgentRunStore(Path(tmp))
            manager = RunManager(
                store=store,
                agent_factory=lambda: SimpleNamespace(),
                max_concurrent_runs=1,
            )
            with TestClient(create_app(manager)) as client:
                self.assertEqual(client.get("/api/runs/missing").status_code, 404)
                self.assertEqual(
                    client.post("/api/runs/missing/cancel").status_code, 404
                )

    def test_fast_expected_route_routed_away_exposes_reason(self):
        class _GuardAgent:
            langfuse = None

            def __init__(self, store):
                self.store = store

            def run(
                self,
                request,
                *,
                run_id=None,
                trace_id=None,
                callbacks=None,
                cancel_check=None,
                route_guard=None,
            ):
                raise RoutePolicyError(
                    mode="supervisor", reason="deliverable-style request"
                )

        with tempfile.TemporaryDirectory() as tmp:
            store = AgentRunStore(Path(tmp))
            manager = RunManager(
                store=store,
                agent_factory=lambda: _GuardAgent(store),
                max_concurrent_runs=1,
            )
            with TestClient(create_app(manager)) as client:
                created = client.post(
                    "/api/runs", json={"request": "快排", "expected_route": "fast"}
                )
                self.assertEqual(created.status_code, 202)
                run_id = created.json()["run_id"]

                deadline = time.monotonic() + 3
                detail = None
                while time.monotonic() < deadline:
                    detail = client.get(f"/api/runs/{run_id}").json()
                    if detail["status"] in {"routed_away", "failed"}:
                        break
                    time.sleep(0.01)

                self.assertIsNotNone(detail)
                self.assertEqual(detail["status"], "routed_away")
                self.assertEqual(detail["route"], "supervisor")
                self.assertEqual(detail["route_reason"], "deliverable-style request")
                self.assertIn("router: deliverable-style request", detail["error"])

                # SSE stream must terminate for a routed_away run.
                events = client.get(f"/api/runs/{run_id}/events")
                self.assertEqual(events.status_code, 200)
                self.assertIn("event: routed_away", events.text)

    def test_supervisor_expected_route_routed_away_exposes_reason(self):
        class _GuardAgent:
            langfuse = None

            def __init__(self, store):
                self.store = store

            def run(
                self,
                request,
                *,
                run_id=None,
                trace_id=None,
                callbacks=None,
                cancel_check=None,
                route_guard=None,
            ):
                raise RoutePolicyError(
                    mode="fast", reason="focused single-question request"
                )

        with tempfile.TemporaryDirectory() as tmp:
            store = AgentRunStore(Path(tmp))
            manager = RunManager(
                store=store,
                agent_factory=lambda: _GuardAgent(store),
                max_concurrent_runs=1,
            )
            with TestClient(create_app(manager)) as client:
                created = client.post(
                    "/api/runs", json={"request": "多章节报告", "expected_route": "supervisor"}
                )
                self.assertEqual(created.status_code, 202)
                run_id = created.json()["run_id"]

                deadline = time.monotonic() + 3
                detail = None
                while time.monotonic() < deadline:
                    detail = client.get(f"/api/runs/{run_id}").json()
                    if detail["status"] in {"routed_away", "failed"}:
                        break
                    time.sleep(0.01)

                self.assertIsNotNone(detail)
                self.assertEqual(detail["status"], "routed_away")
                self.assertEqual(detail["route"], "fast")
                self.assertEqual(detail["route_reason"], "focused single-question request")
                self.assertIn("router: focused single-question request", detail["error"])

    def test_metrics_are_wired_into_agent_runtime_hooks(self):
        captured = {}

        class _MetricAgent(_FakeAgent):
            def attach_metrics(self, metrics):
                captured["metrics"] = metrics

            def run(self, request, *, run_id=None, trace_id=None, callbacks=None):
                captured["metrics"].record_schema_validation(stage="plan", ok=True)
                return super().run(request, run_id=run_id, trace_id=trace_id, callbacks=callbacks)

        with tempfile.TemporaryDirectory() as tmp:
            store = AgentRunStore(Path(tmp))
            manager = RunManager(
                store=store,
                agent_factory=lambda: _MetricAgent(store),
                max_concurrent_runs=1,
            )
            with TestClient(create_app(manager)) as client:
                created = client.post("/api/runs", json={"request": "测试问题"})
                run_id = created.json()["run_id"]

                deadline = time.monotonic() + 3
                detail = None
                while time.monotonic() < deadline:
                    detail = client.get(f"/api/runs/{run_id}").json()
                    if detail["status"] == "completed":
                        break
                    time.sleep(0.01)

                self.assertEqual(detail["status"], "completed")
                self.assertEqual(
                    detail["metrics"]["schema"], [{"stage": "plan", "ok": True, "reason": None}]
                )

    def test_success_persists_metrics_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AgentRunStore(Path(tmp))
            manager = RunManager(
                store=store,
                agent_factory=lambda: _FakeAgent(store),
                max_concurrent_runs=1,
            )
            with TestClient(create_app(manager)) as client:
                created = client.post("/api/runs", json={"request": "测试问题"})
                run_id = created.json()["run_id"]

                deadline = time.monotonic() + 3
                detail = None
                while time.monotonic() < deadline:
                    detail = client.get(f"/api/runs/{run_id}").json()
                    if detail["status"] == "completed":
                        break
                    time.sleep(0.01)

                self.assertEqual(detail["status"], "completed")
                self.assertIsNotNone(detail["metrics"])
                self.assertTrue(store.metrics_path_for(run_id).is_file())
                # Reloading a persisted run still carries its metrics.
                reloaded = manager.get(run_id)
                self.assertIsNotNone(reloaded.metrics)

    def test_degraded_completed_rolls_up_with_layer(self):
        class _DegradedAgent(_FakeAgent):
            def attach_metrics(self, metrics):
                self.metrics = metrics

            def run(
                self,
                request,
                *,
                run_id=None,
                trace_id=None,
                callbacks=None,
                route_guard=None,
            ):
                self.metrics.record_outcome(
                    level="tool", result="degraded", reason="retryable-infra"
                )
                return super().run(
                    request, run_id=run_id, trace_id=trace_id, callbacks=callbacks
                )

        with tempfile.TemporaryDirectory() as tmp:
            store = AgentRunStore(Path(tmp))
            manager = RunManager(
                store=store,
                agent_factory=lambda: _DegradedAgent(store),
                max_concurrent_runs=1,
            )
            with TestClient(create_app(manager)) as client:
                created = client.post("/api/runs", json={"request": "坦克评估"})
                run_id = created.json()["run_id"]
                detail = self._wait_terminal(client, run_id, {"completed"})
                self.assertEqual(detail["status"], "completed")
                self.assertEqual(detail["degraded"]["layers"], ["retrieval"])

    def test_trace_jsonl_persisted_per_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AgentRunStore(Path(tmp))
            manager = RunManager(
                store=store,
                agent_factory=lambda: _FakeAgent(store),
                max_concurrent_runs=1,
            )
            with TestClient(create_app(manager)) as client:
                created = client.post("/api/runs", json={"request": "测试问题"})
                run_id = created.json()["run_id"]
                detail = self._wait_terminal(client, run_id, {"completed"})
                self.assertEqual(detail["status"], "completed")
                trace = store.trace_path_for(run_id)
                self.assertTrue(trace.is_file())
                lines = trace.read_text(encoding="utf-8").splitlines()
                self.assertTrue(lines)
                for line in lines:
                    import json

                    self.assertIsInstance(json.loads(line), dict)
                # Every in-memory event was persisted (queued/running/completed >= 3).
                self.assertGreaterEqual(len(lines), 3)

    def test_routed_away_records_wasted_tokens_and_calls(self):
        class _WasteGuard(_FakeAgent):
            def attach_metrics(self, metrics):
                self.metrics = metrics

            def run(
                self,
                request,
                *,
                run_id=None,
                trace_id=None,
                callbacks=None,
                cancel_check=None,
                route_guard=None,
            ):
                self.metrics.record_model_call(
                    model="m", prompt_tokens=100, completion_tokens=50,
                    latency_ms=10.0, ok=True,
                )
                self.metrics.record_tool_call(
                    tool="search_knowledge", latency_ms=5.0, ok=True
                )
                raise RoutePolicyError(
                    mode="supervisor", reason="deliverable-style request"
                )

        with tempfile.TemporaryDirectory() as tmp:
            store = AgentRunStore(Path(tmp))
            manager = RunManager(
                store=store,
                agent_factory=lambda: _WasteGuard(store),
                max_concurrent_runs=1,
            )
            with TestClient(create_app(manager)) as client:
                created = client.post(
                    "/api/runs",
                    json={"request": "快排", "expected_route": "fast"},
                )
                run_id = created.json()["run_id"]
                detail = self._wait_terminal(client, run_id, {"routed_away", "failed"})
                self.assertEqual(detail["status"], "routed_away")
                self.assertEqual(detail["wasted_tokens"], 150)
                self.assertEqual(detail["spurious_tool_calls"], 1)

    def _wait_terminal(self, client, run_id: str, terminal: set) -> dict:
        deadline = time.monotonic() + 5
        detail = None
        while time.monotonic() < deadline:
            detail = client.get(f"/api/runs/{run_id}").json()
            if detail["status"] in terminal:
                return detail
            time.sleep(0.01)
        self.fail(f"run {run_id} did not reach {terminal}: {detail}")


if __name__ == "__main__":
    unittest.main()