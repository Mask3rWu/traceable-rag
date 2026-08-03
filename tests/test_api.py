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


class _FakeAgent:
    langfuse = None

    def __init__(self, store: AgentRunStore) -> None:
        self.store = store

    def run(self, request, *, run_id=None, trace_id=None, callbacks=None):
        for callback in callbacks or []:
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


if __name__ == "__main__":
    unittest.main()
