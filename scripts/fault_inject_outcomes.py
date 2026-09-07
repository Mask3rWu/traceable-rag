"""故障注入：验证 run 结果分类在真实 run 边界上落到预期终态与 reason。

只读诊断脚本，不改任何生产逻辑。驱动 :class:`RunManager`（不启 HTTP）让假 agent
在运行期抛不同异常 / 记录 degraded，断言：终态正确、本地 trace 末尾事件带预期
``level×result×reason``、且 ``logic``/``control`` 不被误标 ``retryable-infra``。

用法：
    python scripts/fault_inject_outcomes.py
每类注入输出 PASS/FAIL，全量通过后 exit 0。
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

if str(PROJECT_ROOT) not in sys.path:  # noqa: E402
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.manager import RunManager  # noqa: E402
from src.research.agent_models import AgentAnswer, AgentRun, RouteDecision  # noqa: E402
from src.research.agent_store import AgentRunStore  # noqa: E402
from src.research.outcome import SoftSignal, classify  # noqa: E402


def _wait_terminal(manager: RunManager, run_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        detail = manager.get(run_id)
        last = detail.model_dump()
        if last["status"] in {"completed", "incomplete", "failed", "cancelled"}:
            return last
        time.sleep(0.01)
    raise AssertionError(f"run {run_id} did not reach terminal: {last}")


def _trace_reason(store: AgentRunStore, run_id: str) -> dict | None:
    """Read the last trace line's classification triple (if present)."""
    path = store.trace_path_for(run_id)
    if not path.is_file():
        return None
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for line in reversed(lines):
        data = json.loads(line)
        if data.get("type") in {"failed", "completed"} and "reason" in data.get("data", {}):
            return data["data"]
    return None


class _RaiseAgent:
    langfuse = None

    def __init__(self, store: AgentRunStore, exc: BaseException) -> None:
        self.store = store
        self.exc = exc

    def run(self, request, *, run_id=None, trace_id=None, callbacks=None, **kwargs):
        raise self.exc


class _DegradedAgent:
    langfuse = None

    def __init__(self, store: AgentRunStore) -> None:
        self.store = store

    def attach_metrics(self, metrics) -> None:
        self.metrics = metrics

    def run(self, request, *, run_id=None, trace_id=None, callbacks=None, **kwargs):
        self.metrics.record_outcome(
            level="tool", result="degraded", reason="retryable-infra"
        )
        run = AgentRun(
            run_id=run_id,
            request=request,
            route=RouteDecision(mode="fast", reason="smoke"),
            answer=AgentAnswer(content="ok"),
            trace_id=trace_id,
        )
        return run, self.store.save(run)


def _quality_outcome(agent_factory, store) -> dict:
    manager = RunManager(
        store=store, agent_factory=agent_factory, max_concurrent_runs=1
    )
    run = manager.create("诊断问题")
    detail = _wait_terminal(manager, run.run_id)
    detail["trace"] = _trace_reason(store, run.run_id)
    return detail


def main() -> int:
    failures: list[str] = []
    checks: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        store = AgentRunStore(Path(tmp))

        infra = _quality_outcome(
            lambda: _RaiseAgent(store, ConnectionError("db down")), store
        )
        logic = _quality_outcome(
            lambda: _RaiseAgent(
                store, ValueError("model produced empty evidence")
            ),
            store,
        )
        degraded = _quality_outcome(lambda: _DegradedAgent(store), store)

        def check(name: str, cond: bool, detail: str) -> None:
            line = f"{'PASS' if cond else 'FAIL'}: {name} — {detail}"
            print(line)
            checks.append(name)
            if not cond:
                failures.append(name)

        check(
            "infra->failed+retryable-infra",
            infra["status"] == "failed"
            and infra["trace"] is not None
            and infra["trace"]["reason"] == "retryable-infra"
            and infra["trace"]["level"] == "task",
            f"status={infra['status']} trace={infra['trace']}",
        )
        check(
            "logic->failed+logic",
            logic["status"] == "failed"
            and logic["trace"] is not None
            and logic["trace"]["reason"] == "logic",
            f"status={logic['status']} trace={logic['trace']}",
        )
        check(
            "degraded->completed+retrieval layer",
            degraded["status"] == "completed"
            and degraded["degraded"] is not None
            and "retrieval" in degraded["degraded"]["layers"],
            f"status={degraded['status']} degraded={degraded['degraded']}",
        )
        # 6.3 同为 failed，reason 必须不同（基础设施 vs 内容质量）。
        check(
            "infra vs logic distinguishable",
            logic["status"] == "failed"
            and infra["status"] == "failed"
            and infra["trace"]["reason"] != logic["trace"]["reason"],
            f"infra={infra['trace']['reason']} logic={logic['trace']['reason']}",
        )
        # control 不是可重试失败（单位级反向断言，脚本级不可控取消时机）。
        control = classify("task", SoftSignal("cancel"))
        check(
            "control not retryable-infra",
            control.reason == "control" and control.reason != "retryable-infra",
            f"control={control.reason}",
        )
        check(
            "logic not mislabeled retryable-infra",
            logic["trace"]["reason"] != "retryable-infra",
            f"logic reason={logic['trace']['reason']}",
        )

    print(f"\npassed {len(checks) - len(failures)}/{len(checks)} checks")
    if failures:
        print(f"FAIL: {failures}")
        return 1
    print("exit 0 — all fault-injection outcome checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())