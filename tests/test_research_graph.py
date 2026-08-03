from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage

from src.research.agent_models import RouteDecision
from src.research.agent_store import AgentRunStore
from src.research.graph import AgentRuntime
from src.research.tools import EvidenceWorkspace


def _tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


class _BoundModel:
    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = responses

    def invoke(self, messages, config=None):
        del messages, config
        if not self.responses:
            raise AssertionError("No scripted response remains")
        return self.responses.pop(0)


class _StructuredRouter:
    def __init__(self, model: "_ScriptedModel") -> None:
        self.model = model

    def invoke(self, messages, config=None):
        del messages, config
        return RouteDecision(mode=self.model.route, reason="scripted route")


class _ScriptedModel:
    def __init__(
        self,
        *,
        route: str,
        fast: list[AIMessage] | None = None,
        supervisor: list[AIMessage] | None = None,
        worker: list[AIMessage] | None = None,
    ) -> None:
        self.route = route
        self.fast = fast or []
        self.supervisor = supervisor or []
        self.worker = worker or []

    def bind_tools(self, tools):
        names = {item.name for item in tools}
        if "delegate_research" in names:
            return _BoundModel(self.supervisor)
        if "submit_research" in names:
            return _BoundModel(self.worker)
        return _BoundModel(self.fast)

    def with_structured_output(self, schema, method=None):
        del schema, method
        return _StructuredRouter(self)


class _Verifier:
    def verify_claim(self, claim, evidence_by_id):
        del evidence_by_id
        return claim.model_copy(update={"citation_verified": True})


class _Workspace(EvidenceWorkspace):
    def __init__(self) -> None:
        self.verifier = _Verifier()
        self.reset_count = 0

    @property
    def search_count(self):
        return 1

    @property
    def evidence(self):
        return []

    def reset(self):
        self.reset_count += 1

    def make_retrieval_tools(self):
        return []

    def validate_evidence_ids(self, evidence_ids):
        del evidence_ids

    def evidence_by_id(self):
        return {}


class ResearchGraphTest(unittest.TestCase):
    def test_router_uses_fast_react_path(self):
        model = _ScriptedModel(
            route="fast",
            fast=[
                _tool_call(
                    "submit_answer",
                    {
                        "content": "快速答案",
                        "evidence_ids": ["ev-test"],
                        "limitations": [],
                    },
                    "fast-submit",
                )
            ],
        )
        workspace = _Workspace()
        with tempfile.TemporaryDirectory() as tmp:
            runtime = AgentRuntime(
                model=model,
                workspace=workspace,
                store=AgentRunStore(Path(tmp)),
            )
            run, path = runtime.run("一个聚焦问题")

        self.assertEqual(run.route.mode, "fast")
        self.assertEqual(run.answer.content, "快速答案")
        self.assertTrue(path.name == "run.json")
        self.assertEqual(workspace.reset_count, 1)

    def test_supervisor_with_insufficient_worker_refuses_ungrounded_answer(self):
        model = _ScriptedModel(
            route="supervisor",
            supervisor=[
                _tool_call(
                    "delegate_research",
                    {
                        "task": "研究毁伤等级",
                        "acceptance_criteria": ["识别等级体系"],
                    },
                    "delegate-1",
                ),
                _tool_call(
                    "submit_answer",
                    {
                        "content": "标准草案",
                        "evidence_ids": [],
                        "limitations": ["知识库证据不足"],
                    },
                    "supervisor-submit",
                ),
            ],
            worker=[
                _tool_call(
                    "submit_research",
                    {
                        "task": "研究毁伤等级",
                        "status": "insufficient",
                        "summary": "未找到足够证据",
                        "claims": [],
                        "conflicts": [],
                        "gaps": ["缺少等级定义"],
                        "evidence_ids": [],
                    },
                    "worker-submit",
                )
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            runtime = AgentRuntime(
                model=model,
                workspace=_Workspace(),
                store=AgentRunStore(Path(tmp)),
                max_steps=2,
            )
            run, _ = runtime.run("生成一份毁伤评估标准")

        self.assertEqual(run.route.mode, "supervisor")
        self.assertEqual(run.answer.content, "研究调度未能在执行预算内完成。")
        self.assertEqual(len(run.worker_packets), 1)
        self.assertEqual(run.worker_packets[0].task, "研究毁伤等级")
        self.assertEqual(run.worker_packets[0].status, "insufficient")


if __name__ == "__main__":
    unittest.main()
