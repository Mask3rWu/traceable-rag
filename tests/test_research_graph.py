from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.research.agent_models import (
    ConsistencyReport,
    DocumentPlan,
    ResearchPacket,
    RouteDecision,
)
from src.research.agent_store import AgentRunStore
from src.research.graph import AgentRuntime
from src.research.tools import EvidenceAliasRegistry, EvidenceWorkspace


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


class _StructuredOutput:
    def __init__(self, model: "_ScriptedModel", schema) -> None:
        self.model = model
        self.schema = schema

    def invoke(self, messages, config=None):
        del messages, config
        if self.schema is RouteDecision:
            return RouteDecision(mode=self.model.route, reason="scripted route")
        if self.schema is DocumentPlan:
            return self.model.plan
        if self.schema is ConsistencyReport:
            return self.model.review
        raise AssertionError(f"Unsupported structured schema: {self.schema}")


class _ScriptedModel:
    def __init__(
        self,
        *,
        route: str,
        plan: DocumentPlan | None = None,
        review: ConsistencyReport | None = None,
        fast: list[AIMessage] | None = None,
        worker: list[AIMessage] | None = None,
    ) -> None:
        self.route = route
        self.plan = plan
        self.review = review or ConsistencyReport()
        self.fast = fast or []
        self.worker = worker or []
        self.review_calls = 0

    def bind_tools(self, tools):
        names = {item.name for item in tools}
        if "submit_chapter" in names:
            return _BoundModel(self.worker)
        return _BoundModel(self.fast)

    def with_structured_output(self, schema, method=None):
        del method
        return _StructuredOutput(self, schema)


class _Workspace(EvidenceWorkspace):
    def __init__(self) -> None:
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


def _chapter(*, chapter_id: str, ordinal: int, title: str, **extras) -> dict:
    base = {
        "chapter_id": chapter_id,
        "ordinal": ordinal,
        "title": title,
        "objective": f"{title}研究",
        "research_questions": [f"{title}是什么"],
        "acceptance_criteria": [f"形成{title}"],
    }
    base.update(extras)
    return base


class ResearchGraphTest(unittest.TestCase):
    def test_failed_packet_keeps_last_tool_validation_error(self):
        chapter = DocumentPlan.model_validate(
            {
                "title": "标准",
                "rationale": "测试",
                "chapters": [_chapter(chapter_id="scope", ordinal=1, title="范围")],
            }
        ).chapters[0]
        packet = AgentRuntime._failed_packet(
            chapter,
            {
                "task": chapter.objective,
                "messages": [
                    HumanMessage(content="研究"),
                    ToolMessage(
                        content="gaps must not contain diagnostic self-assessment",
                        tool_call_id="submit-1",
                        name="submit_chapter",
                        status="error",
                    ),
                ],
            },
            False,
        )

        self.assertEqual(packet.status, "failed")
        self.assertIn("diagnostic self-assessment", packet.diagnostics[0])

    def test_document_plan_rejects_dependency_cycle(self):
        with self.assertRaisesRegex(ValueError, "cycle"):
            DocumentPlan.model_validate(
                {
                    "title": "循环计划",
                    "rationale": "测试",
                    "chapters": [
                        _chapter(chapter_id="a", ordinal=1, title="A", depends_on=["b"]),
                        _chapter(chapter_id="b", ordinal=2, title="B", depends_on=["a"]),
                    ],
                }
            )

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
        plan = DocumentPlan.model_validate(
            {
                "title": "毁伤评估标准",
                "rationale": "按章节研究",
                "chapters": [_chapter(chapter_id="levels", ordinal=1, title="毁伤等级")],
            }
        )
        model = _ScriptedModel(
            route="supervisor",
            plan=plan,
            worker=[
                _tool_call(
                    "submit_chapter",
                    {"status": "insufficient", "gaps": ["缺少等级定义"]},
                    "worker-submit",
                ),
                _tool_call(
                    "submit_chapter",
                    {"status": "insufficient", "gaps": ["缺少等级定义"]},
                    "worker-followup-submit",
                ),
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
        self.assertIn("本章参考依据不足", run.answer.content)
        self.assertEqual(run.outcome, "incomplete")
        self.assertEqual(run.document_plan, plan)
        self.assertEqual(len(run.worker_packets), 1)
        self.assertEqual(run.worker_packets[0].task, "毁伤等级研究")
        self.assertEqual(run.worker_packets[0].status, "insufficient")

    def test_chapter_dependency_passes_foundation_contract_to_later_worker(self):
        plan = DocumentPlan.model_validate(
            {
                "title": "毁伤评估标准",
                "rationale": "先确定全局分级，再研究能力章节",
                "chapters": [
                    _chapter(
                        chapter_id="levels",
                        ordinal=1,
                        title="总体原则",
                        produces_contracts=["D-LEVELS"],
                    ),
                    _chapter(
                        chapter_id="movement",
                        ordinal=2,
                        title="运动能力",
                        depends_on=["levels"],
                        required_contracts=["D-LEVELS"],
                    ),
                ],
            }
        )

        def packet_args(chapter_id: str, title: str, contract: bool = False):
            return {
                "status": "sufficient",
                "prose": f"{title}正文",
                "rules": [
                    {
                        "basis": "synthesized",
                        "evidence_ids": ["ev-test"],
                        "contract_id": "D-LEVELS" if contract else None,
                    }
                ],
                "contracts": (
                    [
                        {
                            "contract_id": "D-LEVELS",
                            "type": "classification",
                            "canonical_terms": ["轻度", "中度", "重度"],
                        }
                    ]
                    if contract
                    else []
                ),
            }

        model = _ScriptedModel(
            route="supervisor",
            plan=plan,
            worker=[
                _tool_call(
                    "submit_chapter",
                    packet_args("levels", "总体原则", contract=True),
                    "levels-submit",
                ),
                _tool_call(
                    "submit_chapter",
                    packet_args("movement", "运动能力"),
                    "movement-submit",
                ),
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            runtime = AgentRuntime(
                model=model,
                workspace=_Workspace(),
                store=AgentRunStore(Path(tmp)),
            )
            run, _ = runtime.run("生成毁伤评估标准")

        self.assertEqual([item.chapter_id for item in run.worker_packets], ["levels", "movement"])
        self.assertEqual(run.worker_packets[0].contracts[0].contract_id, "D-LEVELS")
        self.assertEqual(run.worker_packets[0].contracts[0].canonical_terms, ["轻度", "中度", "重度"])
        self.assertNotIn("ev-test", run.answer.content)
        self.assertEqual(run.answer.evidence_ids, ["ev-test"])

    def test_terms_contract_requires_canonical_terms(self):
        chapter = DocumentPlan.model_validate(
            {
                "title": "标准",
                "rationale": "测试",
                "chapters": [_chapter(chapter_id="levels", ordinal=1, title="等级")],
            }
        ).chapters[0]
        packet = ResearchPacket.model_validate(
            {
                "task": chapter.objective,
                "chapter_id": chapter.chapter_id,
                "chapter_title": chapter.title,
                "status": "sufficient",
                "summary": "研究完成",
                "prose": "采用统一分级",
                "rules": [{"basis": "source", "evidence_ids": ["ev-test"]}],
                "contracts": [
                    {"contract_id": "D-LEVELS", "type": "terms", "canonical_terms": []}
                ],
            }
        )
        runtime = AgentRuntime(model=_ScriptedModel(route="fast"), workspace=_Workspace())

        with self.assertRaisesRegex(ValueError, "canonical_terms"):
            runtime._validate_packet(packet, chapter)

    def test_rule_references_unknown_contract_rejected(self):
        chapter = DocumentPlan.model_validate(
            {
                "title": "标准",
                "rationale": "测试",
                "chapters": [
                    _chapter(chapter_id="levels", ordinal=1, title="等级")
                ],
            }
        ).chapters[0]
        packet = ResearchPacket.model_validate(
            {
                "task": chapter.objective,
                "chapter_id": chapter.chapter_id,
                "chapter_title": chapter.title,
                "status": "sufficient",
                "summary": "研究完成",
                "prose": "采用统一分级",
                "rules": [
                    {
                        "basis": "source",
                        "evidence_ids": ["ev-test"],
                        "contract_id": "D-UNKNOWN",
                    }
                ],
            }
        )
        runtime = AgentRuntime(model=_ScriptedModel(route="fast"), workspace=_Workspace())

        with self.assertRaisesRegex(ValueError, "unknown contract"):
            runtime._validate_packet(packet, chapter)

    def test_contract_ids_unique_within_chapter(self):
        chapter = DocumentPlan.model_validate(
            {
                "title": "标准",
                "rationale": "测试",
                "chapters": [_chapter(chapter_id="levels", ordinal=1, title="等级")],
            }
        ).chapters[0]
        packet = ResearchPacket.model_validate(
            {
                "task": chapter.objective,
                "chapter_id": chapter.chapter_id,
                "chapter_title": chapter.title,
                "status": "sufficient",
                "summary": "研究完成",
                "prose": "采用统一分级",
                "rules": [{"basis": "source", "evidence_ids": ["ev-test"]}],
                "contracts": [
                    {"contract_id": "D-LEVELS", "type": "terms", "canonical_terms": ["A"]},
                    {"contract_id": "D-LEVELS", "type": "terms", "canonical_terms": ["B"]},
                ],
            }
        )
        runtime = AgentRuntime(model=_ScriptedModel(route="fast"), workspace=_Workspace())

        with self.assertRaisesRegex(ValueError, "unique within a chapter"):
            runtime._validate_packet(packet, chapter)

    def test_sufficient_requires_prose_and_rules(self):
        chapter = DocumentPlan.model_validate(
            {
                "title": "标准",
                "rationale": "测试",
                "chapters": [_chapter(chapter_id="scope", ordinal=1, title="范围")],
            }
        ).chapters[0]
        packet = ResearchPacket.model_validate(
            {
                "task": chapter.objective,
                "chapter_id": chapter.chapter_id,
                "chapter_title": chapter.title,
                "status": "sufficient",
                "summary": "研究完成",
                "prose": "",
                "rules": [],
            }
        )
        runtime = AgentRuntime(model=_ScriptedModel(route="fast"), workspace=_Workspace())

        with self.assertRaisesRegex(ValueError, "requires prose"):
            runtime._validate_packet(packet, chapter)

    def test_chapter_rejects_markdown_heading_in_prose(self):
        chapter = DocumentPlan.model_validate(
            {
                "title": "标准",
                "rationale": "测试",
                "chapters": [_chapter(chapter_id="data", ordinal=1, title="数据要求")],
            }
        ).chapters[0]
        packet = ResearchPacket.model_validate(
            {
                "task": chapter.objective,
                "chapter_id": chapter.chapter_id,
                "chapter_title": chapter.title,
                "status": "sufficient",
                "summary": "研究完成",
                "prose": "# 5.1 数据需求类型\n要求",
                "rules": [{"basis": "source", "evidence_ids": ["ev-test"]}],
            }
        )
        runtime = AgentRuntime(model=_ScriptedModel(route="fast"), workspace=_Workspace())

        with self.assertRaisesRegex(ValueError, "Markdown headings"):
            runtime._validate_packet(packet, chapter)

    def test_chapter_rejects_internal_ids_in_public_prose(self):
        chapter = DocumentPlan.model_validate(
            {
                "title": "标准",
                "rationale": "测试",
                "chapters": [_chapter(chapter_id="scope", ordinal=1, title="范围")],
            }
        ).chapters[0]
        packet = ResearchPacket.model_validate(
            {
                "task": chapter.objective,
                "chapter_id": chapter.chapter_id,
                "chapter_title": chapter.title,
                "status": "sufficient",
                "summary": "研究完成",
                "prose": "结论 C1",
                "rules": [{"basis": "source", "evidence_ids": ["ev-test"]}],
            }
        )
        runtime = AgentRuntime(model=_ScriptedModel(route="fast"), workspace=_Workspace())

        with self.assertRaisesRegex(ValueError, "internal evidence"):
            runtime._validate_packet(packet, chapter)

    def test_assembler_keeps_public_prose_separate_from_evidence_metadata(self):
        plan = DocumentPlan.model_validate(
            {
                "title": "标准",
                "rationale": "测试",
                "chapters": [_chapter(chapter_id="scope", ordinal=1, title="范围")],
            }
        )
        packet = ResearchPacket.model_validate(
            {
                "task": "规定范围",
                "chapter_id": "scope",
                "chapter_title": "范围",
                "status": "sufficient",
                "summary": "研究完成",
                "prose": "第一段结论。\n\n第二段结论。",
                "rules": [{"basis": "source", "evidence_ids": ["ev-test"]}],
            }
        )

        answer = AgentRuntime._assemble_answer(plan, [packet], [])

        self.assertIn("第一段结论。\n\n第二段结论。", answer.content)
        self.assertNotIn("ev-test", answer.content)
        self.assertEqual(answer.evidence_ids, ["ev-test"])

    def test_chapter_rejects_prose_over_character_budget(self):
        chapter = DocumentPlan.model_validate(
            {
                "title": "标准",
                "rationale": "测试",
                "chapters": [_chapter(chapter_id="scope", ordinal=1, title="范围")],
            }
        ).chapters[0]
        packet = ResearchPacket.model_validate(
            {
                "task": chapter.objective,
                "chapter_id": chapter.chapter_id,
                "chapter_title": chapter.title,
                "status": "sufficient",
                "summary": "研究完成",
                "prose": "超长正文内容非常多过多",
                "rules": [{"basis": "source", "evidence_ids": ["ev-test"]}],
            }
        )
        runtime = AgentRuntime(
            model=_ScriptedModel(route="fast"),
            workspace=_Workspace(),
            chapter_max_chars=10,
        )

        with self.assertRaisesRegex(ValueError, "character budget"):
            runtime._validate_packet(packet, chapter)

    def test_canonicalize_injects_identity_and_omits_evidence_ids(self):
        chapter = DocumentPlan.model_validate(
            {
                "title": "标准",
                "rationale": "测试",
                "chapters": [
                    _chapter(
                        chapter_id="scope",
                        ordinal=1,
                        title="范围",
                        depends_on=["foundation"],
                    ),
                    _chapter(chapter_id="foundation", ordinal=2, title="基础"),
                ],
            }
        ).chapters[0]

        # A worker emits only the research artifact: no identity fields and no
        # top-level evidence_ids. _canonicalize_packet_payload must inject the
        # identity from the plan and leave evidence_ids to _validate_packet,
        # which derives it from rules[].evidence_ids.
        payload = {
            "status": "sufficient",
            "prose": "适用范围",
            "rules": [{"basis": "source", "evidence_ids": ["ev-test"]}],
        }

        result = AgentRuntime._canonicalize_packet_payload(
            payload, chapter, EvidenceAliasRegistry()
        )

        self.assertEqual(result["task"], "范围研究")
        self.assertEqual(result["chapter_id"], "scope")
        self.assertEqual(result["chapter_title"], "范围")
        self.assertEqual(result["depends_on"], ["foundation"])
        self.assertNotIn("evidence_ids", result)

    def test_structural_consistency_flags_terminology_drift(self):
        plan = DocumentPlan.model_validate(
            {
                "title": "毁伤标准",
                "rationale": "测试",
                "chapters": [
                    _chapter(
                        chapter_id="levels",
                        ordinal=1,
                        title="等级",
                        produces_contracts=["D-LEVELS"],
                    ),
                    _chapter(
                        chapter_id="scope",
                        ordinal=2,
                        title="范围",
                        depends_on=["levels"],
                        required_contracts=["D-LEVELS"],
                    ),
                ],
            }
        )
        packets = [
            ResearchPacket.model_validate(
                {
                    "task": "等级",
                    "chapter_id": "levels",
                    "chapter_title": "等级",
                    "status": "sufficient",
                    "summary": "完成",
                    "prose": "采用K级判定",
                    "rules": [{"basis": "source", "evidence_ids": ["ev-test"]}],
                    "contracts": [
                        {"contract_id": "D-LEVELS", "type": "terms", "canonical_terms": ["K级", "M级"]}
                    ],
                }
            ),
            ResearchPacket.model_validate(
                {
                    "task": "范围",
                    "chapter_id": "scope",
                    "chapter_title": "范围",
                    "status": "sufficient",
                    "summary": "完成",
                    "prose": "该目标判定为Q级。",
                    "rules": [{"basis": "source", "evidence_ids": ["ev-test"]}],
                }
            ),
        ]

        issues = AgentRuntime._structural_consistency_issues(plan, packets)

        drift = [item for item in issues if item.issue_id.startswith("terminology-drift")]
        self.assertTrue(drift, f"expected a terminology-drift issue, got {issues}")
        self.assertIn("Q级", drift[0].description)

    def test_structural_consistency_flags_unmet_required_contract(self):
        plan = DocumentPlan.model_validate(
            {
                "title": "毁伤标准",
                "rationale": "测试",
                "chapters": [
                    _chapter(
                        chapter_id="levels",
                        ordinal=1,
                        title="等级",
                        produces_contracts=["D-LEVELS"],
                    ),
                    _chapter(
                        chapter_id="scope",
                        ordinal=2,
                        title="范围",
                        depends_on=["levels"],
                        required_contracts=["D-LEVELS"],
                    ),
                ],
            }
        )
        # The levels chapter is supposed to promulgate D-LEVELS but its packet
        # does not carry it, so the consumer's required contract goes unmet.
        packets = [
            ResearchPacket.model_validate(
                {
                    "task": "等级",
                    "chapter_id": "levels",
                    "chapter_title": "等级",
                    "status": "sufficient",
                    "summary": "完成",
                    "prose": "正文",
                    "rules": [{"basis": "source", "evidence_ids": ["ev-test"]}],
                }
            ),
            ResearchPacket.model_validate(
                {
                    "task": "范围",
                    "chapter_id": "scope",
                    "chapter_title": "范围",
                    "status": "sufficient",
                    "summary": "完成",
                    "prose": "正文",
                    "rules": [{"basis": "source", "evidence_ids": ["ev-test"]}],
                }
            ),
        ]

        issues = AgentRuntime._structural_consistency_issues(plan, packets)

        self.assertTrue(
            any(item.issue_id.startswith("unmet-contract") for item in issues)
        )


if __name__ == "__main__":
    unittest.main()