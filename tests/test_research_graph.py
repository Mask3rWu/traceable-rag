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
    def test_failed_packet_keeps_last_tool_validation_error(self):
        chapter = DocumentPlan.model_validate(
            {
                "title": "标准",
                "rationale": "测试",
                "chapters": [
                    {
                        "chapter_id": "scope",
                        "ordinal": 1,
                        "title": "范围",
                        "objective": "形成范围",
                        "research_questions": ["范围是什么"],
                        "acceptance_criteria": ["形成范围"],
                    }
                ],
            }
        ).chapters[0]
        packet = AgentRuntime._failed_packet(
            chapter,
            {
                "task": chapter.objective,
                "messages": [
                    HumanMessage(content="研究"),
                    ToolMessage(
                        content="normative decision must state validation requirements",
                        tool_call_id="submit-1",
                        name="submit_chapter",
                        status="error",
                    ),
                ],
            },
            False,
        )

        self.assertEqual(packet.status, "failed")
        self.assertIn("validation requirements", packet.diagnostics[0])

    def test_document_plan_rejects_dependency_cycle(self):
        with self.assertRaisesRegex(ValueError, "cycle"):
            DocumentPlan.model_validate(
                {
                    "title": "循环计划",
                    "rationale": "测试",
                    "chapters": [
                        {
                            "chapter_id": "a",
                            "ordinal": 1,
                            "title": "A",
                            "objective": "A",
                            "research_questions": ["A"],
                            "depends_on": ["b"],
                            "acceptance_criteria": ["A"],
                        },
                        {
                            "chapter_id": "b",
                            "ordinal": 2,
                            "title": "B",
                            "objective": "B",
                            "research_questions": ["B"],
                            "depends_on": ["a"],
                            "acceptance_criteria": ["B"],
                        },
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
                "chapters": [
                    {
                        "chapter_id": "levels",
                        "ordinal": 1,
                        "title": "一、毁伤等级",
                        "objective": "研究毁伤等级",
                        "research_questions": ["等级如何划分"],
                        "acceptance_criteria": ["识别等级体系"],
                    }
                ],
            }
        )
        model = _ScriptedModel(
            route="supervisor",
            plan=plan,
            worker=[
                _tool_call(
                    "submit_chapter",
                    {
                        "task": "研究毁伤等级",
                        "chapter_id": "levels",
                        "chapter_title": "一、毁伤等级",
                        "status": "insufficient",
                        "summary": "未找到足够证据",
                        "claims": [],
                        "conflicts": [],
                        "gaps": ["缺少等级定义"],
                        "evidence_ids": [],
                    },
                    "worker-submit",
                ),
                _tool_call(
                    "submit_chapter",
                    {
                        "task": "研究毁伤等级",
                        "chapter_id": "levels",
                        "chapter_title": "一、毁伤等级",
                        "status": "insufficient",
                        "summary": "补充检索后仍无足够证据",
                        "claims": [],
                        "conflicts": [],
                        "gaps": ["缺少等级定义"],
                        "evidence_ids": [],
                    },
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
        self.assertEqual(run.worker_packets[0].task, "研究毁伤等级")
        self.assertEqual(run.worker_packets[0].status, "insufficient")

    def test_chapter_dependency_passes_foundation_decision_to_later_worker(self):
        plan = DocumentPlan.model_validate(
            {
                "title": "毁伤评估标准",
                "rationale": "先确定全局分级，再研究能力章节",
                "chapters": [
                    {
                        "chapter_id": "levels",
                        "ordinal": 1,
                        "title": "一、总体原则",
                        "objective": "确定统一分级",
                        "research_questions": ["采用几级分类"],
                        "produces_decisions": ["D-LEVELS"],
                        "acceptance_criteria": ["形成有证据的分级决策"],
                    },
                    {
                        "chapter_id": "movement",
                        "ordinal": 2,
                        "title": "二、运动能力",
                        "objective": "形成运动能力标准",
                        "research_questions": ["如何映射统一分级"],
                        "depends_on": ["levels"],
                        "required_decisions": ["D-LEVELS"],
                        "acceptance_criteria": ["沿用统一分级"],
                    },
                ],
            }
        )

        def packet_args(chapter_id: str, title: str, decision: bool = False):
            claim_id = f"C-{chapter_id}"
            block_id = f"B-{chapter_id}"
            decision_id = "D-LEVELS" if decision else None
            return {
                "task": title,
                "chapter_id": chapter_id,
                "chapter_title": title,
                "depends_on": [] if decision else ["levels"],
                "status": "sufficient",
                "summary": f"{title}研究完成",
                "content_blocks": [
                    {
                        "block_id": block_id,
                        "markdown": f"{title}正文",
                        "claim_ids": [claim_id],
                        "decision_ids": [decision_id] if decision_id else [],
                        "evidence_ids": ["ev-test"],
                    }
                ],
                "claims": [
                    {
                        "claim_id": claim_id,
                        "text": f"{title}结论",
                        "conclusion_type": "synthesized",
                        "citations": [
                            {"evidence_id": "ev-test", "quote": "原文"}
                        ],
                    }
                ],
                "decisions": (
                    [
                        {
                            "decision_id": decision_id,
                            "statement": "采用四级毁伤等级",
                            "decision_type": "synthesized",
                            "rationale": "来源中的状态可映射为四个可操作等级",
                            "claim_ids": [claim_id],
                            "evidence_ids": ["ev-test"],
                            "confidence": "medium",
                            "applies_to_chapters": ["movement"],
                        }
                    ]
                    if decision_id
                    else []
                ),
                "evidence_ids": ["ev-test"],
            }

        model = _ScriptedModel(
            route="supervisor",
            plan=plan,
            worker=[
                _tool_call(
                    "submit_chapter",
                    packet_args("levels", "一、总体原则", decision=True),
                    "levels-submit",
                ),
                _tool_call(
                    "submit_chapter",
                    packet_args("movement", "二、运动能力"),
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
        self.assertEqual(run.worker_packets[0].decisions[0].decision_id, "D-LEVELS")
        self.assertNotIn("ev-test", run.answer.content)
        self.assertEqual(run.answer.evidence_ids, ["ev-test"])

    def test_normative_decision_requires_auditable_design_metadata(self):
        chapter = DocumentPlan.model_validate(
            {
                "title": "标准",
                "rationale": "基于参考资料生成新规则",
                "deliverable_mode": "normative_synthesis",
                "chapters": [
                    {
                        "chapter_id": "levels",
                        "ordinal": 1,
                        "title": "毁伤等级",
                        "objective": "拟定等级",
                        "research_questions": ["如何分级"],
                        "produces_decisions": ["D-LEVELS"],
                        "acceptance_criteria": ["形成可验证分级"],
                    }
                ],
            }
        ).chapters[0]
        packet = ResearchPacket.model_validate(
            {
                "task": chapter.objective,
                "chapter_id": chapter.chapter_id,
                "chapter_title": chapter.title,
                "status": "sufficient",
                "summary": "形成规范性方案",
                "content_blocks": [
                    {
                        "block_id": "B-levels",
                        "markdown": "建议采用四级分类",
                        "claim_ids": ["C-levels"],
                        "decision_ids": ["D-LEVELS"],
                        "evidence_ids": ["ev-test"],
                    }
                ],
                "claims": [
                    {
                        "claim_id": "C-levels",
                        "text": "参考资料存在按功能状态分级的做法",
                        "conclusion_type": "synthesized",
                        "citations": [{"evidence_id": "ev-test", "quote": "原文"}],
                    }
                ],
                "decisions": [
                    {
                        "decision_id": "D-LEVELS",
                        "statement": "本标准建议采用四级分类",
                        "decision_type": "normative",
                        "rationale": "将参考资料中的功能状态迁移为统一等级",
                        "claim_ids": ["C-levels"],
                        "evidence_ids": ["ev-test"],
                        "confidence": "low",
                    }
                ],
            }
        )
        runtime = AgentRuntime(model=_ScriptedModel(route="fast"), workspace=_Workspace())

        with self.assertRaisesRegex(ValueError, "assumptions"):
            runtime._validate_packet(packet, chapter)

    def test_chapter_rejects_content_block_heading(self):
        chapter = DocumentPlan.model_validate(
            {
                "title": "标准",
                "rationale": "按章节生成",
                "chapters": [
                    {
                        "chapter_id": "data",
                        "ordinal": 1,
                        "title": "数据要求",
                        "objective": "规定数据要求",
                        "research_questions": ["需要哪些数据"],
                        "acceptance_criteria": ["形成数据要求"],
                    }
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
                "content_blocks": [
                    {
                        "block_id": "B-data",
                        "heading": "5.1 数据需求类型",
                        "markdown": "数据要求",
                        "claim_ids": ["C-data"],
                        "evidence_ids": ["ev-test"],
                    }
                ],
                "claims": [
                    {
                        "claim_id": "C-data",
                        "text": "资料规定了数据要求",
                        "conclusion_type": "direct",
                        "citations": [{"evidence_id": "ev-test", "quote": "原文"}],
                    }
                ],
            }
        )
        runtime = AgentRuntime(model=_ScriptedModel(route="fast"), workspace=_Workspace())

        with self.assertRaisesRegex(ValueError, "must not define a heading"):
            runtime._validate_packet(packet, chapter)

    def test_chapter_rejects_internal_ids_in_public_prose(self):
        chapter = DocumentPlan.model_validate(
            {
                "title": "标准",
                "rationale": "按章节生成",
                "chapters": [
                    {
                        "chapter_id": "scope",
                        "ordinal": 1,
                        "title": "范围",
                        "objective": "规定范围",
                        "research_questions": ["适用范围是什么"],
                        "acceptance_criteria": ["形成范围"],
                    }
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
                "content_blocks": [
                    {
                        "block_id": "B-scope",
                        "markdown": "结论 C1",
                        "claim_ids": ["C-scope"],
                        "evidence_ids": ["ev-test"],
                    }
                ],
                "claims": [
                    {
                        "claim_id": "C-scope",
                        "text": "适用范围已定义",
                        "conclusion_type": "direct",
                        "citations": [{"evidence_id": "ev-test", "quote": "原文"}],
                    }
                ],
            }
        )
        runtime = AgentRuntime(model=_ScriptedModel(route="fast"), workspace=_Workspace())

        with self.assertRaisesRegex(ValueError, "internal evidence"):
            runtime._validate_packet(packet, chapter)

    def test_assembler_keeps_public_prose_separate_from_evidence_metadata(self):
        plan = DocumentPlan.model_validate(
            {
                "title": "标准",
                "rationale": "按章节生成",
                "chapters": [
                    {
                        "chapter_id": "scope",
                        "ordinal": 1,
                        "title": "范围",
                        "objective": "规定范围",
                        "research_questions": ["适用范围是什么"],
                        "acceptance_criteria": ["形成范围"],
                    }
                ],
            }
        )
        packet = ResearchPacket.model_validate(
            {
                "task": "规定范围",
                "chapter_id": "scope",
                "chapter_title": "范围",
                "status": "sufficient",
                "summary": "研究完成",
                "content_blocks": [
                    {
                        "block_id": "B-scope",
                        "markdown": "第一段结论。\n\n第二段结论。",
                        "claim_ids": ["C-scope"],
                        "evidence_ids": ["ev-test"],
                    }
                ],
                "claims": [
                    {
                        "claim_id": "C-scope",
                        "text": "适用范围已定义",
                        "conclusion_type": "direct",
                        "citations": [{"evidence_id": "ev-test", "quote": "原文"}],
                    }
                ],
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
                "rationale": "按章节生成",
                "chapters": [
                    {
                        "chapter_id": "scope",
                        "ordinal": 1,
                        "title": "范围",
                        "objective": "规定范围",
                        "research_questions": ["适用范围是什么"],
                        "acceptance_criteria": ["形成范围"],
                    }
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
                "content_blocks": [
                    {
                        "block_id": "B-scope",
                        "markdown": "超长正文内容非常多过多",
                        "claim_ids": ["C-scope"],
                        "evidence_ids": ["ev-test"],
                    }
                ],
                "claims": [
                    {
                        "claim_id": "C-scope",
                        "text": "资料支持适用范围",
                        "conclusion_type": "direct",
                        "citations": [{"evidence_id": "ev-test", "quote": "原文"}],
                    }
                ],
            }
        )
        runtime = AgentRuntime(
            model=_ScriptedModel(route="fast"),
            workspace=_Workspace(),
            chapter_max_chars=10,
        )

        with self.assertRaisesRegex(ValueError, "character budget"):
            runtime._validate_packet(packet, chapter)

    def test_chapter_rejects_evidence_without_claim_or_decision_reason(self):
        chapter = DocumentPlan.model_validate(
            {
                "title": "标准",
                "rationale": "按章节生成",
                "chapters": [
                    {
                        "chapter_id": "scope",
                        "ordinal": 1,
                        "title": "范围",
                        "objective": "规定范围",
                        "research_questions": ["适用范围是什么"],
                        "acceptance_criteria": ["形成范围"],
                    }
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
                "content_blocks": [
                    {
                        "block_id": "B-scope",
                        "markdown": "适用范围",
                        "claim_ids": ["C-scope"],
                        "evidence_ids": ["ev-test", "ev-unexplained"],
                    }
                ],
                "claims": [
                    {
                        "claim_id": "C-scope",
                        "text": "资料支持适用范围",
                        "conclusion_type": "direct",
                        "citations": [{"evidence_id": "ev-test", "quote": "原文"}],
                    }
                ],
            }
        )
        runtime = AgentRuntime(model=_ScriptedModel(route="fast"), workspace=_Workspace())

        with self.assertRaisesRegex(ValueError, "without a Claim or Decision reason"):
            runtime._validate_packet(packet, chapter)


if __name__ == "__main__":
    unittest.main()
