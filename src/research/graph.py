"""Routed LangGraph runtime with chapter-planned ReAct research workers."""
from __future__ import annotations

import json
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from src.research.agent_models import (
    AgentAnswer,
    AgentRun,
    ChapterPlan,
    ConsistencyIssue,
    ConsistencyReport,
    DocumentPlan,
    ResearchPacket,
    RouteDecision,
)
from src.research.agent_store import AgentRunStore
from src.research.tools import EvidenceWorkspace


FAST_PROMPT = """You are a traceable knowledge-base question-answering agent.
Rewrite the user's need into focused searches. Use search_knowledge, inspect previews, and
read only the evidence needed. You may search again when coverage is weak. Answer only from
retrieved evidence. Cite evidence IDs inline and finish by calling submit_answer."""

PLANNER_PROMPT = """You are the planning component of a research supervisor. Create a
chapter-level plan for the requested structured deliverable. Each chapter must have bounded
research questions and acceptance criteria. Put shared terminology, scope, classification
frameworks, and other global decisions in foundational chapters. Express execution order with
depends_on. A chapter that creates a reusable decision declares its stable ID in
produces_decisions; every consumer declares that ID in required_decisions and depends on its
producer. Independent chapters may run in parallel. Do not make factual claims or invent
evidence in the plan. Every chapter must first summarize, compare, and verify the relevant
sources. Choose evidence_summary when that evidence synthesis is the final deliverable. Choose
normative_synthesis when the request additionally asks for a new standard, taxonomy, threshold
system, or operating rule; this mode includes the evidence synthesis and adds a separate design
layer based on it. Use concise
stable ASCII identifiers and preserve the user's language.
Return only a JSON object matching the supplied schema."""

WORKER_PROMPT = """You are a chapter research worker handling one bounded chapter. Work
through every research question using iterative search_knowledge and read_evidence calls. Start
by summarizing, comparing, and verifying relevant source conclusions; this evidence synthesis is
required in both deliverable modes.
Use small focused queries, shortlist the most relevant evidence, inspect exact source excerpts,
and stop broad searching once the questions have reasonable coverage. Respect all upstream
decisions. Every factual claim needs one or more evidence IDs; the system fills exact source
quotes. Every
evidence item used in chapter prose must have an EvidenceContribution explaining why it is
relevant and the concise auditable inference it supports. This inference is a justification,
not hidden chain-of-thought. Put prose in ContentBlocks and link every used evidence ID to the
block, claim, or decision it supports. Distinguish direct, synthesized, normative, and
hypothesis conclusions. In normative_synthesis mode, the requested standard is allowed to be
new, but only after evidence synthesis: use the verified source comparison as design input and
keep source conclusions separate from proposed rules. Create explicit normative rules instead of
requiring a source that already contains the finished standard. Mark designed rules and
thresholds as normative, explain the transfer rationale, list assumptions and alternatives,
state validation requirements, and use lower confidence where empirical calibration is absent.
Do not mark a chapter insufficient merely because the exact requested standard is absent.
Never present a proposed rule as a source fact. Finish by calling submit_chapter with a compact
structured chapter artifact."""

ROUTER_PROMPT = """Classify the execution mode for a knowledge-base request. Choose fast
for a focused question answerable with a few searches. Choose supervisor for requests requiring
multi-part research, cross-source comparison, an assessment standard, report, taxonomy, rules,
or another substantial structured deliverable. Return a JSON object matching this schema:
{"mode": "fast|supervisor", "reason": "short operational reason"}."""

REVIEW_PROMPT = """You are a consistency reviewer for a structured research deliverable.
Review only the supplied chapter summaries, verified claims, public decision rationales, and
open conflicts. Identify contradictory terminology, classifications, thresholds, or rules and
missing cross-chapter alignment. Do not add facts or evidence. Report only actionable issues
using known chapter IDs. Return a JSON object matching the supplied schema."""


class ReactState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    steps: int
    result: dict[str, Any]
    task: str


class RootState(TypedDict, total=False):
    request: str
    route: dict[str, Any]
    plan: dict[str, Any]
    packets: list[dict[str, Any]]
    issues: list[dict[str, Any]]
    answer: dict[str, Any]


class AgentRuntime:
    def __init__(
        self,
        *,
        model: BaseChatModel,
        workspace: EvidenceWorkspace,
        store: AgentRunStore | None = None,
        max_steps: int = 12,
        fast_max_steps: int | None = None,
        worker_max_steps: int | None = None,
        supervisor_max_steps: int | None = None,
        max_workers: int = 4,
        max_subtasks: int = 8,
    ) -> None:
        if min(max_steps, max_workers, max_subtasks) <= 0:
            raise ValueError("Agent budgets must be greater than zero")
        self.model = model
        self.workspace = workspace
        self.store = store or AgentRunStore()
        self.max_steps = max_steps
        self.fast_max_steps = fast_max_steps or min(max_steps, 8)
        self.worker_max_steps = worker_max_steps or max(max_steps, 30)
        # Retained as the coordinator/root graph compatibility budget.
        self.supervisor_max_steps = supervisor_max_steps or max_steps
        self.max_workers = max_workers
        self.max_subtasks = max_subtasks
        self._packets: list[ResearchPacket] = []
        self._lock = threading.RLock()
        self._run_lock = threading.Lock()
        self._fast_graph = self._build_fast_graph()
        self.graph = self._build_root_graph()

    @property
    def packets(self) -> list[ResearchPacket]:
        with self._lock:
            return list(self._packets)

    @staticmethod
    def _submit_answer_tool(
        validator: Callable[[AgentAnswer], None] | None = None,
    ) -> BaseTool:
        @tool(args_schema=AgentAnswer)
        def submit_answer(
            content: str, evidence_ids: list[str], limitations: list[str]
        ) -> str:
            """Submit the final answer and stop this agent."""

            answer = AgentAnswer(
                content=content, evidence_ids=evidence_ids, limitations=limitations
            )
            if validator is not None:
                validator(answer)
            return "submitted"

        return submit_answer

    def _submit_chapter_tool(self, chapter: ChapterPlan) -> BaseTool:
        @tool(args_schema=ResearchPacket)
        def submit_chapter(**kwargs: Any) -> str:
            """Submit a grounded chapter artifact and stop this worker."""

            packet = ResearchPacket.model_validate(kwargs)
            self._validate_packet(packet, chapter)
            return "submitted"

        return submit_chapter

    def _build_react_graph(
        self,
        *,
        prompt: str,
        tools: Sequence[BaseTool],
        submit_name: str,
        result_model: type[BaseModel],
        exhausted_result: Callable[[ReactState, bool], BaseModel],
        graph_name: str,
        step_limit: int,
    ):
        bound_model = self.model.bind_tools(list(tools))
        tool_node = ToolNode(tools, handle_tool_errors=True)

        def call_model(state: ReactState, config: RunnableConfig) -> dict:
            response = bound_model.invoke(
                [SystemMessage(content=prompt), *state["messages"]], config=config
            )
            return {"messages": [response], "steps": state.get("steps", 0) + 1}

        def next_step(state: ReactState) -> str:
            message = state["messages"][-1]
            calls = message.tool_calls if isinstance(message, AIMessage) else []
            if calls:
                return "tools"
            if state.get("steps", 0) >= step_limit:
                return "exhausted"
            return "exhausted"

        def after_tools(state: ReactState) -> str:
            message = state["messages"][-1]
            if (
                isinstance(message, ToolMessage)
                and message.name == submit_name
                and getattr(message, "status", "success") != "error"
            ):
                return "submit"
            if state.get("steps", 0) >= step_limit:
                return "exhausted"
            return "agent"

        def submit(state: ReactState) -> dict:
            message = next(
                item
                for item in reversed(state["messages"])
                if isinstance(item, AIMessage)
                and any(call["name"] == submit_name for call in item.tool_calls)
            )
            call = next(
                item for item in message.tool_calls if item["name"] == submit_name
            )
            result = result_model.model_validate(call["args"])
            return {"result": result.model_dump(mode="json")}

        def exhausted(state: ReactState) -> dict:
            result = exhausted_result(state, state.get("steps", 0) >= step_limit)
            return {"result": result.model_dump(mode="json")}

        builder = StateGraph(ReactState)
        builder.add_node("agent", call_model)
        builder.add_node("tools", tool_node)
        builder.add_node("submit", submit)
        builder.add_node("exhausted", exhausted)
        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent",
            next_step,
            {"tools": "tools", "submit": "submit", "exhausted": "exhausted"},
        )
        builder.add_conditional_edges(
            "tools",
            after_tools,
            {"agent": "agent", "submit": "submit", "exhausted": "exhausted"},
        )
        builder.add_edge("submit", END)
        builder.add_edge("exhausted", END)
        return builder.compile(name=graph_name)

    def _build_fast_graph(self):
        tools = self.workspace.make_retrieval_tools()
        tools.append(self._submit_answer_tool(self._validate_fast_answer))
        return self._build_react_graph(
            prompt=FAST_PROMPT,
            tools=tools,
            submit_name="submit_answer",
            result_model=AgentAnswer,
            exhausted_result=lambda _, budget_exhausted: AgentAnswer(
                content="未能在执行预算内形成可靠答案。",
                limitations=[
                    "Agent step budget exhausted"
                    if budget_exhausted
                    else "Agent stopped without submitting a grounded answer"
                ],
            ),
            graph_name="fast-react-agent",
            step_limit=self.fast_max_steps,
        )

    def _build_chapter_graph(self, chapter: ChapterPlan):
        tools = [
            *self.workspace.make_retrieval_tools(),
            self._submit_chapter_tool(chapter),
        ]

        return self._build_react_graph(
            prompt=WORKER_PROMPT,
            tools=tools,
            submit_name="submit_chapter",
            result_model=ResearchPacket,
            exhausted_result=lambda state, budget_exhausted: self._failed_packet(
                chapter, state, budget_exhausted
            ),
            graph_name="chapter-research-worker",
            step_limit=self.worker_max_steps,
        )

    @staticmethod
    def _failed_packet(
        chapter: ChapterPlan, state: ReactState, budget_exhausted: bool
    ) -> ResearchPacket:
        diagnostics: list[str] = []
        for message in reversed(state.get("messages", [])):
            if isinstance(message, ToolMessage) and getattr(message, "status", None) == "error":
                diagnostics.append(str(message.content))
                break
        if budget_exhausted:
            diagnostics.append("Agent step budget exhausted")
        elif not diagnostics:
            diagnostics.append("Worker stopped without a valid submit_chapter call")
        return ResearchPacket(
            task=state.get("task", chapter.objective),
            chapter_id=chapter.chapter_id,
            chapter_title=chapter.title,
            depends_on=chapter.depends_on,
            status="failed",
            summary="The chapter worker failed to submit a valid artifact.",
            diagnostics=list(dict.fromkeys(diagnostics)),
        )

    def _validate_fast_answer(self, answer: AgentAnswer) -> None:
        if self.workspace.search_count <= 0:
            raise ValueError("Search the knowledge base before submitting an answer")
        if not answer.evidence_ids:
            raise ValueError("A grounded answer requires at least one evidence ID")
        self.workspace.validate_evidence_ids(answer.evidence_ids)

    def _can_submit_fast_answer(self, args: dict[str, Any]) -> bool:
        try:
            self._validate_fast_answer(AgentAnswer.model_validate(args))
        except ValueError:
            return False
        return True

    def _validate_plan(self, plan: DocumentPlan) -> None:
        if len(plan.chapters) > self.max_subtasks:
            raise ValueError(
                f"Document plan exceeds chapter budget ({self.max_subtasks})"
            )

    def _validate_packet(self, packet: ResearchPacket, chapter: ChapterPlan) -> None:
        if packet.chapter_id != chapter.chapter_id:
            raise ValueError("Chapter artifact has the wrong chapter ID")
        if packet.chapter_title != chapter.title:
            raise ValueError("Chapter artifact has the wrong chapter title")

        claim_ids = {item.claim_id for item in packet.claims}
        decision_ids = {item.decision_id for item in packet.decisions}
        block_ids = {item.block_id for item in packet.content_blocks}
        if len(claim_ids) != len(packet.claims):
            raise ValueError("Claim IDs must be unique within a chapter")
        if len(decision_ids) != len(packet.decisions):
            raise ValueError("Decision IDs must be unique within a chapter")
        if len(block_ids) != len(packet.content_blocks):
            raise ValueError("Content block IDs must be unique within a chapter")

        cited = {
            citation.evidence_id
            for claim in packet.claims
            for citation in claim.citations
        }
        used = set(cited)
        used.update(
            evidence_id
            for decision in packet.decisions
            for evidence_id in decision.evidence_ids
        )
        used.update(
            evidence_id
            for block in packet.content_blocks
            for evidence_id in block.evidence_ids
        )
        self.workspace.validate_evidence_ids(used)
        evidence_by_id = self.workspace.evidence_by_id()
        packet.claims = [
            self.workspace.verifier.verify_claim(claim, evidence_by_id)
            for claim in packet.claims
        ]

        for block in packet.content_blocks:
            if set(block.claim_ids) - claim_ids:
                raise ValueError("Content block references an unknown claim")
            if set(block.decision_ids) - decision_ids:
                raise ValueError("Content block references an unknown decision")
            if not block.claim_ids and not block.decision_ids:
                raise ValueError("Every content block must expose its claim or decision")

        for decision in packet.decisions:
            if set(decision.claim_ids) - claim_ids:
                raise ValueError("Decision references an unknown claim")
            if decision.decision_type == "normative":
                if not decision.assumptions:
                    raise ValueError("A normative decision must state its assumptions")
                if not decision.alternatives:
                    raise ValueError("A normative decision must record alternatives")
                if not decision.validation_requirements:
                    raise ValueError(
                        "A normative decision must state validation requirements"
                    )

        if packet.status == "sufficient":
            if not packet.content_blocks or not packet.claims or not used:
                raise ValueError(
                    "A sufficient chapter requires prose, verified claims, and used evidence"
                )
            missing_decisions = set(chapter.produces_decisions) - decision_ids
            if missing_decisions:
                raise ValueError(
                    f"Chapter did not produce required decisions: {sorted(missing_decisions)}"
                )
        elif packet.content_blocks:
            raise ValueError("A non-sufficient chapter cannot submit final prose blocks")
        packet.evidence_ids = sorted(used)

    @staticmethod
    def _ancestor_packets(
        chapter: ChapterPlan, completed: dict[str, ResearchPacket]
    ) -> list[ResearchPacket]:
        ordered: list[ResearchPacket] = []
        seen: set[str] = set()

        def collect(chapter_id: str) -> None:
            if chapter_id in seen:
                return
            packet = completed[chapter_id]
            for dependency in packet.depends_on:
                collect(dependency)
            seen.add(chapter_id)
            ordered.append(packet)

        for dependency in chapter.depends_on:
            collect(dependency)
        return ordered

    @classmethod
    def _upstream_context(
        cls, chapter: ChapterPlan, completed: dict[str, ResearchPacket]
    ) -> dict[str, Any]:
        packets = cls._ancestor_packets(chapter, completed)
        return {
            "chapters": [
                {
                    "chapter_id": item.chapter_id,
                    "summary": item.summary,
                    "status": item.status,
                }
                for item in packets
            ],
            "decisions": [
                decision.model_dump(mode="json")
                for item in packets
                for decision in item.decisions
            ],
            "claims": [
                {
                    "claim_id": claim.claim_id,
                    "text": claim.text,
                    "conclusion_type": claim.conclusion_type,
                    "evidence_ids": [citation.evidence_id for citation in claim.citations],
                }
                for item in packets
                for claim in item.claims
            ],
        }

    def _run_chapter(
        self,
        plan: DocumentPlan,
        chapter: ChapterPlan,
        completed: dict[str, ResearchPacket],
        config: RunnableConfig,
        previous_attempt: ResearchPacket | None = None,
    ) -> ResearchPacket:
        request = {
            "document_title": plan.title,
            "deliverable_mode": plan.deliverable_mode,
            "chapter": chapter.model_dump(mode="json"),
            "upstream": self._upstream_context(chapter, completed),
            "previous_attempt": (
                {
                    "summary": previous_attempt.summary,
                    "gaps": previous_attempt.gaps,
                    "diagnostics": previous_attempt.diagnostics,
                    "evidence_ids": previous_attempt.evidence_ids,
                }
                if previous_attempt is not None
                else None
            ),
            "instructions": (
                "Return only this chapter. Use the declared chapter_id and chapter_title. "
                "Every Evidence ID appearing in prose must have a contribution record. "
                "When previous_attempt reports evidence gaps, focus new searches on them. "
                "When it reports diagnostics, correct the structured submission before doing "
                "more broad retrieval."
            ),
        }
        graph = self._build_chapter_graph(chapter)
        metadata = dict(config.get("metadata") or {})
        metadata.update({"chapter_id": chapter.chapter_id, "chapter_title": chapter.title})
        state = graph.invoke(
            {
                "messages": [
                    HumanMessage(content=json.dumps(request, ensure_ascii=False))
                ],
                "steps": 0,
                "task": chapter.objective,
            },
            {
                **config,
                "metadata": metadata,
                "run_name": f"chapter-worker:{chapter.chapter_id}",
                "recursion_limit": self.worker_max_steps * 2 + 4,
            },
        )
        packet = ResearchPacket.model_validate(state["result"])
        self._validate_packet(packet, chapter)
        return packet

    def _run_chapter_with_followup(
        self,
        plan: DocumentPlan,
        chapter: ChapterPlan,
        completed: dict[str, ResearchPacket],
        config: RunnableConfig,
    ) -> ResearchPacket:
        first = self._run_chapter(plan, chapter, completed, config)
        if first.status == "sufficient":
            return first
        if first.status == "insufficient" and not first.gaps:
            return first
        return self._run_chapter(
            plan, chapter, completed, config, previous_attempt=first
        )

    @staticmethod
    def _blocked_packet(chapter: ChapterPlan, gaps: list[str]) -> ResearchPacket:
        return ResearchPacket(
            task=chapter.objective,
            chapter_id=chapter.chapter_id,
            chapter_title=chapter.title,
            depends_on=chapter.depends_on,
            status="blocked",
            summary="Chapter research was blocked by unmet upstream dependencies.",
            gaps=gaps,
        )

    def _execute_plan(
        self, plan: DocumentPlan, config: RunnableConfig
    ) -> list[ResearchPacket]:
        remaining = {item.chapter_id: item for item in plan.chapters}
        completed: dict[str, ResearchPacket] = {}
        while remaining:
            ready = sorted(
                (
                    chapter
                    for chapter in remaining.values()
                    if set(chapter.depends_on) <= set(completed)
                ),
                key=lambda item: item.ordinal,
            )
            if not ready:
                raise RuntimeError("No executable chapter remains in the dependency graph")

            runnable: list[ChapterPlan] = []
            for chapter in ready:
                ancestors = self._ancestor_packets(chapter, completed)
                failed_dependencies = [
                    item.chapter_id
                    for item in ancestors
                    if item.status != "sufficient"
                ]
                available_decisions = {
                    decision.decision_id
                    for item in ancestors
                    for decision in item.decisions
                }
                missing_decisions = sorted(
                    set(chapter.required_decisions) - available_decisions
                )
                gaps = [
                    *(f"Upstream chapter is not sufficient: {item}" for item in failed_dependencies),
                    *(
                        f"Required decision is unavailable: {item}"
                        for item in missing_decisions
                    ),
                ]
                if gaps:
                    completed[chapter.chapter_id] = self._blocked_packet(chapter, gaps)
                else:
                    runnable.append(chapter)

            if runnable:
                snapshot = dict(completed)
                with ThreadPoolExecutor(
                    max_workers=min(self.max_workers, len(runnable)),
                    thread_name_prefix="chapter-worker",
                ) as executor:
                    futures = {
                        executor.submit(
                            self._run_chapter_with_followup,
                            plan,
                            chapter,
                            snapshot,
                            config,
                        ): chapter
                        for chapter in runnable
                    }
                    wave_results: dict[str, ResearchPacket] = {}
                    for future in as_completed(futures):
                        chapter = futures[future]
                        wave_results[chapter.chapter_id] = future.result()
                completed.update(wave_results)

            for chapter in ready:
                remaining.pop(chapter.chapter_id)

        ordered = [
            completed[item.chapter_id]
            for item in sorted(plan.chapters, key=lambda chapter: chapter.ordinal)
        ]
        with self._lock:
            self._packets = ordered
        return ordered

    @staticmethod
    def _structural_consistency_issues(
        plan: DocumentPlan, packets: list[ResearchPacket]
    ) -> list[ConsistencyIssue]:
        issues: list[ConsistencyIssue] = []
        by_chapter = {item.chapter_id: item for item in packets}
        decision_statements: dict[str, str] = {}
        for packet in packets:
            if packet.status != "sufficient":
                labels = {
                    "insufficient": "证据不足",
                    "failed": "执行失败",
                    "blocked": "被上游依赖阻塞",
                }
                issues.append(
                    ConsistencyIssue(
                        issue_id=f"chapter-insufficient:{packet.chapter_id}",
                        severity="error",
                        chapter_ids=[packet.chapter_id] if packet.chapter_id else [],
                        description=(
                            f"章节“{packet.chapter_title}”"
                            f"{labels.get(packet.status, '未完成')}。"
                        ),
                        recommendation=(
                            "修复章节执行或结构化提交错误后重试。"
                            if packet.status == "failed"
                            else "先完成上游章节和所需决策。"
                            if packet.status == "blocked"
                            else "补充可迁移依据，或明确标注规范性综合及验证要求。"
                        ),
                    )
                )
            for decision in packet.decisions:
                previous = decision_statements.get(decision.decision_id)
                if previous is not None and previous != decision.statement:
                    issues.append(
                        ConsistencyIssue(
                            issue_id=f"decision-conflict:{decision.decision_id}",
                            severity="error",
                            chapter_ids=[packet.chapter_id] if packet.chapter_id else [],
                            description=f"决策 {decision.decision_id} 在章节间表述不一致。",
                            recommendation="由决策所属基础章节统一该决策。",
                        )
                    )
                decision_statements[decision.decision_id] = decision.statement
            for conflict in packet.conflicts:
                if conflict.status == "open":
                    issues.append(
                        ConsistencyIssue(
                            issue_id=f"open-conflict:{conflict.conflict_id}",
                            severity="warning",
                            chapter_ids=[packet.chapter_id] if packet.chapter_id else [],
                            description=conflict.description,
                            recommendation="在定稿前审查未解决的来源冲突。",
                        )
                    )

        for chapter in plan.chapters:
            packet = by_chapter.get(chapter.chapter_id)
            if packet is None:
                continue
            produced = {item.decision_id for item in packet.decisions}
            for decision_id in set(chapter.produces_decisions) - produced:
                issues.append(
                    ConsistencyIssue(
                        issue_id=f"missing-decision:{chapter.chapter_id}:{decision_id}",
                        severity="error",
                        chapter_ids=[chapter.chapter_id],
                        description=f"章节未形成计划要求的决策 {decision_id}。",
                        recommendation="重新执行该章节并补齐决策依据。",
                    )
                )
        return issues

    @staticmethod
    def _review_payload(
        plan: DocumentPlan, packets: list[ResearchPacket]
    ) -> dict[str, Any]:
        return {
            "plan": {
                "title": plan.title,
                "chapters": [
                    {
                        "chapter_id": item.chapter_id,
                        "title": item.title,
                        "depends_on": item.depends_on,
                    }
                    for item in plan.chapters
                ],
            },
            "artifacts": [
                {
                    "chapter_id": packet.chapter_id,
                    "status": packet.status,
                    "summary": packet.summary,
                    "claims": [
                        {
                            "claim_id": claim.claim_id,
                            "text": claim.text,
                            "conclusion_type": claim.conclusion_type,
                        }
                        for claim in packet.claims
                    ],
                    "decisions": [
                        {
                            "decision_id": decision.decision_id,
                            "statement": decision.statement,
                            "decision_type": decision.decision_type,
                            "rationale": decision.rationale,
                            "confidence": decision.confidence,
                        }
                        for decision in packet.decisions
                    ],
                    "conflicts": [
                        item.model_dump(mode="json") for item in packet.conflicts
                    ],
                }
                for packet in packets
            ],
        }

    @staticmethod
    def _assemble_answer(
        plan: DocumentPlan,
        packets: list[ResearchPacket],
        issues: list[ConsistencyIssue],
    ) -> AgentAnswer:
        lines = [f"# {plan.title}"]
        used_evidence: list[str] = []
        limitations: list[str] = []
        for packet in packets:
            lines.extend(["", f"## {packet.chapter_title or packet.task}"])
            if packet.status != "sufficient":
                labels = {
                    "insufficient": "本章参考依据不足",
                    "failed": "本章执行失败",
                    "blocked": "本章因上游未完成而跳过",
                }
                lines.append(f"{labels.get(packet.status, '本章未完成')}：{packet.summary}")
                limitations.extend(packet.gaps)
                limitations.extend(packet.diagnostics)
                continue
            for block in packet.content_blocks:
                if block.heading:
                    lines.extend(["", f"### {block.heading}"])
                text = block.markdown.strip()
                missing_inline = [
                    item for item in block.evidence_ids if item not in text
                ]
                if missing_inline:
                    text += " " + " ".join(f"[{item}]" for item in missing_inline)
                lines.extend(["", text])
                for evidence_id in block.evidence_ids:
                    if evidence_id not in used_evidence:
                        used_evidence.append(evidence_id)
            limitations.extend(packet.gaps)

        limitations.extend(
            item.description for item in issues if item.severity == "error"
        )
        return AgentAnswer(
            content="\n".join(lines),
            evidence_ids=used_evidence,
            limitations=list(dict.fromkeys(limitations)),
        )

    def _build_root_graph(self):
        router = self.model.with_structured_output(RouteDecision, method="json_mode")
        planner = self.model.with_structured_output(DocumentPlan, method="json_mode")
        reviewer = self.model.with_structured_output(ConsistencyReport, method="json_mode")

        def route(state: RootState, config: RunnableConfig) -> dict:
            decision = router.invoke(
                [
                    SystemMessage(content=ROUTER_PROMPT),
                    HumanMessage(content=state["request"]),
                ],
                config=config,
            )
            return {"route": decision.model_dump(mode="json")}

        def route_mode(state: RootState) -> str:
            return state["route"]["mode"]

        def fast(state: RootState, config: RunnableConfig) -> dict:
            result = self._fast_graph.invoke(
                {"messages": [HumanMessage(content=state["request"])], "steps": 0},
                {
                    **config,
                    "run_name": "fast-react-agent",
                    "recursion_limit": self.fast_max_steps * 2 + 4,
                },
            )
            return {"answer": result["result"]}

        def plan_document(state: RootState, config: RunnableConfig) -> dict:
            schema = json.dumps(DocumentPlan.model_json_schema(), ensure_ascii=False)
            plan = planner.invoke(
                [
                    SystemMessage(
                        content=(
                            f"{PLANNER_PROMPT}\nMaximum chapters: {self.max_subtasks}.\n"
                            f"JSON schema: {schema}"
                        )
                    ),
                    HumanMessage(content=state["request"]),
                ],
                config=config,
            )
            self._validate_plan(plan)
            return {"plan": plan.model_dump(mode="json")}

        def research_chapters(state: RootState, config: RunnableConfig) -> dict:
            plan = DocumentPlan.model_validate(state["plan"])
            packets = self._execute_plan(plan, config)
            return {"packets": [item.model_dump(mode="json") for item in packets]}

        def review(state: RootState, config: RunnableConfig) -> dict:
            plan = DocumentPlan.model_validate(state["plan"])
            packets = [ResearchPacket.model_validate(item) for item in state["packets"]]
            issues = self._structural_consistency_issues(plan, packets)
            if any(item.status != "sufficient" for item in packets):
                return {"issues": [item.model_dump(mode="json") for item in issues]}
            report = reviewer.invoke(
                [
                    SystemMessage(
                        content=(
                            f"{REVIEW_PROMPT}\n"
                            f"JSON schema: {json.dumps(ConsistencyReport.model_json_schema(), ensure_ascii=False)}"
                        )
                    ),
                    HumanMessage(
                        content=json.dumps(
                            self._review_payload(plan, packets), ensure_ascii=False
                        )
                    ),
                ],
                config=config,
            )
            known_chapters = {item.chapter_id for item in plan.chapters}
            for issue in report.issues:
                if set(issue.chapter_ids) - known_chapters:
                    continue
                if issue.issue_id not in {item.issue_id for item in issues}:
                    issues.append(issue)
            return {"issues": [item.model_dump(mode="json") for item in issues]}

        def assemble(state: RootState) -> dict:
            plan = DocumentPlan.model_validate(state["plan"])
            packets = [ResearchPacket.model_validate(item) for item in state["packets"]]
            issues = [ConsistencyIssue.model_validate(item) for item in state["issues"]]
            answer = self._assemble_answer(plan, packets, issues)
            return {"answer": answer.model_dump(mode="json")}

        builder = StateGraph(RootState)
        builder.add_node("router", route)
        builder.add_node("fast_agent", fast)
        builder.add_node("chapter_planner", plan_document)
        builder.add_node("chapter_research", research_chapters)
        builder.add_node("consistency_review", review)
        builder.add_node("document_assembler", assemble)
        builder.add_edge(START, "router")
        builder.add_conditional_edges(
            "router",
            route_mode,
            {"fast": "fast_agent", "supervisor": "chapter_planner"},
        )
        builder.add_edge("fast_agent", END)
        builder.add_edge("chapter_planner", "chapter_research")
        builder.add_edge("chapter_research", "consistency_review")
        builder.add_edge("consistency_review", "document_assembler")
        builder.add_edge("document_assembler", END)
        return builder.compile(name="research-router")

    def run(
        self,
        request: str,
        *,
        config: RunnableConfig | None = None,
        run_id: str | None = None,
        trace_id: str | None = None,
    ) -> tuple[AgentRun, Any]:
        if not request.strip():
            raise ValueError("request must not be blank")
        with self._run_lock:
            self.workspace.reset()
            with self._lock:
                self._packets.clear()
            state = self.graph.invoke(
                {"request": request.strip()},
                config or {"recursion_limit": self.max_steps * 4 + 8},
            )
            route = RouteDecision.model_validate(state["route"])
            answer = AgentAnswer.model_validate(state["answer"])
            self.workspace.validate_evidence_ids(answer.evidence_ids)
            plan = (
                DocumentPlan.model_validate(state["plan"])
                if route.mode == "supervisor"
                else None
            )
            issues = [
                ConsistencyIssue.model_validate(item)
                for item in state.get("issues", [])
            ]
            outcome = (
                "completed"
                if (
                    route.mode == "fast"
                    and bool(answer.evidence_ids)
                    or route.mode == "supervisor"
                    and all(item.status == "sufficient" for item in self.packets)
                    and not any(item.severity == "error" for item in issues)
                )
                else "incomplete"
            )
            run = AgentRun(
                **({"run_id": run_id} if run_id is not None else {}),
                request=request.strip(),
                route=route,
                outcome=outcome,
                answer=answer,
                document_plan=plan,
                consistency_issues=issues,
                evidence=self.workspace.evidence,
                worker_packets=self.packets,
                trace_id=trace_id,
            )
            path = self.store.save(run)
            return run, path
