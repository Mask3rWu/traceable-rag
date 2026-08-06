"""Routed LangGraph runtime with chapter-planned ReAct research workers."""
from __future__ import annotations

import json
import re
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
    RunCheckpoint,
)
from src.research.agent_store import AgentRunStore
from src.research.tools import EvidenceAliasRegistry, EvidenceWorkspace


FAST_PROMPT = """You are a traceable knowledge-base question-answering agent.
Rewrite the user's need into focused searches. Use search_knowledge, inspect previews, and
read only the evidence needed. You may search again when coverage is weak. Answer only from
retrieved evidence. The answer is public user content: provide the conclusion directly, without
source names, document titles, citations, or evidence/Claim/Decision IDs. Keep evidence IDs only
in the structured evidence_ids field and finish by calling submit_answer."""

PLANNER_PROMPT = """You are the planning component of a research supervisor. Create a
chapter-level plan for the requested structured deliverable. Each chapter must have bounded
research questions and acceptance criteria. Put shared terminology, scope, classification
frameworks, and other global decisions in foundational chapters. Express execution order with
depends_on. A chapter that creates a reusable decision declares its stable ID in
produces_decisions; every consumer declares that ID in required_decisions and depends on its
producer. Each plan entry is exactly one final top-level chapter. Give chapters disjoint scopes;
never ask one chapter to recreate the complete document or material assigned to another chapter.
Independent chapters may run in parallel. Do not copy a source document's table of contents or
chapter numbering into the plan. Do not make factual claims or invent
evidence in the plan. Every chapter must first summarize, compare, and verify the relevant
sources. Choose evidence_summary when that evidence synthesis is the final deliverable. Choose
normative_synthesis when the request additionally asks for a new standard, taxonomy, threshold
system, or operating rule; this mode includes the evidence synthesis and adds a separate design
layer based on it. Use concise
stable ASCII identifiers and preserve the user's language.
The final deliverable is an operational standard for its end users, not a literature review.
Plan chapters around conclusions, definitions, criteria, tables, decision rules, procedures,
templates, and examples. Do not create a chapter whose user-facing content is mainly a source
comparison or an evidence summary; source comparison belongs in the structured claims,
decisions, conflicts, and evidence metadata.

The foundational chapter that owns the terminology decision must populate that decision's
glossary with one GlossaryEntry per controlled-vocabulary axis: a short axis name, the list of
canonical terms only (never forbidden aliases), and an optional scope note. Every downstream
chapter whose prose must obey that vocabulary declares the terminology decision's ID in
required_glossary (in addition to required_decisions). Glossaries make terminology an executable
contract, not a free-text declaration.
Return only a JSON object matching the supplied schema."""

WORKER_PROMPT = """You are a chapter research worker handling one bounded chapter. Work
through every research question using iterative search_knowledge and read_evidence calls. First
summarize, compare, and verify relevant source conclusions internally; this evidence synthesis
is required in both deliverable modes but must remain in the structured audit metadata, not in
the public ContentBlock.
Use small focused queries, shortlist the most relevant evidence, inspect exact source excerpts,
and stop broad searching once the questions have reasonable coverage. Respect all upstream
decisions. Every factual claim needs one or more evidence IDs; the system fills exact source
quotes. Submit exactly one ContentBlock for this chapter. Its heading must be null and its
markdown must not contain Markdown headings or recreate nested chapters. Link every Claim,
Decision, and used evidence ID to that single block. Keep the evidence synthesis compact; do not
reproduce a source standard's structure, table of contents, or unrelated clauses. Distinguish
direct, synthesized, normative, and
hypothesis conclusions. In normative_synthesis mode, the requested standard is allowed to be
new, but only after evidence synthesis: use the verified source comparison as design input and
keep source conclusions separate from proposed rules. Create explicit normative rules instead of
requiring a source that already contains the finished standard. Mark designed rules and
thresholds as normative, explain the transfer rationale, list assumptions and alternatives,
state validation requirements, and use lower confidence where empirical calibration is absent.
Do not mark a chapter insufficient merely because the exact requested standard is absent.
Never present a proposed rule as a source fact.

The ContentBlock is public, end-user standard text. It must contain conclusions and operational
rules only: definitions, requirements, thresholds, tables, decision steps, output formats, and
examples. Do not put source names, author names, document titles, standard numbers, evidence IDs,
Claim/Decision IDs, inline citations, literature comparisons, or phrases such as "according to"
or "the source shows" in the ContentBlock. Keep all source reasoning in Claim.citations and
DecisionRecord.rationale/evidence_ids, which are audit metadata shown separately by the system.
Do not expose internal labels such as C1, D1, CH4-C1, or ev-... in public prose.

One ContentBlock is a structured container, not one paragraph. Use multiple paragraphs separated
by blank lines, bullet lists, tables, decision trees, and examples as appropriate. Do not collapse
the whole chapter into a single dense paragraph. Finish by calling submit_chapter with a compact
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
    nudged: bool


class RootState(TypedDict, total=False):
    request: str
    route: dict[str, Any]
    plan: dict[str, Any]
    packets: list[dict[str, Any]]
    issues: list[dict[str, Any]]
    answer: dict[str, Any]
    review_revised: bool


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
        document_max_chars: int = 6000,
        chapter_max_chars: int = 1600,
        chapter_max_claims: int = 10,
        chapter_max_decisions: int = 4,
    ) -> None:
        if min(
            max_steps,
            max_workers,
            max_subtasks,
            document_max_chars,
            chapter_max_chars,
            chapter_max_claims,
            chapter_max_decisions,
        ) <= 0:
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
        self.document_max_chars = document_max_chars
        self.chapter_max_chars = chapter_max_chars
        self.chapter_max_claims = chapter_max_claims
        self.chapter_max_decisions = chapter_max_decisions
        self._packets: list[ResearchPacket] = []
        self._evidence_aliases: dict[str, EvidenceAliasRegistry] = {}
        self._cancel_check: Callable[[], bool] = lambda: False
        self._current_run_id: str | None = None
        self._current_parent_run_id: str | None = None
        self._current_attempt = 1
        self._current_request = ""
        self._lock = threading.RLock()
        self._run_lock = threading.Lock()
        self._fast_aliases = EvidenceAliasRegistry()
        self._fast_graph = self._build_fast_graph(self._fast_aliases)
        self.graph = self._build_root_graph()

    @property
    def packets(self) -> list[ResearchPacket]:
        with self._lock:
            return list(self._packets)

    @staticmethod
    def _submit_answer_tool(
        validator: Callable[[AgentAnswer], None] | None = None,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> BaseTool:
        @tool(args_schema=AgentAnswer)
        def submit_answer(
            content: str, evidence_ids: list[str], limitations: list[str]
        ) -> str:
            """Submit the final answer and stop this agent."""

            payload = {
                "content": content,
                "evidence_ids": evidence_ids,
                "limitations": limitations,
            }
            answer = AgentAnswer.model_validate(transform(payload) if transform else payload)
            if validator is not None:
                validator(answer)
            return "submitted"

        return submit_answer

    def _submit_chapter_tool(
        self,
        chapter: ChapterPlan,
        aliases: EvidenceAliasRegistry,
        chapter_char_limit: int | None = None,
    ) -> BaseTool:
        @tool(args_schema=ResearchPacket)
        def submit_chapter(**kwargs: Any) -> str:
            """Submit a grounded chapter artifact and stop this worker."""

            packet = ResearchPacket.model_validate(
                self._canonicalize_packet_payload(kwargs, chapter, aliases)
            )
            self._validate_packet(packet, chapter, chapter_char_limit)
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
        result_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
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
            return "nudge" if not state.get("nudged") else "exhausted"

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
            args = call["args"]
            if result_transform is not None:
                args = result_transform(args)
            result = result_model.model_validate(args)
            return {"result": result.model_dump(mode="json")}

        def nudge(_: ReactState) -> dict:
            return {
                "messages": [
                    HumanMessage(
                        content=(
                            f"You must finish now by calling {submit_name}. "
                            "Do not perform more broad searches. Correct any reported "
                            "validation error and submit the structured result."
                        )
                    )
                ],
                "nudged": True,
            }

        def exhausted(state: ReactState) -> dict:
            result = exhausted_result(state, state.get("steps", 0) >= step_limit)
            return {"result": result.model_dump(mode="json")}

        builder = StateGraph(ReactState)
        builder.add_node("agent", call_model)
        builder.add_node("tools", tool_node)
        builder.add_node("submit", submit)
        builder.add_node("nudge", nudge)
        builder.add_node("exhausted", exhausted)
        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent",
            next_step,
            {
                "tools": "tools",
                "submit": "submit",
                "nudge": "nudge",
                "exhausted": "exhausted",
            },
        )
        builder.add_conditional_edges(
            "tools",
            after_tools,
            {"agent": "agent", "submit": "submit", "exhausted": "exhausted"},
        )
        builder.add_edge("submit", END)
        builder.add_edge("nudge", "agent")
        builder.add_edge("exhausted", END)
        return builder.compile(name=graph_name)

    def _build_fast_graph(self, aliases: EvidenceAliasRegistry):
        translate = aliases.translate_payload
        tools = self._retrieval_tools(aliases)
        tools.append(self._submit_answer_tool(self._validate_fast_answer, translate))
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
            result_transform=translate,
        )

    def _build_chapter_graph(
        self,
        chapter: ChapterPlan,
        aliases: EvidenceAliasRegistry,
        chapter_char_limit: int | None = None,
    ):
        prose_limit = chapter_char_limit or self.chapter_max_chars
        tools = [
            *self._retrieval_tools(aliases),
            self.workspace.make_terminology_tool(),
            self._submit_chapter_tool(chapter, aliases, prose_limit),
        ]

        return self._build_react_graph(
            prompt=(
                f"{WORKER_PROMPT}\nHard output contract for this chapter: exactly one "
                f"ContentBlock, at most {prose_limit} characters of chapter prose, "
                f"{self.chapter_max_claims} Claims, and "
                f"{self.chapter_max_decisions} Decisions. Use short unnumbered local "
                "labels inside prose when needed. The document assembler owns the block "
                "heading and all chapter numbering.\n"
                "Terminology self-check: when upstream glossary decisions are provided in "
                "the request, call check_terminology with your draft content_blocks and "
                "those decisions before submit_chapter. If it returns suspect_terms, "
                "revise the prose to use only the canonical terms from the relevant axis. "
                "check_terminology is advisory and never blocks submission; if you judge a "
                "flagged term is not a controlled-vocabulary drift, you may keep it. "
                "Always use the canonical terms from the glossary when writing terminology."
                " Evidence references exposed by search are short aliases such as E1 and E2. "
                "Use those aliases exactly in read_evidence and every structured evidence field; "
                "the system resolves them to stable provenance IDs before persistence."
            ),
            tools=tools,
            submit_name="submit_chapter",
            result_model=ResearchPacket,
            exhausted_result=lambda state, budget_exhausted: self._failed_packet(
                chapter, state, budget_exhausted
            ),
            graph_name="chapter-research-worker",
            step_limit=self.worker_max_steps,
            result_transform=lambda args: self._canonicalize_packet_payload(
                args, chapter, aliases
            ),
        )

    def _retrieval_tools(self, aliases: EvidenceAliasRegistry) -> list[BaseTool]:
        try:
            return self.workspace.make_retrieval_tools(aliases, self._cancel_check)
        except TypeError as exc:
            if "positional argument" not in str(exc) and "unexpected keyword" not in str(exc):
                raise
            return self.workspace.make_retrieval_tools()

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
        if re.search(r"(?i)\bev-[a-z0-9]+\b|\b(?:C|D)\d+\b", answer.content):
            raise ValueError(
                "Public answer content must not contain internal evidence, Claim, or Decision IDs"
            )
        self.workspace.validate_evidence_ids(answer.evidence_ids)

    @staticmethod
    def _canonicalize_packet_payload(
        payload: dict[str, Any],
        chapter: ChapterPlan,
        aliases: EvidenceAliasRegistry,
    ) -> dict[str, Any]:
        normalized = ResearchPacket.model_validate(payload).model_dump(mode="json")
        translated = aliases.translate_payload(normalized)
        stable_decisions = set(chapter.produces_decisions)
        prefix = f"{chapter.chapter_id}:"
        decisions = translated.get("decisions") or []
        decision_map: dict[str, str] = {}
        for decision in decisions:
            raw_id = str(decision.get("decision_id", ""))
            if raw_id in stable_decisions or raw_id.startswith(prefix):
                canonical = raw_id
            else:
                canonical = prefix + raw_id
            decision_map[raw_id] = canonical
            decision["decision_id"] = canonical
        for block in translated.get("content_blocks") or []:
            block["decision_ids"] = [
                decision_map.get(item, item if item in stable_decisions else prefix + item)
                for item in block.get("decision_ids", [])
            ]
        translated["depends_on"] = list(chapter.depends_on)
        return translated

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

    def _validate_packet(
        self,
        packet: ResearchPacket,
        chapter: ChapterPlan,
        chapter_char_limit: int | None = None,
    ) -> None:
        prose_limit = chapter_char_limit or self.chapter_max_chars
        if packet.chapter_id != chapter.chapter_id:
            raise ValueError("Chapter artifact has the wrong chapter ID")
        if packet.chapter_title != chapter.title:
            raise ValueError("Chapter artifact has the wrong chapter title")

        if packet.content_blocks and len(packet.content_blocks) != 1:
            raise ValueError(
                "A chapter artifact must contain exactly one ContentBlock; merge all "
                f"chapter prose into one block (got {len(packet.content_blocks)})"
            )
        prose_chars = sum(
            len(block.markdown) + len(block.heading or "")
            for block in packet.content_blocks
        )
        if prose_chars > prose_limit:
            raise ValueError(
                "Chapter prose exceeds the character budget "
                f"({prose_chars} > {prose_limit}); remove detail owned by "
                "other chapters and condense the evidence synthesis"
            )
        if len(packet.claims) > self.chapter_max_claims:
            raise ValueError(
                "Chapter artifact exceeds the Claim budget "
                f"({len(packet.claims)} > {self.chapter_max_claims}); keep only claims "
                "needed by this chapter"
            )
        if len(packet.decisions) > self.chapter_max_decisions:
            raise ValueError(
                "Chapter artifact exceeds the Decision budget "
                f"({len(packet.decisions)} > {self.chapter_max_decisions})"
            )
        for block in packet.content_blocks:
            if block.heading:
                raise ValueError(
                    "The single ContentBlock must not define a heading; the document "
                    f"assembler uses the ChapterPlan title: {block.heading!r}"
                )
            if re.search(r"(?m)^\s*#{1,6}\s+", block.markdown):
                raise ValueError(
                    "Chapter prose must not contain Markdown headings; use bold labels, "
                    "lists, or tables inside the single ContentBlock"
                )
            if re.search(r"(?i)\bev-[a-z0-9]+\b|\b(?:C|D)\d+\b", block.markdown):
                raise ValueError(
                    "Public chapter prose must not contain internal evidence, Claim, or "
                    "Decision IDs; keep them in structured metadata"
                )

        claim_ids = {item.claim_id for item in packet.claims}
        decision_ids = {item.decision_id for item in packet.decisions}
        block_ids = {item.block_id for item in packet.content_blocks}
        if len(claim_ids) != len(packet.claims):
            raise ValueError("Claim IDs must be unique within a chapter")
        if len(decision_ids) != len(packet.decisions):
            raise ValueError("Decision IDs must be unique within a chapter")
        if len(block_ids) != len(packet.content_blocks):
            raise ValueError("Content block IDs must be unique within a chapter")

        uncited_claims = [item.claim_id for item in packet.claims if not item.citations]
        if uncited_claims:
            raise ValueError(
                f"Every Claim must cite evidence: {sorted(uncited_claims)}"
            )
        cited = {
            citation.evidence_id
            for claim in packet.claims
            for citation in claim.citations
        }
        decision_evidence = {
            evidence_id
            for decision in packet.decisions
            for evidence_id in decision.evidence_ids
        }
        used = cited | decision_evidence

        if packet.content_blocks:
            block = packet.content_blocks[0]
            if set(block.claim_ids) != claim_ids:
                raise ValueError(
                    "The single ContentBlock must reference every Claim exactly once"
                )
            if set(block.decision_ids) != decision_ids:
                raise ValueError(
                    "The single ContentBlock must reference every Decision exactly once"
                )
            if set(block.evidence_ids) != used:
                unexplained = set(block.evidence_ids) - used
                missing = used - set(block.evidence_ids)
                details = []
                if unexplained:
                    details.append(
                        f"evidence without a Claim or Decision reason: {sorted(unexplained)}"
                    )
                if missing:
                    details.append(f"reasoned evidence missing from block: {sorted(missing)}")
                raise ValueError(
                    "The single ContentBlock evidence must exactly match evidence explained "
                    "by Claims or Decisions; " + "; ".join(details)
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
            if len(packet.content_blocks) != 1 or not packet.claims or not used:
                raise ValueError(
                    "A sufficient chapter requires exactly one prose block, verified claims, "
                    "and used evidence"
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
        chapter: ChapterPlan,
        completed: dict[str, ResearchPacket],
        plan: DocumentPlan,
    ) -> list[ResearchPacket]:
        ordered: list[ResearchPacket] = []
        seen: set[str] = set()

        def collect(chapter_id: str) -> None:
            if chapter_id in seen:
                return
            packet = completed[chapter_id]
            planned = next(item for item in plan.chapters if item.chapter_id == chapter_id)
            for dependency in planned.depends_on:
                collect(dependency)
            seen.add(chapter_id)
            ordered.append(packet)

        for dependency in chapter.depends_on:
            collect(dependency)
        return ordered

    @classmethod
    def _upstream_context(
        cls,
        chapter: ChapterPlan,
        completed: dict[str, ResearchPacket],
        plan: DocumentPlan,
    ) -> dict[str, Any]:
        packets = cls._ancestor_packets(chapter, completed, plan)
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

    @classmethod
    def _glossary_context(
        cls,
        chapter: ChapterPlan,
        completed: dict[str, ResearchPacket],
        plan: DocumentPlan,
    ) -> list[dict[str, Any]]:
        """Flatten glossary-bearing decisions declared in required_glossary.

        Only decisions that both the chapter requested via required_glossary and
        actually carry a non-empty glossary are surfaced, so workers see an
        executable vocabulary list rather than a free-text declaration.
        """
        packets = cls._ancestor_packets(chapter, completed, plan)
        wanted = set(chapter.required_glossary)
        glossaries: list[dict[str, Any]] = []
        for item in packets:
            for decision in item.decisions:
                if decision.decision_id in wanted and decision.glossary:
                    glossaries.append(decision.model_dump(mode="json"))
        return glossaries

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
            "document_structure": [
                {
                    "ordinal": item.ordinal,
                    "chapter_id": item.chapter_id,
                    "title": item.title,
                }
                for item in sorted(plan.chapters, key=lambda item: item.ordinal)
            ],
            "chapter": chapter.model_dump(mode="json"),
            "upstream": self._upstream_context(chapter, completed, plan),
            "glossary": self._glossary_context(chapter, completed, plan),
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
                "Treat document_structure as a hard ownership boundary: do not write sections "
                "owned by another chapter. Do not add chapter numbers or reuse numbering from "
                "sources. The ContentBlock is public standard text: write conclusions and "
                "operational rules only, with no source names, citations, evidence IDs, or "
                "internal Claim/Decision IDs. Use multiple paragraphs separated by blank lines "
                "when the chapter has multiple conclusions. Submit exactly one ContentBlock "
                "with heading=null and no Markdown headings. It must reference every Claim and "
                "Decision, and its evidence_ids must exactly match evidence explained by Claim "
                "citations or Decisions; these fields are metadata and are not printed in the "
                "public document. "
                "When previous_attempt reports evidence gaps, focus new searches on them. "
                "When it reports diagnostics, correct the structured submission before doing "
                "more broad retrieval."
            ),
        }
        chapter_char_limit = min(
            self.chapter_max_chars,
            max(1, self.document_max_chars // len(plan.chapters)),
        )
        aliases = self._evidence_aliases.setdefault(chapter.chapter_id, EvidenceAliasRegistry())
        graph = self._build_chapter_graph(chapter, aliases, chapter_char_limit)
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
        self._validate_packet(packet, chapter, chapter_char_limit)
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
        self,
        plan: DocumentPlan,
        config: RunnableConfig,
        initial_packets: Sequence[ResearchPacket] | None = None,
    ) -> list[ResearchPacket]:
        if self._cancel_check():
            raise RuntimeError("Research run was cancelled")
        remaining = {item.chapter_id: item for item in plan.chapters}
        completed: dict[str, ResearchPacket] = {
            item.chapter_id: item for item in (initial_packets or [])
        }
        for chapter_id in completed:
            remaining.pop(chapter_id, None)
        while remaining:
            if self._cancel_check():
                raise RuntimeError("Research run was cancelled")
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
                ancestors = self._ancestor_packets(chapter, completed, plan)
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
                        if self._cancel_check():
                            raise RuntimeError("Research run was cancelled")
                        chapter = futures[future]
                        packet = future.result()
                        packet.depends_on = list(chapter.depends_on)
                        wave_results[chapter.chapter_id] = packet
                completed.update(wave_results)
                self._save_checkpoint(plan, completed)

            for chapter in ready:
                remaining.pop(chapter.chapter_id)

        ordered = [
            completed[item.chapter_id]
            for item in sorted(plan.chapters, key=lambda chapter: chapter.ordinal)
        ]
        with self._lock:
            self._packets = ordered
        return ordered

    def _save_checkpoint(
        self, plan: DocumentPlan, completed: dict[str, ResearchPacket]
    ) -> None:
        if not self._current_run_id:
            return
        self.store.save_checkpoint(
            RunCheckpoint(
                run_id=self._current_run_id,
                request=self._current_request or plan.title,
                route=RouteDecision(
                    mode="supervisor", reason="resumable chapter checkpoint"
                ),
                document_plan=plan,
                evidence=self.workspace.evidence,
                worker_packets=list(completed.values()),
                evidence_aliases={
                    key: registry.export()
                    for key, registry in self._evidence_aliases.items()
                },
                parent_run_id=self._current_parent_run_id,
                attempt=self._current_attempt,
            )
        )

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

    def _revise_reviewed_chapters(
        self,
        plan: DocumentPlan,
        packets: list[ResearchPacket],
        issues: list[ConsistencyIssue],
        config: RunnableConfig,
    ) -> tuple[list[ResearchPacket], bool]:
        errors_by_chapter: dict[str, list[str]] = {}
        for issue in issues:
            if issue.severity != "error":
                continue
            for chapter_id in issue.chapter_ids:
                errors_by_chapter.setdefault(chapter_id, []).append(
                    f"Consistency review: {issue.description} Required correction: {issue.recommendation}"
                )
        if not errors_by_chapter:
            return packets, False

        completed = {item.chapter_id: item for item in packets if item.chapter_id}
        by_plan = {item.chapter_id: item for item in plan.chapters}
        revised = False
        for chapter_id in sorted(
            errors_by_chapter, key=lambda item: by_plan[item].ordinal
        ):
            if self._cancel_check():
                raise RuntimeError("Research run was cancelled")
            previous = completed[chapter_id]
            repair_context = previous.model_copy(
                update={
                    "diagnostics": list(
                        dict.fromkeys(
                            [*previous.diagnostics, *errors_by_chapter[chapter_id]]
                        )
                    )
                }
            )
            candidate = self._run_chapter(
                plan, by_plan[chapter_id], completed, config, previous_attempt=repair_context
            )
            if candidate.status == "sufficient":
                completed[chapter_id] = candidate
                revised = True
            else:
                previous.diagnostics = list(
                    dict.fromkeys(
                        [
                            *previous.diagnostics,
                            "Consistency revision failed; original sufficient chapter retained",
                            *candidate.diagnostics,
                        ]
                    )
                )
        ordered = [completed[item.chapter_id] for item in sorted(plan.chapters, key=lambda x: x.ordinal)]
        with self._lock:
            self._packets = ordered
        return ordered, revised

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
                if "\\n" in text and "\n" not in text:
                    text = text.replace("\\n", "\n")
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
            messages = [
                SystemMessage(
                    content=(
                        f"{PLANNER_PROMPT}\nMaximum chapters: {self.max_subtasks}.\n"
                        f"JSON schema: {schema}"
                    )
                ),
                HumanMessage(content=state["request"]),
            ]
            last_error: Exception | None = None
            for _ in range(2):
                try:
                    plan = planner.invoke(messages, config=config)
                    self._validate_plan(plan)
                    return {"plan": plan.model_dump(mode="json")}
                except Exception as exc:
                    last_error = exc
                    messages.append(
                        HumanMessage(
                            content=(
                                "The previous plan failed schema validation: "
                                f"{type(exc).__name__}: {exc}. Return corrected JSON only."
                            )
                        )
                    )
            raise RuntimeError(f"Planner failed after one correction retry: {last_error}")

        def research_chapters(state: RootState, config: RunnableConfig) -> dict:
            plan = DocumentPlan.model_validate(state["plan"])
            packets = self._execute_plan(plan, config)
            return {"packets": [item.model_dump(mode="json") for item in packets]}

        def review(state: RootState, config: RunnableConfig) -> dict:
            plan = DocumentPlan.model_validate(state["plan"])
            packets = [ResearchPacket.model_validate(item) for item in state["packets"]]
            issues = self._structural_consistency_issues(plan, packets)
            if any(item.status != "sufficient" for item in packets):
                return {
                    "issues": [item.model_dump(mode="json") for item in issues],
                    "review_revised": False,
                }
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
            revised_packets, revised = self._revise_reviewed_chapters(
                plan, packets, issues, config
            )
            return {
                "packets": [item.model_dump(mode="json") for item in revised_packets],
                "issues": [item.model_dump(mode="json") for item in issues],
                "review_revised": revised,
            }

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
        parent_run_id: str | None = None,
        attempt: int = 1,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[AgentRun, Any]:
        if not request.strip():
            raise ValueError("request must not be blank")
        with self._run_lock:
            self._current_run_id = run_id
            self._current_parent_run_id = parent_run_id
            self._current_attempt = attempt
            self._current_request = request.strip()
            self._cancel_check = cancel_check or (lambda: False)
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
            review_revised = bool(state.get("review_revised", False))
            outcome = (
                "completed"
                if (
                    route.mode == "fast"
                    and bool(answer.evidence_ids)
                    or route.mode == "supervisor"
                    and all(item.status == "sufficient" for item in self.packets)
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
                evidence_aliases={
                    key: registry.export()
                    for key, registry in self._evidence_aliases.items()
                },
                parent_run_id=parent_run_id,
                attempt=attempt,
                review_revised=review_revised,
                review_verified=not review_revised,
                requires_human_review=any(
                    item.severity == "error" for item in issues
                ),
                trace_id=trace_id,
            )
            path = self.store.save(run)
            self._cancel_check = lambda: False
            return run, path

    def resume(
        self,
        checkpoint: RunCheckpoint,
        *,
        run_id: str | None = None,
        trace_id: str | None = None,
        config: RunnableConfig | None = None,
        start_chapter: str | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[AgentRun, Any]:
        """Continue a supervisor run without repeating completed chapter work."""
        with self._run_lock:
            self._current_run_id = run_id
            self._current_parent_run_id = checkpoint.run_id
            self._current_attempt = checkpoint.attempt + 1
            self._current_request = checkpoint.request
            self._cancel_check = cancel_check or (lambda: False)
            self.workspace.reset()
            self.workspace.restore(checkpoint.evidence)
            self._evidence_aliases = {}
            for chapter_id, mapping in checkpoint.evidence_aliases.items():
                aliases = EvidenceAliasRegistry()
                aliases.restore(mapping)
                self._evidence_aliases[chapter_id] = aliases

            plan = checkpoint.document_plan
            initial_packets = list(checkpoint.worker_packets)
            if start_chapter:
                chapter_by_id = {item.chapter_id: item for item in plan.chapters}
                if start_chapter not in chapter_by_id:
                    raise ValueError(f"Unknown chapter: {start_chapter}")
                start_ordinal = chapter_by_id[start_chapter].ordinal
                initial_packets = [
                    packet
                    for packet in initial_packets
                    if packet.chapter_id in chapter_by_id
                    and chapter_by_id[packet.chapter_id].ordinal < start_ordinal
                ]
            else:
                initial_packets = [
                    packet
                    for packet in initial_packets
                    if packet.status == "sufficient"
                ]

            runnable_config = config or {"recursion_limit": self.max_steps * 4 + 8}
            packets = self._execute_plan(
                plan, runnable_config, initial_packets=initial_packets
            )
            issues = self._structural_consistency_issues(plan, packets)
            revised = False
            if all(item.status == "sufficient" for item in packets):
                reviewer = self.model.with_structured_output(
                    ConsistencyReport, method="json_mode"
                )
                report = reviewer.invoke(
                    [
                        SystemMessage(
                            content=(
                                f"{REVIEW_PROMPT}\nJSON schema: "
                                f"{json.dumps(ConsistencyReport.model_json_schema(), ensure_ascii=False)}"
                            )
                        ),
                        HumanMessage(
                            content=json.dumps(
                                self._review_payload(plan, packets), ensure_ascii=False
                            )
                        ),
                    ],
                    config=runnable_config,
                )
                known_chapters = {item.chapter_id for item in plan.chapters}
                issues.extend(
                    item
                    for item in report.issues
                    if not (set(item.chapter_ids) - known_chapters)
                    and item.issue_id not in {current.issue_id for current in issues}
                )
                packets, revised = self._revise_reviewed_chapters(
                    plan, packets, issues, runnable_config
                )
            answer = self._assemble_answer(plan, packets, issues)
            run = AgentRun(
                **({"run_id": run_id} if run_id else {}),
                request=checkpoint.request,
                route=checkpoint.route,
                outcome=(
                    "completed"
                    if all(item.status == "sufficient" for item in packets)
                    else "incomplete"
                ),
                answer=answer,
                document_plan=plan,
                consistency_issues=issues,
                evidence=self.workspace.evidence,
                worker_packets=packets,
                evidence_aliases={
                    key: aliases.export()
                    for key, aliases in self._evidence_aliases.items()
                },
                parent_run_id=checkpoint.run_id,
                attempt=checkpoint.attempt + 1,
                review_revised=revised,
                review_verified=not revised,
                requires_human_review=any(
                    item.severity == "error" for item in issues
                ),
                trace_id=trace_id,
            )
            path = self.store.save(run)
            self._cancel_check = lambda: False
            return run, path
