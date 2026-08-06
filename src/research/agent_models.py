"""Persisted contracts for routed ReAct agent runs."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from src.research.models import Claim, Conflict, Evidence


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RouteDecision(BaseModel):
    mode: Literal["fast", "supervisor"]
    reason: str = Field(min_length=1)


class ChapterPlan(BaseModel):
    chapter_id: str = Field(min_length=1)
    ordinal: int = Field(ge=1)
    title: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    research_questions: list[str] = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)
    produces_decisions: list[str] = Field(default_factory=list)
    required_decisions: list[str] = Field(default_factory=list)
    required_glossary: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(min_length=1)


class DocumentPlan(BaseModel):
    title: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    deliverable_mode: Literal["evidence_summary", "normative_synthesis"] = (
        "evidence_summary"
    )
    chapters: list[ChapterPlan] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_graph(self) -> "DocumentPlan":
        chapter_ids = [item.chapter_id for item in self.chapters]
        if len(chapter_ids) != len(set(chapter_ids)):
            raise ValueError("Chapter IDs must be unique")
        ordinals = [item.ordinal for item in self.chapters]
        if len(ordinals) != len(set(ordinals)):
            raise ValueError("Chapter ordinals must be unique")

        known = set(chapter_ids)
        producers: dict[str, str] = {}
        for chapter in self.chapters:
            unknown = set(chapter.depends_on) - known
            if unknown:
                raise ValueError(
                    f"Chapter {chapter.chapter_id} has unknown dependencies: {sorted(unknown)}"
                )
            if chapter.chapter_id in chapter.depends_on:
                raise ValueError(f"Chapter {chapter.chapter_id} depends on itself")
            for decision_id in chapter.produces_decisions:
                if decision_id in producers:
                    raise ValueError(f"Decision {decision_id} has multiple producers")
                producers[decision_id] = chapter.chapter_id

        visiting: set[str] = set()
        visited: set[str] = set()
        by_id = {item.chapter_id: item for item in self.chapters}

        def visit(chapter_id: str) -> None:
            if chapter_id in visiting:
                raise ValueError("Chapter dependency graph contains a cycle")
            if chapter_id in visited:
                return
            visiting.add(chapter_id)
            for dependency in by_id[chapter_id].depends_on:
                visit(dependency)
            visiting.remove(chapter_id)
            visited.add(chapter_id)

        for chapter_id in chapter_ids:
            visit(chapter_id)

        for chapter in self.chapters:
            missing = set(chapter.required_decisions) - set(producers)
            if missing:
                raise ValueError(
                    f"Chapter {chapter.chapter_id} requires unproduced decisions: {sorted(missing)}"
                )
            ancestors: set[str] = set()

            def collect_ancestors(chapter_id: str) -> None:
                for dependency in by_id[chapter_id].depends_on:
                    if dependency not in ancestors:
                        ancestors.add(dependency)
                        collect_ancestors(dependency)

            collect_ancestors(chapter.chapter_id)
            inaccessible = {
                decision_id
                for decision_id in chapter.required_decisions
                if producers[decision_id] not in ancestors
            }
            if inaccessible:
                raise ValueError(
                    f"Chapter {chapter.chapter_id} cannot reach required decisions: "
                    f"{sorted(inaccessible)}"
                )
        return self


class ContentBlock(BaseModel):
    block_id: str = Field(min_length=1)
    heading: str | None = None
    markdown: str = Field(min_length=1)
    claim_ids: list[str] = Field(default_factory=list)
    decision_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class GlossaryEntry(BaseModel):
    """One axis of a controlled-vocabulary decision (e.g. "杀伤等级", "毁伤程度").

    Only canonical terms are listed; forbidden aliases are intentionally omitted
    so the glossary travels to downstream workers as an executable vocabulary,
    not a free-text declaration.
    """

    axis: str = Field(min_length=1)
    canonical_terms: list[str] = Field(min_length=1)
    scope: str = ""


class DecisionRecord(BaseModel):
    decision_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    decision_type: Literal["direct", "synthesized", "normative", "hypothesis"]
    rationale: str = Field(min_length=1)
    claim_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)
    validation_requirements: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    applies_to_chapters: list[str] = Field(default_factory=list)
    glossary: list[GlossaryEntry] = Field(default_factory=list)


class ConsistencyIssue(BaseModel):
    issue_id: str = Field(min_length=1)
    severity: Literal["warning", "error"]
    chapter_ids: list[str] = Field(default_factory=list)
    description: str = Field(min_length=1)
    recommendation: str = Field(min_length=1)


class ConsistencyReport(BaseModel):
    issues: list[ConsistencyIssue] = Field(default_factory=list)


class ChapterSubmission(BaseModel):
    """The model-facing artifact a chapter worker emits via submit_chapter.

    It carries only what the model must generate: the research content and its
    audit metadata. Program-owned identity (task, chapter_id, chapter_title,
    depends_on) is injected from the ChapterPlan by the runtime, and the
    top-level evidence_ids is derived from Claim/Decision citations in
    _validate_packet, so neither appears here. status is restricted to the two
    outcomes a worker can reach on its own: failed/blocked packets are built by
    the runtime (_failed_packet/_blocked_packet) and never emitted by the model.
    """

    status: Literal["sufficient", "insufficient"]
    summary: str = Field(min_length=1)
    content_blocks: list[ContentBlock] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    decisions: list[DecisionRecord] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)


class ResearchPacket(BaseModel):
    task: str = Field(min_length=1)
    chapter_id: str | None = None
    chapter_title: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    status: Literal["sufficient", "insufficient", "failed", "blocked"]
    summary: str = Field(min_length=1)
    content_blocks: list[ContentBlock] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    decisions: list[DecisionRecord] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class AgentAnswer(BaseModel):
    content: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class AgentRun(BaseModel):
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    request: str = Field(min_length=1)
    route: RouteDecision
    outcome: Literal["completed", "incomplete"] = "completed"
    answer: AgentAnswer
    document_plan: DocumentPlan | None = None
    consistency_issues: list[ConsistencyIssue] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    worker_packets: list[ResearchPacket] = Field(default_factory=list)
    evidence_aliases: dict[str, dict[str, str]] = Field(default_factory=dict)
    parent_run_id: str | None = None
    attempt: int = Field(default=1, ge=1)
    review_revised: bool = False
    review_verified: bool = True
    requires_human_review: bool = False
    trace_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class RunCheckpoint(BaseModel):
    run_id: str
    request: str
    route: RouteDecision
    document_plan: DocumentPlan
    evidence: list[Evidence] = Field(default_factory=list)
    worker_packets: list[ResearchPacket] = Field(default_factory=list)
    evidence_aliases: dict[str, dict[str, str]] = Field(default_factory=dict)
    parent_run_id: str | None = None
    attempt: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utc_now)
