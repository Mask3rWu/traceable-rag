"""Explicit, checkpointed workflow for a traceable research run."""
from __future__ import annotations

from uuid import uuid4

from src.research.client import ResearchModel
from src.research.evidence import CitationVerifier, EvidenceResolver, merge_evidence
from src.research.models import Claim, Conflict, ResearchRun, ToolCall
from src.research.store import ResearchRunStore
from src.retrieval.service import RetrievalService


class ResearchWorkflow:
    def __init__(
        self,
        *,
        model: ResearchModel,
        retrieval: RetrievalService,
        resolver: EvidenceResolver,
        verifier: CitationVerifier,
        store: ResearchRunStore | None = None,
        max_queries: int = 4,
        evidence_limit: int = 10,
    ) -> None:
        if max_queries <= 0 or evidence_limit <= 0:
            raise ValueError("max_queries and evidence_limit must be greater than zero")
        self.model = model
        self.retrieval = retrieval
        self.resolver = resolver
        self.verifier = verifier
        self.store = store or ResearchRunStore()
        self.max_queries = max_queries
        self.evidence_limit = evidence_limit

    def run(self, question: str) -> ResearchRun:
        if not question.strip():
            raise ValueError("question must not be blank")
        run = ResearchRun(question=question.strip())
        self.store.save(run)
        try:
            self._plan(run)
            self._retrieve(run)
            self._synthesize(run)
            self._verify(run)
            run.status = "completed"
            self.store.save(run)
            return run
        except Exception as exc:
            run.status = "failed"
            run.error = f"{type(exc).__name__}: {exc}"
            self.store.save(run)
            raise

    def _plan(self, run: ResearchRun) -> None:
        run.status = "planning"
        self.store.save(run)
        planned = self.model.plan_queries(run.question, max_queries=self.max_queries)
        queries: list[str] = []
        for query in [run.question, *planned]:
            normalized = query.strip()
            if normalized and normalized not in queries:
                queries.append(normalized)
        run.queries = queries[: self.max_queries]
        run.tool_calls.append(
            ToolCall(
                call_id=uuid4().hex,
                tool="plan_queries",
                arguments={"question": run.question, "max_queries": self.max_queries},
                result={"queries": run.queries},
            )
        )
        self.store.save(run)

    def _retrieve(self, run: ResearchRun) -> None:
        run.status = "retrieving"
        self.store.save(run)
        for query in run.queries:
            results = self.retrieval.search(query, limit=self.evidence_limit)
            resolved = self.resolver.resolve_many(query, results)
            run.evidence = merge_evidence(run.evidence, resolved)
            run.tool_calls.append(
                ToolCall(
                    call_id=uuid4().hex,
                    tool="search",
                    arguments={"query": query, "limit": self.evidence_limit},
                    result={
                        "evidence_ids": [item.evidence_id for item in resolved],
                        "chunk_ids": [item.chunk_id for item in resolved],
                    },
                )
            )
            self.store.save(run)
        if not run.evidence:
            raise RuntimeError("Research retrieval returned no evidence")

    def _synthesize(self, run: ResearchRun) -> None:
        run.status = "synthesizing"
        self.store.save(run)
        draft = self.model.synthesize(run.question, run.evidence)
        self._validate_draft(draft.claims, draft.conflicts)
        run.claims = draft.claims
        run.conflicts = draft.conflicts
        run.summary = draft.summary
        run.tool_calls.append(
            ToolCall(
                call_id=uuid4().hex,
                tool="synthesize",
                arguments={
                    "question": run.question,
                    "evidence_ids": [item.evidence_id for item in run.evidence],
                },
                result={
                    "claim_ids": [item.claim_id for item in run.claims],
                    "conflict_ids": [item.conflict_id for item in run.conflicts],
                },
            )
        )
        self.store.save(run)

    @staticmethod
    def _validate_draft(claims: list[Claim], conflicts: list[Conflict]) -> None:
        claim_ids = [item.claim_id for item in claims]
        conflict_ids = [item.conflict_id for item in conflicts]
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError("Research draft contains duplicate claim IDs")
        if len(set(conflict_ids)) != len(conflict_ids):
            raise ValueError("Research draft contains duplicate conflict IDs")
        unresolved = [
            item.conflict_id
            for item in conflicts
            if item.status == "resolved" and not (item.resolution or "").strip()
        ]
        if unresolved:
            raise ValueError(
                "Resolved conflicts require a resolution: " + ", ".join(unresolved)
            )

    def _verify(self, run: ResearchRun) -> None:
        run.status = "verifying"
        self.store.save(run)
        run.evidence = [self.verifier.verify_evidence(item) for item in run.evidence]
        evidence_by_id = {item.evidence_id: item for item in run.evidence}
        run.claims = [
            self.verifier.verify_claim(claim, evidence_by_id) for claim in run.claims
        ]
        self._verify_conflicts(run)
        run.tool_calls.append(
            ToolCall(
                call_id=uuid4().hex,
                tool="verify_citations",
                arguments={
                    "claim_ids": [item.claim_id for item in run.claims],
                },
                result={
                    "verified_evidence": len(run.evidence),
                    "verified_claims": len(run.claims),
                },
            )
        )
        self.store.save(run)

    @staticmethod
    def _verify_conflicts(run: ResearchRun) -> None:
        claim_ids = {item.claim_id for item in run.claims}
        evidence_ids = {item.evidence_id for item in run.evidence}
        for conflict in run.conflicts:
            unknown_claims = set(conflict.claim_ids) - claim_ids
            unknown_evidence = set(conflict.evidence_ids) - evidence_ids
            if unknown_claims or unknown_evidence:
                raise ValueError(
                    f"Conflict {conflict.conflict_id} has unknown references: "
                    f"claims={sorted(unknown_claims)}, "
                    f"evidence={sorted(unknown_evidence)}"
                )
