"""LangChain tools backed by the existing traceable retrieval layer."""
from __future__ import annotations

import json
import threading
from collections.abc import Sequence

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from src.research.evidence import CitationVerifier, EvidenceResolver, merge_evidence
from src.research.models import Evidence
from src.retrieval.service import RetrievalService


class SearchInput(BaseModel):
    query: str = Field(min_length=1, description="A focused knowledge-base query")
    top_k: int | None = Field(default=None, ge=1, le=20)


class ReadEvidenceInput(BaseModel):
    evidence_ids: list[str] = Field(min_length=1, max_length=4)


class EvidenceWorkspace:
    """Shared, thread-safe evidence registry for one top-level agent run."""

    def __init__(
        self,
        *,
        retrieval: RetrievalService,
        resolver: EvidenceResolver,
        verifier: CitationVerifier,
        default_top_k: int = 8,
        max_evidence_reads: int = 12,
    ) -> None:
        self.retrieval = retrieval
        self.resolver = resolver
        self.verifier = verifier
        self.default_top_k = default_top_k
        self.max_evidence_reads = max_evidence_reads
        self._evidence: list[Evidence] = []
        self._search_count = 0
        self._lock = threading.RLock()

    @property
    def evidence(self) -> list[Evidence]:
        with self._lock:
            return list(self._evidence)

    def reset(self) -> None:
        with self._lock:
            self._evidence.clear()
            self._search_count = 0

    @property
    def search_count(self) -> int:
        with self._lock:
            return self._search_count

    def evidence_by_id(self) -> dict[str, Evidence]:
        return {item.evidence_id: item for item in self.evidence}

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        limit = min(top_k or self.default_top_k, 20)
        results = self.retrieval.search(query, limit=limit)
        resolved = [
            self.verifier.verify_evidence(item)
            for item in self.resolver.resolve_many(query, results)
        ]
        with self._lock:
            self._evidence = merge_evidence(self._evidence, resolved)
            self._search_count += 1
        return [
            {
                "evidence_id": item.evidence_id,
                "source_file": item.source_file,
                "pages": [item.page_start, item.page_end],
                "section_path": item.section_path,
                "snippet": item.quote[:600],
            }
            for item in resolved
        ]

    def read(self, evidence_ids: Sequence[str]) -> list[dict]:
        known = self.evidence_by_id()
        requested = list(dict.fromkeys(evidence_ids))
        unknown = [item for item in requested if item not in known]
        if unknown:
            raise ValueError(f"Unknown evidence IDs: {', '.join(unknown)}")
        return [
            {
                "evidence_id": known[item].evidence_id,
                "source_file": known[item].source_file,
                "pages": [known[item].page_start, known[item].page_end],
                "section_path": known[item].section_path,
                "quote": known[item].quote,
                "quote_truncated": known[item].quote_truncated,
            }
            for item in requested
        ]

    def validate_evidence_ids(self, evidence_ids: Sequence[str]) -> None:
        known = self.evidence_by_id()
        unknown = set(evidence_ids) - set(known)
        if unknown:
            raise ValueError(f"Unknown evidence IDs: {sorted(unknown)}")

    def make_retrieval_tools(self) -> list[BaseTool]:
        workspace = self
        read_ids: set[str] = set()
        read_lock = threading.Lock()

        @tool(args_schema=SearchInput)
        def search_knowledge(query: str, top_k: int | None = None) -> str:
            """Search the local knowledge base and return ranked evidence previews."""

            return json.dumps(
                workspace.search(query, top_k), ensure_ascii=False
            )

        @tool(args_schema=ReadEvidenceInput)
        def read_evidence(evidence_ids: list[str]) -> str:
            """Read full source excerpts for selected evidence IDs."""

            requested = set(evidence_ids)
            with read_lock:
                new_ids = requested - read_ids
                if len(read_ids) + len(new_ids) > workspace.max_evidence_reads:
                    raise ValueError(
                        f"Evidence read budget exceeded ({workspace.max_evidence_reads})"
                    )
                read_ids.update(new_ids)
            return json.dumps(workspace.read(evidence_ids), ensure_ascii=False)

        return [search_knowledge, read_evidence]
