"""LangChain tools backed by the existing traceable retrieval layer."""
from __future__ import annotations

import json
import threading
from collections.abc import Callable, Sequence

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from src.research.evidence import EvidenceResolver, merge_evidence
from src.research.models import Evidence
from src.retrieval.service import RetrievalService


class SearchInput(BaseModel):
    query: str = Field(min_length=1, description="A focused knowledge-base query")
    top_k: int | None = Field(default=None, ge=1, le=20)


class ReadEvidenceInput(BaseModel):
    evidence_ids: list[str] = Field(min_length=1, max_length=4)


class EvidenceAliasRegistry:
    """Worker-local aliases for stable evidence IDs exposed to the model."""

    def __init__(self, aliases: dict[str, str] | None = None) -> None:
        self._alias_to_id = dict(aliases or {})
        self._id_to_alias = {value: key for key, value in self._alias_to_id.items()}
        self._lock = threading.RLock()

    def alias(self, evidence_id: str) -> str:
        with self._lock:
            existing = self._id_to_alias.get(evidence_id)
            if existing is not None:
                return existing
            alias = f"E{len(self._alias_to_id) + 1}"
            self._alias_to_id[alias] = evidence_id
            self._id_to_alias[evidence_id] = alias
            return alias

    def resolve(self, value: str) -> str:
        with self._lock:
            if value in self._id_to_alias:
                return value
            # Keep compatibility with persisted/test fixtures that already use
            # stable IDs; new model responses should use E1/E2 aliases.
            if value.lower().startswith("ev-"):
                return value
            try:
                return self._alias_to_id[value.upper()]
            except KeyError as exc:
                available = ", ".join(self._alias_to_id) or "none"
                raise ValueError(
                    f"Unknown evidence alias {value}; available aliases: {available}"
                ) from exc

    def export(self) -> dict[str, str]:
        with self._lock:
            return dict(self._alias_to_id)

    def restore(self, aliases: dict[str, str]) -> None:
        with self._lock:
            self._alias_to_id = dict(aliases)
            self._id_to_alias = {
                evidence_id: alias
                for alias, evidence_id in self._alias_to_id.items()
            }

    def translate_payload(self, value):
        """Translate evidence fields in a model-produced structured payload."""
        if isinstance(value, list):
            return [self.translate_payload(item) for item in value]
        if not isinstance(value, dict):
            return value
        translated = {}
        for key, item in value.items():
            if key == "evidence_id" and isinstance(item, str):
                translated[key] = self.resolve(item)
            elif key == "evidence_ids" and isinstance(item, list):
                translated[key] = [self.resolve(entry) for entry in item]
            else:
                translated[key] = self.translate_payload(item)
        return translated


class EvidenceWorkspace:
    """Shared, thread-safe evidence registry for one top-level agent run."""

    def __init__(
        self,
        *,
        retrieval: RetrievalService,
        resolver: EvidenceResolver,
        default_top_k: int = 8,
        max_evidence_reads: int = 20,
    ) -> None:
        self.retrieval = retrieval
        self.resolver = resolver
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

    def restore(self, evidence: Sequence[Evidence]) -> None:
        with self._lock:
            self._evidence = list(evidence)
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
        resolved = self.resolver.resolve_many(query, results)
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

    def make_retrieval_tools(
        self,
        aliases: EvidenceAliasRegistry | None = None,
        cancelled: Callable[[], bool] | None = None,
        search_limit: int | None = None,
    ) -> list[BaseTool]:
        """Build the ``search_knowledge`` / ``read_evidence`` tool pair.

        ``search_limit`` gates how many ``search_knowledge`` calls this tool pair
        may make before returning a soft ``budget_reached`` signal. It must be
        scoped per closure: the counter/lock below are captured fresh on every
        call, and `_build_chapter_graph` rebuilds tools per worker, so a parallel
        chapter worker gets its own budget. ``None`` keeps the historical
        unbounded behavior (used by the fast agent, which is already bounded by
        its step count).
        """
        workspace = self
        read_ids: set[str] = set()
        read_lock = threading.Lock()
        search_used = 0
        search_lock = threading.Lock()

        @tool(args_schema=SearchInput)
        def search_knowledge(query: str, top_k: int | None = None) -> str:
            """Search the local knowledge base and return ranked evidence previews.

            The result JSON carries ``new_uncovered``: how many of the returned
            evidence were not already in the shared pool. A low or zero count
            means this search added little and you should stop broad searching.
            """
            if cancelled is not None and cancelled():
                raise RuntimeError("Research run was cancelled")

            nonlocal search_used
            if search_limit is not None:
                with search_lock:
                    if search_used >= search_limit:
                        available = [
                            aliases.alias(item)
                            if aliases is not None
                            else item
                            for item in sorted(workspace.evidence_by_id())
                        ]
                        return json.dumps(
                            {
                                "status": "budget_reached",
                                "message": (
                                    f"Search budget reached ({search_limit}). "
                                    "Use the evidence already gathered to write and "
                                    "submit this chapter; do not perform further broad searches."
                                ),
                                "available_evidence_ids": available,
                            },
                            ensure_ascii=False,
                        )
                    search_used += 1

            known_before = workspace.evidence_by_id()
            results = workspace.search(query, top_k)
            new_ids = [
                item["evidence_id"]
                for item in results
                if item["evidence_id"] not in known_before
            ]
            if aliases is not None:
                results = [
                    {**item, "evidence_id": aliases.alias(item["evidence_id"])}
                    for item in results
                ]
            return json.dumps(
                {
                    "status": "ok",
                    "new_uncovered": len(new_ids),
                    "results": results,
                },
                ensure_ascii=False,
            )

        @tool(args_schema=ReadEvidenceInput)
        def read_evidence(evidence_ids: list[str]) -> str:
            """Read full source excerpts for selected evidence IDs."""
            if cancelled is not None and cancelled():
                raise RuntimeError("Research run was cancelled")
            stable_ids = (
                [aliases.resolve(item) for item in evidence_ids]
                if aliases is not None
                else evidence_ids
            )
            requested = set(stable_ids)
            with read_lock:
                new_ids = requested - read_ids
                if len(read_ids) + len(new_ids) > workspace.max_evidence_reads:
                    return json.dumps(
                        {
                            "status": "budget_reached",
                            "message": (
                                f"Evidence read limit reached ({workspace.max_evidence_reads}). "
                                "Use the evidence already read to write and submit this chapter."
                            ),
                            "available_evidence_ids": [
                                aliases.alias(item) if aliases is not None else item
                                for item in sorted(read_ids)
                            ],
                            "requested_in_budget": [
                                aliases.alias(item) if aliases is not None else item
                                for item in sorted(requested & read_ids)
                            ],
                        },
                        ensure_ascii=False,
                    )
                read_ids.update(new_ids)
            results = workspace.read(stable_ids)
            if aliases is not None:
                results = [
                    {**item, "evidence_id": aliases.alias(item["evidence_id"])}
                    for item in results
                ]
            return json.dumps(results, ensure_ascii=False)

        return [search_knowledge, read_evidence]
