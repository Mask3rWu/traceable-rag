"""LangChain tools backed by the existing traceable retrieval layer."""
from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Sequence

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


class CheckTerminologyInput(BaseModel):
    content_blocks: list[dict] = Field(
        min_length=1, description="The draft ContentBlocks to scan, each with a markdown field"
    )
    decisions: list[dict] = Field(
        default_factory=list,
        description="Upstream decisions carrying glossary axes; each may have a glossary list",
    )


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
def extract_glossary(decisions: Sequence[dict]) -> list[dict]:
    """Flatten every decision's ``glossary`` into a list of axis dicts.

    Each returned dict has ``axis``, ``canonical_terms`` (set), and ``scope``.
    """
    axes: list[dict] = []
    for decision in decisions:
        for entry in decision.get("glossary") or []:
            terms = entry.get("canonical_terms") or []
            if not terms:
                continue
            axes.append(
                {
                    "axis": entry.get("axis", ""),
                    "canonical_terms": set(terms),
                    "scope": entry.get("scope", ""),
                    "terms": list(terms),
                }
            )
    return axes


def _axis_patterns(axis: dict) -> list[str]:
    """Build candidate-matching patterns for one glossary axis.

    Patterns are derived purely from the canonical terms so no domain vocabulary
    is hard-coded. Terms are grouped by shared single-character affix:

    * A suffix shared by >=2 terms (e.g. "级" in K级/M级, or "度" in
      轻度/中度/重度 even when a fourth term like 歼灭 lacks it) yields a
      pattern matching ``one preceding char + suffix`` -- i.e. a token shaped
      like the canonical terms.
    * A prefix shared by >=2 terms yields ``prefix + one following char``.

    The matched span is exactly the length of the shortest canonical term
    sharing that affix, so multi-word phrases are not over-matched. Axes whose
    terms share no affix produce no pattern -- drift on such axes is not
    detected, which is the accepted cost of not listing forbidden aliases.
    """
    terms = [t for t in axis["terms"] if t]
    if not terms:
        return []

    patterns: list[str] = []

    by_suffix: dict[str, list[str]] = {}
    for term in terms:
        by_suffix.setdefault(term[-1], []).append(term)
    for suffix, group in by_suffix.items():
        if len(group) < 2:
            continue
        # Span length: the canonical terms in this group, all sharing the suffix.
        span = min(len(t) for t in group)
        preceding = span - 1
        if preceding < 1:
            continue
        patterns.append(rf"[一-鿿A-Za-z]{{{preceding}}}{re.escape(suffix)}")

    by_prefix: dict[str, list[str]] = {}
    for term in terms:
        by_prefix.setdefault(term[0], []).append(term)
    for prefix, group in by_prefix.items():
        if len(group) < 2:
            continue
        span = min(len(t) for t in group)
        following = span - 1
        if following < 1:
            continue
        patterns.append(rf"{re.escape(prefix)}[一-鿿]{{{following}}}")

    return patterns


def scan_terminology(content_blocks: Sequence[dict], decisions: Sequence[dict]) -> list[dict]:
    """Return suspect terms found in ``content_blocks`` prose that match an axis
    pattern but are not in that axis's canonical set.

    This is advisory only: it never raises and may under-report terms that do
    not match any derived pattern. The empty list means "no suspect terms".

    Matches that occur as part of an axis name (e.g. "程度" inside "毁伤程度")
    are not flagged, since mentioning the axis itself is not a vocabulary drift.
    """
    axes = extract_glossary(decisions)
    if not axes:
        return []

    axis_names = [axis["axis"] for axis in axes if axis["axis"]]

    findings: list[dict] = []
    for block in content_blocks:
        markdown = block.get("markdown", "") if isinstance(block, dict) else ""
        if not markdown:
            continue
        for axis in axes:
            for pattern in _axis_patterns(axis):
                for match in re.finditer(pattern, markdown):
                    word = match.group(0)
                    if word in axis["canonical_terms"]:
                        continue
                    if any(word in name or name in word for name in axis_names):
                        continue
                    findings.append(
                        {
                            "axis": axis["axis"],
                            "term": word,
                            "canonical_terms": axis["terms"],
                        }
                    )
    # Deduplicate by (axis, term) preserving order.
    seen: set[tuple[str, str]] = set()
    unique: list[dict] = []
    for item in findings:
        key = (item["axis"], item["term"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


class EvidenceWorkspace:
    """Shared, thread-safe evidence registry for one top-level agent run."""

    def __init__(
        self,
        *,
        retrieval: RetrievalService,
        resolver: EvidenceResolver,
        verifier: CitationVerifier,
        default_top_k: int = 8,
        max_evidence_reads: int = 20,
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

    def make_retrieval_tools(
        self,
        aliases: EvidenceAliasRegistry | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> list[BaseTool]:
        workspace = self
        read_ids: set[str] = set()
        read_lock = threading.Lock()

        @tool(args_schema=SearchInput)
        def search_knowledge(query: str, top_k: int | None = None) -> str:
            """Search the local knowledge base and return ranked evidence previews."""
            if cancelled is not None and cancelled():
                raise RuntimeError("Research run was cancelled")
            results = workspace.search(query, top_k)
            if aliases is not None:
                results = [
                    {**item, "evidence_id": aliases.alias(item["evidence_id"])}
                    for item in results
                ]
            return json.dumps(results, ensure_ascii=False)

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

    @staticmethod
    def make_terminology_tool() -> BaseTool:
        """Build the advisory terminology self-check tool for a chapter worker.

        The tool is stateless: the worker passes its draft content_blocks and the
        upstream glossary decisions, and receives suspect terms (if any). It never
        raises and never blocks submission.
        """

        @tool(args_schema=CheckTerminologyInput)
        def check_terminology(
            content_blocks: list[dict], decisions: list[dict]
        ) -> str:
            """Compare draft prose against the upstream glossary.

            Pass your draft content_blocks and the glossary-bearing decisions from
            upstream. Returns a JSON object with a ``suspect_terms`` list: terms that
            look like they belong to a controlled-vocabulary axis but are not among
            that axis's canonical terms. Revise your prose to use only canonical
            terms when you agree a flagged term is a drift. This tool is advisory:
            it never blocks submission and may miss terms that match no axis pattern.
            """

            suspects = scan_terminology(content_blocks, decisions)
            return json.dumps(
                {
                    "suspect_terms": suspects,
                    "count": len(suspects),
                    "advisory": "Revise to canonical terms where appropriate; submission is never blocked.",
                },
                ensure_ascii=False,
            )

        return check_terminology
