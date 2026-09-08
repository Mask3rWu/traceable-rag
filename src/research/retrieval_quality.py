"""Read-only retrieval-quality metrics derived from a persisted AgentRun.

Metrics are computed over the evidence the agent actually cited (a weak gold
signal) using the single-route ranks already persisted in each evidence's
retrieval trace. No retrieval is re-run and no production state is read or
written: this is a pure, structural summary of one run's outcome.

Three families are exposed:

- route contribution: for cited evidence, how often Dense vs BM25 ranked it
  better (dense_ahead / bm25_ahead / tie);
- single-route rescue: cited evidence that only one route would surface within
  the production retrieval depth (rescue_by_dense / rescue_by_bm25), or that
  both routes place beyond depth and only fusion surfaces (fusion_only);
- agreement / redundancy: cited evidence both routes place within depth
  (both_in_depth) — would survive either route alone.

Production depth is read from the research config default rather than a local
literal, and can be overridden per call.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.config import ResearchModelConfig

#: Production fused depth the research agent consumes downstream (config, not a
#: hardcoded literal). Overridable per call via ``depth``.
DEFAULT_DEPTH = ResearchModelConfig.retrieval_top_k


@dataclass(frozen=True)
class CitedEvidence:
    """One cited evidence plus the single-route ranks of its representative trace."""

    evidence_id: str
    chunk_id: str | None
    dense_rank: int | None
    bm25_rank: int | None
    final_rank: int | None


@dataclass(frozen=True)
class RouteQuality:
    """Raw route-metric counts for one run. ``cited_total`` is the denominator."""

    depth: int
    cited_total: int
    dense_ahead: int
    bm25_ahead: int
    tie: int
    rescue_by_dense: int
    rescue_by_bm25: int
    fusion_only: int
    both_in_depth: int

    def as_dict(self) -> dict:
        n = self.cited_total

        def frac(value: int) -> float | None:
            return round(value / n, 4) if n else None

        return {
            "depth": self.depth,
            "cited_total": n,
            "dense_ahead": self.dense_ahead,
            "bm25_ahead": self.bm25_ahead,
            "tie": self.tie,
            "rescue_by_dense": self.rescue_by_dense,
            "rescue_by_bm25": self.rescue_by_bm25,
            "fusion_only": self.fusion_only,
            "both_in_depth": self.both_in_depth,
            "dense_ahead_frac": frac(self.dense_ahead),
            "bm25_ahead_frac": frac(self.bm25_ahead),
            "rescue_by_dense_frac": frac(self.rescue_by_dense),
            "rescue_by_bm25_frac": frac(self.rescue_by_bm25),
            "fusion_only_frac": frac(self.fusion_only),
            "both_in_depth_frac": frac(self.both_in_depth),
        }


def cited_evidence(result: dict | None) -> list[CitedEvidence]:
    """Collect evidence cited by the agent (answer + worker packets).

    Uses a representative retrieval trace per cited evidence: the LAST trace,
    matching the agent's final evidence state. Evidence without a usable rank
    pair is retained with ``None`` ranks so downstream metric code can skip it
    explicitly rather than count a zero.
    """
    if not result:
        return []
    cited_ids = set((result.get("answer") or {}).get("evidence_ids") or [])
    for packet in result.get("worker_packets") or []:
        cited_ids.update(packet.get("evidence_ids") or [])
    by_id = {item["evidence_id"]: item for item in (result.get("evidence") or [])}

    def representative(evidence: dict) -> dict | None:
        retrieval = evidence.get("retrieval") or []
        return retrieval[-1] if retrieval else None

    out: list[CitedEvidence] = []
    for evidence_id in cited_ids:
        evidence = by_id.get(evidence_id)
        if not evidence:
            continue
        trace = representative(evidence)
        out.append(
            CitedEvidence(
                evidence_id=evidence_id,
                chunk_id=evidence.get("chunk_id"),
                dense_rank=trace.get("dense_rank") if trace else None,
                bm25_rank=trace.get("bm25_rank") if trace else None,
                final_rank=trace.get("final_rank") if trace else None,
            )
        )
    return out


def route_metrics(result: dict | None, *, depth: int | None = None) -> RouteQuality | None:
    """Compute route-quality counts for one run, or ``None`` when unusable.

    Evidence missing either single-route rank is skipped rather than counted,
    so ``cited_total`` is the number of cited evidence with a usable rank pair.
    The four depth categories are mutually exclusive and exhaustive over that
    set.
    """
    if result is None:
        return None
    resolved = depth if depth is not None else DEFAULT_DEPTH
    rows = [
        item
        for item in cited_evidence(result)
        if item.dense_rank is not None and item.bm25_rank is not None
    ]
    return RouteQuality(
        depth=resolved,
        cited_total=len(rows),
        dense_ahead=sum(item.dense_rank < item.bm25_rank for item in rows),
        bm25_ahead=sum(item.bm25_rank < item.dense_rank for item in rows),
        tie=sum(item.dense_rank == item.bm25_rank for item in rows),
        rescue_by_dense=sum(item.bm25_rank > resolved and item.dense_rank <= resolved for item in rows),
        rescue_by_bm25=sum(item.dense_rank > resolved and item.bm25_rank <= resolved for item in rows),
        fusion_only=sum(item.dense_rank > resolved and item.bm25_rank > resolved for item in rows),
        both_in_depth=sum(item.dense_rank <= resolved and item.bm25_rank <= resolved for item in rows),
    )


def aggregate_quality(
    per_question: Sequence[RouteQuality | None], *, depth: int | None = None
) -> RouteQuality:
    """Sum per-question route metrics into a batch-level summary.

    ``None`` entries (runs without usable evidence) contribute nothing rather
    than a zero-scored term.
    """
    resolved = depth if depth is not None else DEFAULT_DEPTH
    counts = {field: 0 for field in ("cited_total", "dense_ahead", "bm25_ahead", "tie",
                                     "rescue_by_dense", "rescue_by_bm25", "fusion_only",
                                     "both_in_depth")}
    for item in per_question:
        if item is None:
            continue
        for field in counts:
            counts[field] += getattr(item, field)
    return RouteQuality(depth=resolved, **counts)