"""Resolve retrieval hits into evidence and validate source citations."""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

from src.research.models import Evidence, EvidenceVisual, RetrievalTrace
from src.retrieval.catalog import ChunkCatalog
from src.retrieval.contracts import SearchResult


def _evidence_id(chunk_id: str) -> str:
    digest = hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()[:12]
    return f"ev-{digest}"


class EvidenceResolver:
    def __init__(self, catalog: ChunkCatalog, *, quote_chars: int = 4000) -> None:
        if quote_chars <= 0:
            raise ValueError("quote_chars must be greater than zero")
        self.catalog = catalog
        self.quote_chars = quote_chars

    def resolve(self, query: str, result: SearchResult) -> Evidence:
        if not query.strip():
            raise ValueError("query must not be blank")
        self.catalog.validate_result(result)
        chunk = self.catalog.source(result.chunk_id).chunk
        quote = chunk.text[: self.quote_chars]
        rank = result.final_rank or result.dense_rank or result.bm25_rank
        if rank is None:
            raise ValueError(f"Retrieval result has no rank: {result.chunk_id}")
        return Evidence(
            evidence_id=_evidence_id(chunk.chunk_id),
            chunk_id=chunk.chunk_id,
            content_hash=result.content_hash,
            document_id=chunk.document_id,
            source_file=chunk.source_file,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            section_path=chunk.heading_path or chunk.section_path,
            block_ids=chunk.block_ids,
            quote=quote,
            quote_truncated=len(quote) < len(chunk.text),
            visual_assets=[
                EvidenceVisual.model_validate(asset.model_dump())
                for asset in chunk.visual_assets
            ],
            retrieval=[
                RetrievalTrace(
                    query=query,
                    final_rank=rank,
                    dense_rank=result.dense_rank,
                    dense_score=result.dense_score,
                    bm25_rank=result.bm25_rank,
                    bm25_score=result.bm25_score,
                    fusion_score=result.fusion_score,
                )
            ],
        )

    def resolve_many(
        self, query: str, results: Sequence[SearchResult]
    ) -> list[Evidence]:
        return [self.resolve(query, result) for result in results]


def merge_evidence(current: Sequence[Evidence], incoming: Sequence[Evidence]) -> list[Evidence]:
    """Deduplicate chunks while retaining every query/rank observation."""

    merged = {item.evidence_id: item for item in current}
    order = [item.evidence_id for item in current]
    for item in incoming:
        existing = merged.get(item.evidence_id)
        if existing is None:
            merged[item.evidence_id] = item
            order.append(item.evidence_id)
            continue
        if existing.content_hash != item.content_hash:
            raise RuntimeError(f"Evidence version mismatch for {item.chunk_id}")
        traces = list(existing.retrieval)
        known = {(trace.query, trace.final_rank) for trace in traces}
        traces.extend(
            trace
            for trace in item.retrieval
            if (trace.query, trace.final_rank) not in known
        )
        merged[item.evidence_id] = existing.model_copy(update={"retrieval": traces})
    return [merged[evidence_id] for evidence_id in order]
