from __future__ import annotations

import unittest

from src.research.evidence import EvidenceResolver, merge_evidence
from src.retrieval.catalog import ChunkCatalog
from src.retrieval.contracts import SearchResult
from src.retrieval.indexing import SourceChunk
from src.schema import Chunk, ChunkVisualAsset


def source_chunk() -> SourceChunk:
    return SourceChunk(
        chunk=Chunk(
            chunk_id="doc_C0001",
            document_id="doc",
            text="装甲破裂会降低结构防护能力。",
            embedding_text="装甲破裂会降低结构防护能力。",
            block_ids=["doc_P003_B01"],
            page_start=3,
            page_end=3,
            section_path=["3.1"],
            heading_path=["3 损伤", "3.1 装甲"],
            source_file="标准.pdf",
            visual_assets=[
                ChunkVisualAsset(
                    block_id="doc_P003_B02",
                    block_type="figure",
                    page=3,
                    status="available",
                )
            ],
        ),
        content_hash="a" * 64,
    )


def result(rank: int = 1) -> SearchResult:
    return SearchResult(
        chunk_id="doc_C0001",
        content_hash="a" * 64,
        bm25_rank=rank,
        bm25_score=8.0,
        final_rank=rank,
    )


class ResearchEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = ChunkCatalog([source_chunk()])
        self.resolver = EvidenceResolver(self.catalog)

    def test_resolves_and_merges_query_traces(self):
        first = self.resolver.resolve("装甲损伤", result())
        second = self.resolver.resolve("防护能力", result(rank=2))

        merged = merge_evidence([first], [second])

        self.assertEqual(len(merged), 1)
        self.assertEqual([item.query for item in merged[0].retrieval], ["装甲损伤", "防护能力"])
        self.assertEqual(merged[0].section_path, ["3 损伤", "3.1 装甲"])
        self.assertEqual(merged[0].visual_assets[0].page, 3)

    def test_rejects_stale_retrieval_result(self):
        stale = result().model_copy(update={"content_hash": "b" * 64})
        with self.assertRaisesRegex(RuntimeError, "Stale retrieval index"):
            self.resolver.resolve("装甲损伤", stale)
