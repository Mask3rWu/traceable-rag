from __future__ import annotations

import unittest

from src.retrieval.catalog import ChunkCatalog
from src.retrieval.contracts import SearchResult
from src.retrieval.evaluation import EvaluationCase, evaluate_rankings
from src.retrieval.indexing import SourceChunk
from src.schema import Chunk


def source(chunk_id: str, block_id: str) -> SourceChunk:
    return SourceChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="doc",
            source_file="doc.pdf",
            text=block_id,
            embedding_text=block_id,
            block_ids=[block_id],
            page_start=1,
            page_end=1,
        ),
        content_hash=f"hash-{chunk_id}",
    )


class EvaluationTest(unittest.TestCase):
    def test_multi_evidence_coverage_and_complete_recall(self):
        catalog = ChunkCatalog([source("one", "block-1"), source("two", "block-2")])
        case = EvaluationCase(
            case_id="doc:Q0001",
            document_id="doc",
            question_id="Q0001",
            question="question",
            evidence_block_ids=frozenset({"block-1", "block-2"}),
            source_granularity="none",
        )
        rankings = [[
            SearchResult(chunk_id="one", content_hash="hash-one", final_rank=1),
            SearchResult(chunk_id="two", content_hash="hash-two", final_rank=2),
        ]]

        at_one, at_two = evaluate_rankings([case], rankings, catalog, ks=[1, 2])

        self.assertEqual(at_one.evidence_recall, 0.5)
        self.assertEqual(at_one.complete_recall, 0.0)
        self.assertEqual(at_one.mrr, 1.0)
        self.assertEqual(at_two.evidence_recall, 1.0)
        self.assertEqual(at_two.complete_recall, 1.0)

    def test_rejects_stale_result(self):
        catalog = ChunkCatalog([source("one", "block-1")])
        case = EvaluationCase(
            "doc:Q0001", "doc", "Q0001", "question", frozenset({"block-1"}), "none"
        )
        rankings = [[SearchResult(chunk_id="one", content_hash="stale")]]
        with self.assertRaisesRegex(RuntimeError, "Stale retrieval index"):
            evaluate_rankings([case], rankings, catalog, ks=[1])

    def test_excludes_unanswerable_cases_from_retrieval_metrics(self):
        catalog = ChunkCatalog([source("one", "block-1")])
        answerable = EvaluationCase(
            "doc:Q0001", "doc", "Q0001", "answer", frozenset({"block-1"}), "none"
        )
        unanswerable = EvaluationCase(
            "doc:Q0002", "doc", "Q0002", "no answer", frozenset(), "none", "unanswerable"
        )
        ranking = [[SearchResult(chunk_id="one", content_hash="hash-one", final_rank=1)], []]

        summary = evaluate_rankings([answerable, unanswerable], ranking, catalog, ks=[1])[0]

        self.assertEqual(summary.evidence_recall, 1.0)
        self.assertEqual(summary.complete_recall, 1.0)
