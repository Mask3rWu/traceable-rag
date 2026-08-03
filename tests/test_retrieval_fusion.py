from __future__ import annotations

import unittest

from src.retrieval.contracts import SearchResult
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import NoopReranker
from src.retrieval.service import RetrievalOptions


def dense(chunk_id: str, rank: int, content_hash: str = "hash") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        content_hash=content_hash,
        dense_rank=rank,
        dense_score=1 - rank / 10,
        final_rank=rank,
    )


def bm25(chunk_id: str, rank: int, content_hash: str = "hash") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        content_hash=content_hash,
        bm25_rank=rank,
        bm25_score=10 - rank,
        final_rank=rank,
    )


class FusionTest(unittest.TestCase):
    def test_rrf_merges_channel_scores_and_ranks(self):
        results = reciprocal_rank_fusion(
            [dense("shared", 1), dense("dense-only", 2)],
            [bm25("bm25-only", 1), bm25("shared", 2)],
            rank_constant=60,
        )

        self.assertEqual(results[0].chunk_id, "shared")
        self.assertEqual(results[0].dense_rank, 1)
        self.assertEqual(results[0].bm25_rank, 2)
        self.assertEqual([result.final_rank for result in results], [1, 2, 3])

    def test_rrf_rejects_stale_channel_index(self):
        with self.assertRaisesRegex(RuntimeError, "Index version mismatch"):
            reciprocal_rank_fusion(
                [dense("shared", 1, "old")], [bm25("shared", 1, "new")]
            )

    def test_noop_reranker_preserves_order_and_applies_limit(self):
        candidates = reciprocal_rank_fusion(
            [dense("first", 1), dense("second", 2)], []
        )
        results = NoopReranker().rerank("query", candidates, limit=1)
        self.assertEqual([result.chunk_id for result in results], ["first"])

    def test_retrieval_options_reject_invalid_limits(self):
        with self.assertRaisesRegex(ValueError, "limits"):
            RetrievalOptions(fusion_limit=0)
