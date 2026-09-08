"""Unit tests for runtime retrieval-quality metrics (route contribution/rescue)."""
from __future__ import annotations

import unittest

from src.research.retrieval_quality import (
    DEFAULT_DEPTH,
    aggregate_quality,
    cited_evidence,
    route_metrics,
)


def _evidence(evidence_id: str, chunk_id: str, traces: list[dict]) -> dict:
    return {"evidence_id": evidence_id, "chunk_id": chunk_id, "retrieval": traces}


def _run(cited: list[str], evidence: list[dict], packets: list[dict] | None = None) -> dict:
    return {
        "answer": {"evidence_ids": cited},
        "worker_packets": packets or [],
        "evidence": evidence,
    }


class CitedEvidenceTest(unittest.TestCase):
    def test_uses_last_trace_as_representative(self):
        run = _run(
            ["ev-a"],
            [
                _evidence(
                    "ev-a", "c1",
                    [
                        {"dense_rank": 50, "bm25_rank": 50},
                        {"dense_rank": 3, "bm25_rank": 7, "final_rank": 2},
                    ],
                )
            ],
        )
        items = cited_evidence(run)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].dense_rank, 3)
        self.assertEqual(items[0].bm25_rank, 7)
        self.assertEqual(items[0].chunk_id, "c1")

    def test_cited_from_answer_and_worker_packets(self):
        run = _run(
            ["ev-a"],
            [_evidence("ev-a", "c1", [{"dense_rank": 1, "bm25_rank": 2}])],
            packets=[{"evidence_ids": ["ev-b", "ev-c"]}],
        )
        run["evidence"].append(_evidence("ev-b", "c2", [{"dense_rank": 2, "bm25_rank": 1}]))
        run["evidence"].append(_evidence("ev-c", "c3", [{"dense_rank": 3, "bm25_rank": 3}]))
        ids = {item.evidence_id for item in cited_evidence(run)}
        self.assertEqual(ids, {"ev-a", "ev-b", "ev-c"})

    def test_missing_rank_kept_as_none(self):
        run = _run(
            ["ev-a"], [_evidence("ev-a", "c1", [{"dense_rank": 2, "bm25_rank": None}])]
        )
        items = cited_evidence(run)
        self.assertEqual(len(items), 1)
        self.assertIsNone(items[0].bm25_rank)

    def test_empty_result(self):
        self.assertEqual(cited_evidence(None), [])
        self.assertEqual(cited_evidence({}), [])


class RouteMetricsTest(unittest.TestCase):
    def test_rescue_classification_is_disjoint(self):
        run = _run(
            [],
            [
                _evidence("ev-a", "c1", [{"dense_rank": 1, "bm25_rank": 10}]),  # rescue_by_dense
                _evidence("ev-b", "c2", [{"dense_rank": 20, "bm25_rank": 1}]),  # rescue_by_bm25
                _evidence("ev-c", "c3", [{"dense_rank": 9, "bm25_rank": 9}]),  # fusion_only
                _evidence("ev-d", "c4", [{"dense_rank": 2, "bm25_rank": 3}]),  # both_in_depth
            ],
        )
        cited = ["ev-a", "ev-b", "ev-c", "ev-d"]
        run["answer"]["evidence_ids"] = cited
        q = route_metrics(run, depth=8)
        self.assertEqual(q.cited_total, 4)
        self.assertEqual(q.rescue_by_dense, 1)
        self.assertEqual(q.rescue_by_bm25, 1)
        self.assertEqual(q.fusion_only, 1)
        self.assertEqual(q.both_in_depth, 1)

    def test_both_beyond_depth_is_fusion_only_not_a_rescue(self):
        run = _run(
            ["ev-x"],
            [_evidence("ev-x", "c1", [{"dense_rank": 50, "bm25_rank": 50}])],
        )
        q = route_metrics(run, depth=8)
        self.assertEqual(q.fusion_only, 1)
        self.assertEqual(q.rescue_by_dense, 0)
        self.assertEqual(q.rescue_by_bm25, 0)

    def test_route_contribution_counts(self):
        run = _run(
            [],
            [
                _evidence("ev-a", "c1", [{"dense_rank": 1, "bm25_rank": 5}]),  # dense ahead
                _evidence("ev-b", "c2", [{"dense_rank": 6, "bm25_rank": 2}]),  # bm25 ahead
            ],
        )
        run["answer"]["evidence_ids"] = ["ev-a", "ev-b"]
        q = route_metrics(run, depth=8)
        self.assertEqual(q.dense_ahead, 1)
        self.assertEqual(q.bm25_ahead, 1)
        self.assertEqual(q.tie, 0)

    def test_missing_rank_is_skipped_not_zero(self):
        run = _run(
            [],
            [
                _evidence("ev-a", "c1", [{"dense_rank": 2, "bm25_rank": None}]),
                _evidence("ev-b", "c2", [{"dense_rank": 3, "bm25_rank": 4}]),
            ],
        )
        run["answer"]["evidence_ids"] = ["ev-a", "ev-b"]
        q = route_metrics(run, depth=8)
        self.assertEqual(q.cited_total, 1)  # ev-a skipped
        self.assertEqual(q.both_in_depth, 1)

    def test_none_result_yields_none_metrics(self):
        self.assertIsNone(route_metrics(None, depth=8))

    def test_default_depth_matches_production(self):
        self.assertEqual(DEFAULT_DEPTH, 8)


class AggregateQualityTest(unittest.TestCase):
    def test_sums_and_ignores_none(self):
        a = route_metrics(_run(["ev-a"], [_evidence("ev-a", "c1", [{"dense_rank": 1, "bm25_rank": 5}])]), depth=8)
        b = route_metrics(_run(["ev-b"], [_evidence("ev-b", "c2", [{"dense_rank": 9, "bm25_rank": 9}])]), depth=8)
        agg = aggregate_quality([a, None, b], depth=8)
        self.assertEqual(agg.cited_total, 2)
        self.assertEqual(agg.dense_ahead, 1)
        self.assertEqual(agg.fusion_only, 1)

    def test_empty_aggregate(self):
        agg = aggregate_quality([None], depth=8)
        self.assertEqual(agg.cited_total, 0)
        self.assertIsNone(agg.as_dict()["dense_ahead_frac"])


if __name__ == "__main__":
    unittest.main()