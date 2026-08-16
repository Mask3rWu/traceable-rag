from __future__ import annotations

import json
import unittest

from src.research.tools import EvidenceAliasRegistry, EvidenceWorkspace


class _ToolWorkspace:
    max_evidence_reads = 1

    def read(self, evidence_ids: list[str]) -> list[dict]:
        return [
            {"evidence_id": evidence_id, "quote": "source"}
            for evidence_id in evidence_ids
        ]


class _SearchWorkspace(_ToolWorkspace):
    def __init__(self) -> None:
        self._pool: dict[str, dict] = {}
        self.search_count = 0

    def evidence_by_id(self) -> dict[str, dict]:
        return dict(self._pool)

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        # Each call surfaces one fresh evidence and returns the whole (deduped)
        # pool, mirroring merge_evidence, so a repeat query reports new=0.
        self.search_count += 1
        self._pool[f"ev-{query}"] = {"evidence_id": f"ev-{query}"}
        return [
            {"evidence_id": eid, "quote": "source"}
            for eid in sorted(self._pool)
        ]


class ResearchToolsTest(unittest.TestCase):
    def test_evidence_alias_registry_is_stable_and_translates_payloads(self):
        aliases = EvidenceAliasRegistry()
        self.assertEqual(aliases.alias("ev-one"), "E1")
        self.assertEqual(aliases.alias("ev-two"), "E2")
        self.assertEqual(aliases.alias("ev-one"), "E1")
        self.assertEqual(
            aliases.translate_payload(
                {"evidence_ids": ["E2"], "citations": [{"evidence_id": "E1"}]}
            ),
            {
                "evidence_ids": ["ev-two"],
                "citations": [{"evidence_id": "ev-one"}],
            },
        )
        restored = EvidenceAliasRegistry()
        restored.restore(aliases.export())
        self.assertEqual(restored.resolve("E1"), "ev-one")
        with self.assertRaisesRegex(ValueError, "Unknown evidence alias E9"):
            aliases.resolve("E9")

    def test_read_budget_returns_soft_signal_and_keeps_read_ids_available(self):
        workspace = _ToolWorkspace()
        tools = EvidenceWorkspace.make_retrieval_tools(
            workspace  # type: ignore[arg-type]
        )
        read_evidence = next(item for item in tools if item.name == "read_evidence")

        first = json.loads(read_evidence.invoke({"evidence_ids": ["ev-1"]}))
        exhausted = json.loads(read_evidence.invoke({"evidence_ids": ["ev-2"]}))
        reread = json.loads(read_evidence.invoke({"evidence_ids": ["ev-1"]}))

        self.assertEqual(first[0]["evidence_id"], "ev-1")
        self.assertEqual(exhausted["status"], "budget_reached")
        self.assertEqual(exhausted["available_evidence_ids"], ["ev-1"])
        self.assertEqual(reread[0]["evidence_id"], "ev-1")

    def test_search_budget_reports_reached_after_limit(self):
        ws = _SearchWorkspace()
        tools = EvidenceWorkspace.make_retrieval_tools(ws, search_limit=2)
        search_knowledge = next(
            item for item in tools if item.name == "search_knowledge"
        )

        first = json.loads(search_knowledge.invoke({"query": "a"}))
        second = json.loads(search_knowledge.invoke({"query": "b"}))
        exhausted = json.loads(search_knowledge.invoke({"query": "c"}))

        self.assertEqual(first["status"], "ok")
        self.assertEqual(second["status"], "ok")
        # The third calls sites budget_reached and stops broad searching.
        self.assertEqual(exhausted["status"], "budget_reached")
        self.assertEqual(ws.search_count, 2)
        self.assertEqual(
            sorted(exhausted["available_evidence_ids"]), ["ev-a", "ev-b"]
        )

    def test_search_new_uncovered_counts_fresh_evidence(self):
        ws = _SearchWorkspace()
        tools = EvidenceWorkspace.make_retrieval_tools(ws)
        search_knowledge = next(
            item for item in tools if item.name == "search_knowledge"
        )

        fresh = json.loads(search_knowledge.invoke({"query": "a"}))
        repeat = json.loads(search_knowledge.invoke({"query": "a"}))

        self.assertEqual(fresh["status"], "ok")
        self.assertEqual(fresh["new_uncovered"], 1)
        self.assertEqual(repeat["new_uncovered"], 0)
        self.assertEqual([r["evidence_id"] for r in repeat["results"]], ["ev-a"])


if __name__ == "__main__":
    unittest.main()
