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


if __name__ == "__main__":
    unittest.main()
