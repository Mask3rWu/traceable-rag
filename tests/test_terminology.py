"""Tests for the advisory terminology self-check used by chapter workers."""
from __future__ import annotations

import json
import unittest

from src.research.tools import extract_glossary, scan_terminology


class ExtractGlossaryTest(unittest.TestCase):
    def test_flattens_glossary_from_multiple_decisions(self):
        decisions = [
            {
                "decision_id": "terminology",
                "glossary": [
                    {"axis": "杀伤等级", "canonical_terms": ["K级", "M级", "F级"]},
                    {"axis": "毁伤程度", "canonical_terms": ["轻度", "中度", "重度"]},
                ],
            },
            {"decision_id": "other", "glossary": []},
            {"decision_id": "no_glossary_field"},
        ]
        axes = extract_glossary(decisions)

        self.assertEqual(len(axes), 2)
        self.assertEqual(axes[0]["axis"], "杀伤等级")
        self.assertEqual(axes[0]["canonical_terms"], {"K级", "M级", "F级"})
        self.assertEqual(axes[1]["axis"], "毁伤程度")

    def test_empty_when_no_glossaries(self):
        self.assertEqual(extract_glossary([]), [])
        self.assertEqual(extract_glossary([{"decision_id": "x"}]), [])


class ScanTerminologyTest(unittest.TestCase):
    def _glossary(self):
        return [
            {
                "decision_id": "terminology",
                "glossary": [
                    {"axis": "杀伤等级", "canonical_terms": ["K级", "M级", "F级", "C级", "P级"]},
                    {"axis": "毁伤程度", "canonical_terms": ["轻度", "中度", "重度", "歼灭"]},
                ],
            }
        ]

    def test_no_glossary_returns_empty(self):
        result = scan_terminology([{"markdown": "任意文字"}], [])
        self.assertEqual(result, [])

    def test_canonical_terms_not_flagged(self):
        decisions = self._glossary()
        markdown = "该目标判定为K级，毁伤程度为重度。M级与F级均可出现，歼灭为最高。"
        result = scan_terminology([{"markdown": markdown}], decisions)
        self.assertEqual(result, [])

    def test_suffix_axis_flags_non_canonical_term(self):
        decisions = self._glossary()
        # "Q级" matches the "X级" suffix pattern but is not in the canonical set.
        markdown = "该目标判定为Q级。"
        result = scan_terminology([{"markdown": markdown}], decisions)
        self.assertTrue(any(item["term"] == "Q级" for item in result))
        suspect = next(item for item in result if item["term"] == "Q级")
        self.assertEqual(suspect["axis"], "杀伤等级")

    def test_non_matching_term_not_flagged(self):
        decisions = self._glossary()
        # "严重" does not match the "X级" suffix pattern nor any shared-stem
        # pattern of the 毁伤程度 axis (轻度/中度/重度/歼灭 share no affix),
        # so it must not be flagged -- the core "no forbidden aliases" property.
        markdown = "该目标严重受损。"
        result = scan_terminology([{"markdown": markdown}], decisions)
        self.assertNotIn("严重", [item["term"] for item in result])

    def test_shared_suffix_variant_flagged(self):
        decisions = self._glossary()
        # "极度" shares the "度" suffix with 轻度/中度/重度 but is not canonical.
        markdown = "该目标毁伤为极度。"
        result = scan_terminology([{"markdown": markdown}], decisions)
        self.assertIn(
            "极度", [item["term"] for item in result if item["axis"] == "毁伤程度"]
        )

    def test_axis_name_substring_not_flagged(self):
        decisions = self._glossary()
        # "程度" is part of the axis name "毁伤程度"; mentioning the axis is not drift.
        markdown = "毁伤程度按规范分为四级。"
        result = scan_terminology([{"markdown": markdown}], decisions)
        self.assertNotIn("程度", [item["term"] for item in result])

    def test_deduplicates_repeated_suspect_terms(self):
        decisions = self._glossary()
        markdown = "Q级与Q级相同，均为Q级。"
        result = scan_terminology([{"markdown": markdown}], decisions)
        terms = [item["term"] for item in result if item["term"] == "Q级"]
        self.assertEqual(len(terms), 1)

    def test_empty_markdown_skipped(self):
        decisions = self._glossary()
        result = scan_terminology([{"markdown": ""}], decisions)
        self.assertEqual(result, [])

    def test_never_raises_on_malformed_block(self):
        decisions = self._glossary()
        # Should not raise even if a block is missing markdown.
        result = scan_terminology([{}], decisions)
        self.assertEqual(result, [])


class CheckTerminologyToolTest(unittest.TestCase):
    """The tool wraps scan_terminology and must never raise or block submission."""

    def test_tool_returns_json_with_suspect_terms(self):
        from src.research.tools import EvidenceWorkspace

        check_terminology = EvidenceWorkspace.make_terminology_tool()
        decisions = [
            {
                "decision_id": "terminology",
                "glossary": [
                    {"axis": "杀伤等级", "canonical_terms": ["K级", "M级"]},
                ],
            }
        ]
        content_blocks = [{"markdown": "判定为Q级"}]
        raw = check_terminology.invoke(
            {"content_blocks": content_blocks, "decisions": decisions}
        )
        parsed = json.loads(raw)
        self.assertIn("suspect_terms", parsed)
        self.assertTrue(any(item["term"] == "Q级" for item in parsed["suspect_terms"]))
        self.assertIn("advisory", parsed)

    def test_tool_never_raises_on_empty_glossary(self):
        from src.research.tools import EvidenceWorkspace

        check_terminology = EvidenceWorkspace.make_terminology_tool()
        raw = check_terminology.invoke(
            {"content_blocks": [{"markdown": "任意文字"}], "decisions": []}
        )
        parsed = json.loads(raw)
        self.assertEqual(parsed["suspect_terms"], [])
        self.assertEqual(parsed["count"], 0)


if __name__ == "__main__":
    unittest.main()
