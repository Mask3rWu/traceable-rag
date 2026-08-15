"""Tests for the advisory terminology self-check used by chapter workers."""
from __future__ import annotations

import json
import unittest

from src.research.tools import extract_terms_contracts, scan_terminology


class ExtractTermsContractsTest(unittest.TestCase):
    """extract_terms_contracts flattens every type==\"terms\" contract."""

    def test_flattens_terms_contracts_from_list(self):
        contracts = [
            {
                "contract_id": "AXIS-KILL",
                "type": "terms",
                "canonical_terms": ["K级", "M级", "F级"],
            },
            {
                "contract_id": "AXIS-DAMAGE",
                "type": "terms",
                "canonical_terms": ["轻度", "中度", "重度"],
            },
            {"contract_id": "TH-1", "type": "threshold", "canonical_terms": []},
            {"contract_id": "no_type"},
        ]
        axes = extract_terms_contracts(contracts)

        self.assertEqual(len(axes), 2)
        self.assertEqual(axes[0]["axis"], "AXIS-KILL")
        self.assertEqual(axes[0]["canonical_terms"], {"K级", "M级", "F级"})
        self.assertEqual(axes[1]["axis"], "AXIS-DAMAGE")

    def test_empty_when_no_terms_contracts(self):
        self.assertEqual(extract_terms_contracts([]), [])
        self.assertEqual(
            extract_terms_contracts([{"contract_id": "x", "type": "threshold"}]),
            [],
        )

    def test_skips_terms_contract_without_canonical_terms(self):
        result = extract_terms_contracts(
            [{"contract_id": "AXIS", "type": "terms", "canonical_terms": []}]
        )
        self.assertEqual(result, [])


class ScanTerminologyTest(unittest.TestCase):
    def _contracts(self):
        return [
            {
                "contract_id": "AXIS-KILL",
                "type": "terms",
                "canonical_terms": ["K级", "M级", "F级", "C级", "P级"],
            },
            {
                "contract_id": "AXIS-DAMAGE",
                "type": "terms",
                "canonical_terms": ["轻度", "中度", "重度", "歼灭"],
            },
        ]

    def test_no_contracts_returns_empty(self):
        result = scan_terminology("任意文字", [])
        self.assertEqual(result, [])

    def test_canonical_terms_not_flagged(self):
        prose = "该目标判定为K级，毁伤为重度。M级与F级均可出现，歼灭为最高。"
        result = scan_terminology(prose, self._contracts())
        self.assertEqual(result, [])

    def test_suffix_axis_flags_non_canonical_term(self):
        # "Q级" matches the "X级" suffix pattern but is not in the canonical set.
        result = scan_terminology("该目标判定为Q级。", self._contracts())
        self.assertTrue(any(item["term"] == "Q级" for item in result))
        suspect = next(item for item in result if item["term"] == "Q级")
        self.assertEqual(suspect["axis"], "AXIS-KILL")

    def test_non_matching_term_not_flagged(self):
        # "严重" does not match the "X级" suffix pattern nor any shared-stem
        # pattern of the AXIS-DAMAGE axis (轻度/中度/重度/歼灭 share no affix),
        # so it must not be flagged -- the core "no forbidden aliases" property.
        result = scan_terminology("该目标严重受损。", self._contracts())
        self.assertNotIn("严重", [item["term"] for item in result])

    def test_shared_suffix_variant_flagged(self):
        # "极度" shares the "度" suffix with 轻度/中度/重度 but is not canonical.
        result = scan_terminology("该目标毁伤为极度。", self._contracts())
        self.assertIn(
            "极度", [item["term"] for item in result if item["axis"] == "AXIS-DAMAGE"]
        )

    def test_dimension_word_matching_affix_pattern_is_flagged(self):
        # After the refactor the axis is a contract_id (e.g. AXIS-DAMAGE), not a
        # human-readable name, so the old "axis-name substring is not drift"
        # protection no longer applies. A dimension word like "程度" that matches
        # the shared "度" suffix is now flagged -- the accepted mirror of the
        # affix heuristic's under-detection on no-affix axes.
        result = scan_terminology("毁伤程度按规范分为四级。", self._contracts())
        self.assertIn("程度", [item["term"] for item in result])

    def test_deduplicates_repeated_suspect_terms(self):
        result = scan_terminology("Q级与Q级相同，均为Q级。", self._contracts())
        terms = [item["term"] for item in result if item["term"] == "Q级"]
        self.assertEqual(len(terms), 1)

    def test_empty_prose_skipped(self):
        result = scan_terminology("", self._contracts())
        self.assertEqual(result, [])


class CheckTerminologyToolTest(unittest.TestCase):
    """The tool wraps scan_terminology and must never raise or block submission."""

    def test_tool_returns_json_with_suspect_terms(self):
        from src.research.tools import EvidenceWorkspace

        check_terminology = EvidenceWorkspace.make_terminology_tool()
        contracts = [
            {"contract_id": "AXIS-KILL", "type": "terms", "canonical_terms": ["K级", "M级"]},
        ]
        raw = check_terminology.invoke({"prose": "判定为Q级", "contracts": contracts})
        parsed = json.loads(raw)
        self.assertIn("suspect_terms", parsed)
        self.assertTrue(any(item["term"] == "Q级" for item in parsed["suspect_terms"]))
        self.assertIn("advisory", parsed)

    def test_tool_never_raises_on_empty_contracts(self):
        from src.research.tools import EvidenceWorkspace

        check_terminology = EvidenceWorkspace.make_terminology_tool()
        raw = check_terminology.invoke({"prose": "任意文字", "contracts": []})
        parsed = json.loads(raw)
        self.assertEqual(parsed["suspect_terms"], [])
        self.assertEqual(parsed["count"], 0)


if __name__ == "__main__":
    unittest.main()