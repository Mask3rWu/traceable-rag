from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.data_processing.relation_validation import (
    validate_relations,
    write_relation_validation_report,
)
from src.schema import Block


def block(block_id: str, block_type: str, section: list[str]) -> Block:
    return Block(
        block_id=block_id,
        document_id="doc",
        page=1,
        block_type=block_type,
        bbox=[0.1, 0.1, 0.9, 0.2],
        bbox_pixel=[10, 10, 90, 20],
        section_path=section,
    )


class RelationValidationTest(unittest.TestCase):
    def test_detects_cross_section_caption_and_writes_report(self):
        figure = block("figure", "figure", ["4", "4.3"])
        caption = block("caption", "caption", ["6"])
        figure.caption_ids = ["caption"]
        caption.caption_of = "figure"

        issues = validate_relations([figure, caption])

        self.assertEqual([item["code"] for item in issues], ["cross_section_caption"])
        self.assertIn("cross_section_caption", figure.quality_flags)
        self.assertIn("cross_section_caption", caption.quality_flags)
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "relation_validation.jsonl"
            write_relation_validation_report(report, issues)
            self.assertEqual(len(report.read_text(encoding="utf-8").splitlines()), 1)

    def test_detects_missing_target_without_flagging_cross_section_formula_reference(self):
        paragraph = block("paragraph", "paragraph", ["1"])
        formula = block("formula", "formula", ["2"])
        paragraph.references = ["formula", "missing"]

        codes = {item["code"] for item in validate_relations([paragraph, formula])}

        self.assertEqual(codes, {"invalid_relation:references"})


if __name__ == "__main__":
    unittest.main()
