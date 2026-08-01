from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.data_processing.visual_enrichment import (
    _context_for,
    _trim_description,
    enrich_document,
    parse_model_output,
)


class FakeClient:
    def describe(self, image_path: Path, context: str) -> str:
        assert image_path.is_file()
        assert "图注" in context
        assert "相邻正文" in context
        return "展示两个部件之间的连接关系。"


class VisualEnrichmentTest(unittest.TestCase):
    def test_parse_model_output_returns_plain_description_and_accepts_legacy_json(self):
        self.assertEqual(parse_model_output("显示趋势。"), "显示趋势。")
        self.assertEqual(parse_model_output('```json\n{"visual_type":"chart","description":"显示趋势。"}\n```'), "显示趋势。")
        with self.assertRaises(ValueError):
            parse_model_output("   ")

    def test_description_limit_keeps_a_complete_sentence(self):
        description = "第一句。" + "第二句内容较长" * 30
        result = _trim_description(description)
        self.assertLessEqual(len(result), 150)
        self.assertEqual(result, "第一句。")

    def test_enrich_document_writes_traceable_results_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            doc_dir = root / "processed" / "parsed" / "sample"
            doc_dir.mkdir(parents=True)
            crop = doc_dir / "assets" / "crops" / "figure.png"
            crop.parent.mkdir(parents=True)
            crop.write_bytes(b"png")
            doc = {
                "document_id": "sample",
                "pages": [{"page": 1, "blocks": [
                    {"block_id": "h", "block_type": "heading", "text": "1 范围", "section_path": ["1"], "page": 1, "order": 1},
                    {"block_id": "p", "block_type": "paragraph", "text": "正文说明图示流程。", "section_path": ["1"], "page": 1, "order": 2, "references": ["fig"]},
                    {"block_id": "fig", "block_type": "figure", "text": "", "section_path": ["1"], "page": 1, "order": 3, "image_crop": "processed/parsed/sample/assets/crops/figure.png", "caption_ids": ["cap"]},
                    {"block_id": "cap", "block_type": "caption", "text": "图1 流程示意", "section_path": ["1"], "page": 1, "order": 4},
                ]}],
            }
            doc_path = doc_dir / "doc.json"
            doc_path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            result = enrich_document(doc_path, FakeClient())
            self.assertEqual(result["items"][0]["status"], "ok")
            self.assertEqual(result["items"][0]["block_id"], "fig")
            self.assertTrue((doc_dir / "visual_enrichment.json").is_file())
            self.assertEqual(enrich_document(doc_path, FakeClient())["items"][0]["status"], "ok")

    def test_context_contains_caption_and_referencing_text(self):
        blocks = [
            {"block_id": "p", "block_type": "paragraph", "text": "引用图1。", "section_path": ["1"], "page": 1, "order": 1, "references": ["fig"]},
            {"block_id": "fig", "block_type": "figure", "text": "", "section_path": ["1"], "page": 1, "order": 2, "caption_ids": ["cap"]},
            {"block_id": "cap", "block_type": "caption", "text": "图1 示例", "section_path": ["1"], "page": 1, "order": 3},
        ]
        context = _context_for(blocks[1], blocks, {b["block_id"]: b for b in blocks})
        self.assertIn("图1 示例", context)
        self.assertIn("引用图1", context)


if __name__ == "__main__":
    unittest.main()
