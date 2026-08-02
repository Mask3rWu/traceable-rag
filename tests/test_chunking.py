from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.data_processing.chunking import ChunkConfig, build_chunks, chunk_document
from src.schema import Block, Document, Page


def block(block_id: str, kind: str, text: str, order: int, section: list[str]) -> Block:
    return Block(
        block_id=block_id,
        document_id="doc",
        page=1,
        block_type=kind,
        order=order,
        bbox=[0.1, 0.1, 0.9, 0.2],
        bbox_pixel=[10, 10, 90, 20],
        text=text,
        section_path=section,
    )


def document(blocks: list[Block]) -> Document:
    return Document(
        document_id="doc",
        source_file="source.pdf",
        total_pages=1,
        pages=[Page(document_id="doc", page=1, width=100, height=100, blocks=blocks)],
    )


class ChunkingTest(unittest.TestCase):
    def test_preserves_section_boundaries_and_binds_heading_to_first_body(self):
        blocks = [
            block("h1", "heading", "1 范围", 1, ["1"]),
            block("p1", "paragraph", "第一章正文。", 2, ["1"]),
            block("h2", "heading", "2 要求", 3, ["2"]),
            block("p2", "paragraph", "第二章正文。", 4, ["2"]),
        ]
        chunks = build_chunks(document(blocks), config=ChunkConfig(100, 200, 0))
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].block_ids, ["h1", "p1"])
        self.assertEqual(chunks[1].section_path, ["2"])

    def test_keeps_visual_caption_together_and_degrades_without_mllm(self):
        figure = block("fig", "figure", "", 1, ["1"])
        caption = block("cap", "caption", "图1 流程", 2, ["1"])
        figure.caption_ids = ["cap"]
        figure.image_crop = "assets/crops/fig.png"
        caption.caption_of = "fig"
        chunks = build_chunks(document([figure, caption]))
        self.assertEqual(chunks[0].block_ids, ["fig", "cap"])
        self.assertIn("visual_unavailable", chunks[0].quality_flags)
        self.assertEqual(chunks[0].visual_assets[0].status, "unavailable")

    def test_keeps_visual_description_separate_from_parser_text(self):
        figure = block("fig", "figure", "表格OCR", 1, ["1"])
        figure.image_crop = "assets/crops/fig.png"
        chunks = build_chunks(
            document([figure]),
            visual_items={"fig": {"status": "ok", "description": "柱状图呈上升趋势。"}},
            visual_file_available=True,
        )
        self.assertEqual(chunks[0].text, "表格OCR")
        self.assertEqual(chunks[0].visual_text, "柱状图呈上升趋势。")
        self.assertNotIn("visual_unavailable", chunks[0].quality_flags)

    def test_invalid_relation_is_a_warning_and_does_not_stop_chunking(self):
        paragraph = block("p", "paragraph", "正文", 1, [])
        paragraph.references = ["missing"]
        chunks = build_chunks(document([paragraph]))
        self.assertEqual(chunks[0].references, ["missing"])
        self.assertIn("invalid_relation:references", chunks[0].quality_flags)

    def test_reference_carries_the_visual_asset_backlink(self):
        paragraph = block("p", "paragraph", "见图1。", 1, ["1"])
        figure = block("fig", "figure", "", 2, ["2"])
        figure.image_crop = "assets/crops/fig.png"
        paragraph.references = ["fig"]
        chunks = build_chunks(
            document([paragraph, figure]),
            visual_items={"fig": {"status": "ok", "description": "流程图。"}},
            visual_file_available=True,
        )
        self.assertEqual(chunks[0].visual_assets[0].block_id, "fig")
        self.assertEqual(chunks[0].visual_assets[0].relation, "referenced")

    def test_overlap_uses_previous_split_fragment(self):
        paragraph = block("p", "paragraph", "甲" * 25 + "。" + "乙" * 25, 1, ["1"])
        chunks = build_chunks(document([paragraph]), config=ChunkConfig(20, 30, 5))
        self.assertEqual(chunks[1].overlap_text, "甲甲甲甲。")

    def test_splits_only_oversized_plain_text_and_writes_jsonl(self):
        paragraph = block("p", "paragraph", "第一句。" * 30, 1, ["1"])
        doc = document([paragraph])
        chunks = build_chunks(doc, config=ChunkConfig(20, 30, 5))
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.block_ids == ["p"] for chunk in chunks))
        self.assertTrue(all(len(chunk.text) <= 30 for chunk in chunks))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "doc.json"
            path.write_text(doc.model_dump_json(), encoding="utf-8")
            written = chunk_document(path, config=ChunkConfig(20, 30, 5))
            lines = (path.parent / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), len(written))
            self.assertEqual(json.loads(lines[0])["schema_version"], 1)
            self.assertEqual(json.loads(lines[0])["source_file"], "source.pdf")

    def test_malformed_block_is_skipped_without_dropping_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "doc.json"
            payload = document([block("ok", "paragraph", "有效正文", 1, [])]).model_dump(
                mode="json"
            )
            payload["pages"][0]["blocks"].append(
                {
                    "block_id": "bad",
                    "document_id": "doc",
                    "page": 1,
                    "block_type": "paragraph",
                    "text": "缺少坐标",
                }
            )
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            chunks = chunk_document(path)
            self.assertEqual(chunks[0].block_ids, ["ok"])
            self.assertIn("invalid_block_skipped:bad", chunks[0].quality_flags)


if __name__ == "__main__":
    unittest.main()
