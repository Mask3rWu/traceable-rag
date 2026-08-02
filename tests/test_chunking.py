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
            block("h1", "heading", "# 1 Scope", 1, ["1"]),
            block("p1", "paragraph", "First section text.", 2, ["1"]),
            block("h2", "heading", "# 2 Requirements", 3, ["2"]),
            block("p2", "paragraph", "Second section text.", 4, ["2"]),
        ]
        chunks = build_chunks(document(blocks), config=ChunkConfig(100, 200, 0))
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].block_ids, ["h1", "p1"])
        self.assertEqual(chunks[1].section_path, ["2"])
        self.assertEqual(chunks[0].text, "First section text.")
        self.assertEqual(chunks[0].heading_path, ["1 Scope"])
        self.assertNotIn("#", chunks[0].embedding_text)

    def test_skips_heading_only_nodes_but_inherits_them_in_child_context(self):
        blocks = [
            block("h1", "heading", "# 3 Requirements", 1, ["3"]),
            block("h2", "heading", "## 3.2 Details", 2, ["3", "3.2"]),
            block("h3", "heading", "### 3.2.1 Nuclear Survivability", 3, ["3", "3.2", "3.2.1"]),
            block("p", "paragraph", "Clause body.", 4, ["3", "3.2", "3.2.1"]),
        ]
        chunks = build_chunks(document(blocks), config=ChunkConfig(100, 200, 0))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].block_ids, ["h3", "p"])
        self.assertEqual(
            chunks[0].heading_path,
            ["3 Requirements", "3.2 Details", "3.2.1 Nuclear Survivability"],
        )
        self.assertIn("3.2.1 Nuclear Survivability", chunks[0].embedding_text)

    def test_keeps_visual_caption_together_and_degrades_without_mllm(self):
        figure = block("fig", "figure", "", 1, ["1"])
        caption = block("cap", "caption", "Figure 1 workflow", 2, ["1"])
        figure.caption_ids = ["cap"]
        figure.image_crop = "assets/crops/fig.png"
        caption.caption_of = "fig"
        chunks = build_chunks(document([figure, caption]))
        self.assertEqual(chunks[0].block_ids, ["fig", "cap"])
        self.assertIn("visual_unavailable", chunks[0].quality_flags)
        self.assertEqual(chunks[0].visual_assets[0].status, "unavailable")

    def test_keeps_visual_description_separate_from_parser_text(self):
        figure = block("fig", "figure", "Table OCR", 1, ["1"])
        figure.image_crop = "assets/crops/fig.png"
        chunks = build_chunks(
            document([figure]),
            visual_items={"fig": {"status": "ok", "description": "Chart rises."}},
            visual_file_available=True,
        )
        self.assertEqual(chunks[0].text, "Table OCR")
        self.assertEqual(chunks[0].visual_text, "Chart rises.")
        self.assertNotIn("visual_unavailable", chunks[0].quality_flags)

    def test_invalid_relation_is_a_warning_and_does_not_stop_chunking(self):
        paragraph = block("p", "paragraph", "Body", 1, [])
        paragraph.references = ["missing"]
        chunks = build_chunks(document([paragraph]))
        self.assertEqual(chunks[0].references, ["missing"])
        self.assertIn("invalid_relation:references", chunks[0].quality_flags)

    def test_reference_carries_the_visual_asset_backlink(self):
        paragraph = block("p", "paragraph", "See Figure 1.", 1, ["1"])
        figure = block("fig", "figure", "", 2, ["2"])
        figure.image_crop = "assets/crops/fig.png"
        paragraph.references = ["fig"]
        chunks = build_chunks(
            document([paragraph, figure]),
            visual_items={"fig": {"status": "ok", "description": "Workflow."}},
            visual_file_available=True,
        )
        self.assertEqual(chunks[0].visual_assets[0].block_id, "fig")
        self.assertEqual(chunks[0].visual_assets[0].relation, "referenced")

    def test_binds_same_section_formula_to_its_explanatory_text(self):
        paragraph = block("p", "paragraph", "The expression is defined below.", 1, ["1"])
        formula = block("f", "formula", "x = y + z", 2, ["1"])
        paragraph.references = ["f"]
        chunks = build_chunks(document([paragraph, formula]))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].block_ids, ["p", "f"])
        self.assertIn("x = y + z", chunks[0].text)

    def test_does_not_bind_formula_across_section_boundaries(self):
        paragraph = block("p", "paragraph", "See the formula.", 1, ["1"])
        formula = block("f", "formula", "x = y + z", 2, ["2"])
        paragraph.references = ["f"]
        chunks = build_chunks(document([paragraph, formula]))
        self.assertEqual(len(chunks), 2)

    def test_overlap_uses_previous_split_fragment(self):
        paragraph = block("p", "paragraph", "A" * 25 + "." + "B" * 25, 1, ["1"])
        chunks = build_chunks(document([paragraph]), config=ChunkConfig(20, 30, 5))
        self.assertEqual(chunks[1].overlap_text, ".BBBB")

    def test_splits_oversized_heading_and_plain_text(self):
        heading = block("h", "heading", "# 1 Scope", 1, ["1"])
        paragraph = block("p", "paragraph", "One. " * 30, 2, ["1"])
        chunks = build_chunks(document([heading, paragraph]), config=ChunkConfig(20, 30, 5))
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.block_ids == ["h", "p"] for chunk in chunks))
        self.assertTrue(all(len(chunk.text) <= 30 for chunk in chunks))

    def test_writes_schema_v2_jsonl(self):
        doc = document([block("p", "paragraph", "Body", 1, [])])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "doc.json"
            path.write_text(doc.model_dump_json(), encoding="utf-8")
            written = chunk_document(path)
            lines = (path.parent / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), len(written))
            payload = json.loads(lines[0])
            self.assertEqual(payload["schema_version"], 2)
            self.assertEqual(payload["source_file"], "source.pdf")
            self.assertEqual(payload["embedding_text"], "Document: doc\nContent: Body")
            self.assertTrue((path.parent / "relation_validation.jsonl").is_file())

    def test_malformed_block_is_skipped_without_dropping_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "doc.json"
            payload = document([block("ok", "paragraph", "Valid", 1, [])]).model_dump(mode="json")
            payload["pages"][0]["blocks"].append(
                {
                    "block_id": "bad",
                    "document_id": "doc",
                    "page": 1,
                    "block_type": "paragraph",
                    "text": "Missing coordinates",
                }
            )
            path.write_text(json.dumps(payload), encoding="utf-8")
            chunks = chunk_document(path)
            self.assertEqual(chunks[0].block_ids, ["ok"])
            self.assertIn("invalid_block_skipped:bad", chunks[0].quality_flags)


if __name__ == "__main__":
    unittest.main()
