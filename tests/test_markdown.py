from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data_processing.markdown import write_document_markdown
from src.schema import Block, Document, Page


def make_block(
    block_id: str,
    page: int,
    block_type: str,
    order: int,
    text: str = "",
) -> Block:
    return Block(
        block_id=block_id,
        document_id="doc",
        page=page,
        block_type=block_type,
        order=order,
        bbox=[0.1, 0.1, 0.9, 0.2],
        bbox_pixel=[10, 10, 90, 20],
        text=text,
    )


class MarkdownTest(unittest.TestCase):
    def test_renders_continuation_image_and_bilingual_captions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "figure.png"
            Image.new("RGB", (10, 10), "white").save(image_path)

            first = make_block("p1", 1, "paragraph", 1, "跨页段")
            second = make_block("p2", 2, "paragraph", 1, "落。")
            first.continues_to = second.block_id
            second.continuation_of = first.block_id
            figure = make_block("fig", 2, "figure", 2)
            figure.image_crop = str(image_path)
            figure.label_norm = "图1 示例"
            zh = make_block("zh", 2, "caption", 3, "图1 示例")
            zh.caption_language = "zh"
            en = make_block("en", 2, "caption", 4, "Fig.1 Example")
            en.caption_language = "en"
            document = Document(
                document_id="doc",
                source_file="doc.pdf",
                total_pages=2,
                pages=[
                    Page(document_id="doc", page=1, width=100, height=100, blocks=[first]),
                    Page(
                        document_id="doc", page=2, width=100, height=100,
                        blocks=[second, figure, zh, en],
                    ),
                ],
            )

            path = write_document_markdown(document, root)
            markdown = path.read_text(encoding="utf-8")
            self.assertIn("跨页段落。", markdown)
            self.assertIn("![图1 示例](figure.png)", markdown)
            self.assertIn('lang="zh">图1 示例', markdown)
            self.assertIn('lang="en">Fig.1 Example', markdown)

    def test_renders_formula_number_separately_from_formula(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            formula = make_block("formula", 1, "formula", 1, "$$B=I_1-I_0$$")
            formula.formula_no = "1"
            document = Document(
                document_id="doc",
                source_file="doc.pdf",
                total_pages=1,
                pages=[
                    Page(
                        document_id="doc", page=1, width=100, height=100,
                        blocks=[formula],
                    )
                ],
            )

            markdown = write_document_markdown(document, root).read_text(encoding="utf-8")
            self.assertIn("$$B=I_1-I_0$$", markdown)
            self.assertIn('<div align="right">(1)</div>', markdown)
