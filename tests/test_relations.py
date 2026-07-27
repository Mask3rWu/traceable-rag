from __future__ import annotations

import unittest

from src.schema import Block
from src.数据处理.relations import build_relations


def block(block_id: str, block_type: str, text: str, order: int, bbox=None) -> Block:
    bbox_pixel = bbox or [0, order * 20, 100, order * 20 + 10]
    return Block(
        block_id=block_id,
        document_id="doc",
        page=1,
        block_type=block_type,
        order=order,
        bbox=[value / 100 for value in bbox_pixel],
        bbox_pixel=bbox_pixel,
        text=text,
        label_norm=text if block_type == "caption" else None,
        label_no="3" if block_type == "caption" else None,
    )


class RelationsTest(unittest.TestCase):
    def test_caption_and_cross_reference_point_to_figure(self):
        blocks = [
            block("fig", "figure", "", 1, [10, 20, 90, 80]),
            block("cap", "caption", "图3 毁伤流程", 2, [10, 82, 90, 92]),
            block("text", "paragraph", "结果如图 3 所示。", 3),
        ]
        build_relations(blocks)
        self.assertEqual(blocks[1].caption_of, "fig")
        self.assertEqual(blocks[0].label_no, "3")
        self.assertEqual(blocks[2].references, ["fig"])

    def test_markdown_headings_build_section_path(self):
        blocks = [
            block("h1", "heading", "## 5 标题", 1),
            block("h2", "heading", "### 5.3 子标题", 2),
            block("p1", "paragraph", "内容", 3),
            block("h3", "heading", "### 5.4 同级标题", 4),
        ]
        build_relations(blocks)
        self.assertEqual(blocks[2].section_path, ["5", "5.3"])
        self.assertEqual(blocks[3].section_path, ["5", "5.4"])

    def test_unordered_figure_inherits_interpolated_section(self):
        heading = block("doc_P001_B03", "heading", "## 3.2 方法", 2)
        figure = block("doc_P001_B05", "figure", "", 0)
        figure.order = None
        next_heading = block("doc_P001_B09", "heading", "## 3.3 后续", 6)
        build_relations([heading, figure, next_heading])
        self.assertEqual(figure.section_path, ["3.2"])

    def test_appendix_and_section_references(self):
        blocks = [
            block("body", "paragraph", "见附录 A，并参见第 3.2 节。", 1),
            block("section", "heading", "## 3.2 方法", 2),
            block("appendix", "heading", "## 附录 A（资料性附录）", 3),
            block("appendix_text", "paragraph", "附录内容", 4),
        ]
        build_relations(blocks)
        self.assertEqual(blocks[0].references, ["appendix", "section"])
        self.assertTrue(blocks[2].is_appendix)
        self.assertEqual(blocks[3].appendix_type, "资料性附录")

    def test_distinct_captions_do_not_share_one_figure(self):
        blocks = [
            block("cap6", "caption", "图6 缺失图", 1, [10, 10, 90, 20]),
            block("fig7", "figure", "", 2, [10, 50, 90, 80]),
            block("cap7", "caption", "图7 现存图", 3, [10, 82, 90, 92]),
        ]
        blocks[0].label_no = "6"
        blocks[2].label_no = "7"
        build_relations(blocks)
        self.assertIsNone(blocks[0].caption_of)
        self.assertEqual(blocks[2].caption_of, "fig7")


if __name__ == "__main__":
    unittest.main()
