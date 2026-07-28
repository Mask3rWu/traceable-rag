from __future__ import annotations

import unittest

from src.schema import Block
from src.data_processing.relations import build_relations


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

    def test_paren_numbered_headings_are_children_of_parent(self):
        # 论文常见：4.2.1 下用 1）/2） 分点；PP-StructureV3 把它们误标为 ##，
        # 实为 4.2.1 的子级，且同级括号互斥。
        blocks = [
            block("h1", "heading", "## 4 战场图像采集", 1),
            block("h2", "heading", "### 4.2 卫星采集图像", 2),
            block("h3", "heading", "#### 4.2.1 机器学习方法", 3),
            block("p1", "paragraph", "正文", 4),
            block("h4", "heading", "## 1 ）基于贝叶斯方法", 5),
            block("p2", "paragraph", "贝叶斯正文", 6),
            block("h5", "heading", "## 2 ）基于神经网络", 7),
            block("p3", "paragraph", "神经网络正文", 8),
            block("h6", "heading", "#### 4.2.2 变化检测方法", 9),
        ]
        build_relations(blocks)
        # 1） 作为 4.2.1 的子级，深度 +1
        self.assertEqual(
            blocks[4].section_path, ["4", "4.2", "4.2.1", "1）"]
        )
        self.assertEqual(
            blocks[5].section_path, ["4", "4.2", "4.2.1", "1）"]
        )
        # 2） 与 1） 同级：弹掉 1） 再压
        self.assertEqual(
            blocks[6].section_path, ["4", "4.2", "4.2.1", "2）"]
        )
        # 4.2.2 回到 4 级，弹掉括号与 4.2.1
        self.assertEqual(blocks[8].section_path, ["4", "4.2", "4.2.2"])

    def test_paren_headings_reset_under_new_top_section(self):
        # 新一级数字标题后，括号编号应基于新的父级深度，不残留旧栈
        blocks = [
            block("h1", "heading", "## 3 科研图像采集", 1),
            block("h2", "heading", "### 3.2 评估方法", 2),
            block("h3", "heading", "## 1 ）模型仿真", 3),
        ]
        build_relations(blocks)
        self.assertEqual(blocks[2].section_path, ["3", "3.2", "1）"])

    def test_unordered_figure_inherits_interpolated_section(self):
        heading = block("doc_P001_B03", "heading", "## 3.2 方法", 2)
        figure = block("doc_P001_B05", "figure", "", 0)
        figure.order = None
        next_heading = block("doc_P001_B09", "heading", "## 3.3 后续", 6)
        build_relations([heading, figure, next_heading])
        # 点号编号按祖先链对齐，未出现的上级（3）也补进路径，使祖先链完整。
        # 关键属性：无序图继承所在标题的 section_path。
        self.assertEqual(heading.section_path, ["3", "3.2"])
        self.assertEqual(figure.section_path, heading.section_path)
        self.assertEqual(next_heading.section_path, ["3", "3.3"])

    def test_deep_number_ignores_missing_intermediate_heading(self):
        # 国军标深层编号：PP-StructureV3 漏检中间标题 3.1.3 时，3.1.3.x
        # 必须挂到按编号推得的虚拟祖先 3.1.3 下，而非沿用相邻兄弟 3.1.2。
        blocks = [
            block("h1", "heading", "## 3", 1),
            block("h2", "heading", "### 3.1", 2),
            block("h3", "heading", "#### 3.1.2", 3),
            block("h4", "heading", "##### 3.1.3.1 子节", 4),  # 3.1.3 缺失
            block("p1", "paragraph", "正文", 5),
            block("h5", "heading", "##### 3.1.3.2 子节2", 6),
        ]
        build_relations(blocks)
        self.assertEqual(blocks[3].section_path, ["3", "3.1", "3.1.3", "3.1.3.1"])
        self.assertEqual(blocks[4].section_path, ["3", "3.1", "3.1.3", "3.1.3.1"])
        self.assertEqual(blocks[5].section_path, ["3", "3.1", "3.1.3", "3.1.3.2"])

    def test_deep_number_pops_wrong_sibling_subtree(self):
        # 3.1.3.x 出现后，若后续回到 3.1.3 的兄弟 3.1.4，应清掉 3.1.3 子树。
        blocks = [
            block("h1", "heading", "## 3", 1),
            block("h2", "heading", "### 3.1", 2),
            block("h3", "heading", "##### 3.1.3.1", 3),  # 3.1.3 缺失，补虚拟
            block("h4", "heading", "#### 3.1.4", 4),  # 回到 3.1 的子级，应清深层
        ]
        build_relations(blocks)
        self.assertEqual(blocks[2].section_path, ["3", "3.1", "3.1.3", "3.1.3.1"])
        self.assertEqual(blocks[3].section_path, ["3", "3.1", "3.1.4"])

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

    def test_bilingual_captions_share_one_figure(self):
        blocks = [
            block("fig", "figure", "", 1, [10, 20, 90, 70]),
            block("zh", "caption", "图3 冲击波", 2, [10, 72, 90, 80]),
            block("en", "caption", "Fig.3 Shock wave", 3, [10, 82, 90, 90]),
        ]
        build_relations(blocks)
        self.assertEqual(blocks[1].caption_of, "fig")
        self.assertEqual(blocks[2].caption_of, "fig")
        self.assertEqual(blocks[0].caption_ids, ["zh", "en"])

    def test_cross_page_paragraph_keeps_blocks_and_adds_links(self):
        first = block("p1", "paragraph", "给定公式$I_1", 1, [10, 80, 90, 95])
        second = block("p2", "paragraph", "2$继续说明。", 1, [10, 5, 90, 20])
        second.page = 2
        build_relations([first, second])
        self.assertEqual(first.continues_to, "p2")
        self.assertEqual(second.continuation_of, "p1")
        self.assertIn("cross_page_formula", second.quality_flags)

    def test_page_footnote_does_not_break_body_continuation(self):
        first = block("p1", "paragraph", "分类不够详", 1, [10, 80, 90, 95])
        footnote = block("f1", "footnote", "作者信息", 2, [10, 96, 90, 99])
        second = block("p2", "paragraph", "细，没有针对性。", 1, [10, 5, 90, 20])
        second.page = 2
        build_relations([first, footnote, second])
        self.assertEqual(first.continues_to, "p2")
        self.assertEqual(second.continuation_of, "p1")

    def test_unicode_math_cross_page_is_flagged(self):
        first = block("p1", "paragraph", "其中 I∈R，i∈1,", 1, [10, 80, 90, 95])
        second = block("p2", "paragraph", "2，…分别表示。", 1, [10, 5, 90, 20])
        second.page = 2
        build_relations([first, second])
        self.assertIn("cross_page_formula", second.quality_flags)

    def test_formula_number_reference_points_to_formula(self):
        formula = block("formula", "formula", "$$B=I_1-I_0$$", 1)
        formula.formula_no = "1"
        paragraph = block("text", "paragraph", "由式（1）可得。", 2)
        build_relations([formula, paragraph])
        self.assertEqual(paragraph.references, ["formula"])


if __name__ == "__main__":
    unittest.main()
