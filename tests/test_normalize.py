from __future__ import annotations

import unittest

from src.data_processing.normalize import normalize_page_blocks


class NormalizeTest(unittest.TestCase):
    def test_recovers_non_duplicate_high_confidence_layout_image(self):
        page = {
            "width": 100,
            "height": 100,
            "parsing_res_list": [
                {
                    "block_label": "image",
                    "block_content": '<img src="imgs/existing.jpg">',
                    "block_bbox": [0, 0, 20, 20],
                    "block_id": 1,
                    "block_order": None,
                }
            ],
            "layout_det_res": {
                "boxes": [
                    {"label": "image", "score": 0.99, "coordinate": [0, 0, 20, 20]},
                    {"label": "image", "score": 0.95, "coordinate": [30, 30, 60, 60]},
                    {"label": "image", "score": 0.80, "coordinate": [70, 70, 90, 90]},
                ]
            },
        }

        blocks = normalize_page_blocks(page, "doc", 1, 200, 200)

        self.assertEqual(len(blocks), 2)
        fallback = blocks[1]
        self.assertEqual(fallback.block_id, "doc_P001_L01")
        self.assertEqual(fallback.bbox_pixel, [60, 60, 120, 120])
        self.assertEqual(fallback.source_method, "layout")
        self.assertEqual(fallback.confidence, 0.95)
        self.assertTrue(fallback.image_crop.endswith("30_30_60_60.jpg"))

    def test_recovers_missing_bilingual_caption_from_overall_ocr(self):
        page = {
            "width": 100,
            "height": 100,
            "parsing_res_list": [
                {
                    "block_label": "figure_title",
                    "block_content": "Fig.4 Example",
                    "block_bbox": [10, 80, 90, 90],
                    "block_id": 1,
                    "block_order": None,
                }
            ],
            "layout_det_res": {"boxes": []},
            "overall_ocr_res": {
                "rec_texts": ["图4 示例", "Fig.4 Example"],
                "rec_boxes": [[20, 70, 80, 78], [10, 80, 90, 90]],
                "rec_scores": [0.98, 0.99],
            },
        }

        blocks = normalize_page_blocks(page, "doc", 1, 100, 100)

        captions = [block for block in blocks if block.block_type == "caption"]
        self.assertEqual(len(captions), 2)
        self.assertEqual(
            {(block.caption_language, block.text) for block in captions},
            {("zh", "图4 示例"), ("en", "Fig.4 Example")},
        )

    def test_numeric_caption_stops_before_ascii_title_text(self):
        page = {
            "width": 100,
            "height": 100,
            "parsing_res_list": [],
            "layout_det_res": {"boxes": []},
            "overall_ocr_res": {
                "rec_texts": ["图7SAR图像的毁伤评估流程"],
                "rec_boxes": [[10, 80, 90, 90]],
                "rec_scores": [0.98],
            },
        }

        blocks = normalize_page_blocks(page, "doc", 1, 100, 100)

        self.assertEqual(blocks[0].label_no, "7")

    def test_lower_score_layout_image_is_recovered_when_caption_supported(self):
        page = {
            "width": 100,
            "height": 100,
            "parsing_res_list": [
                {
                    "block_label": "figure_title",
                    "block_content": "图3 示例",
                    "block_bbox": [20, 70, 80, 80],
                    "block_id": 1,
                    "block_order": None,
                }
            ],
            "layout_det_res": {
                "boxes": [
                    {"label": "image", "score": 0.82, "coordinate": [20, 20, 80, 68]},
                ]
            },
        }

        blocks = normalize_page_blocks(page, "doc", 1, 100, 100)

        figures = [block for block in blocks if block.block_type == "figure"]
        self.assertEqual(len(figures), 1)
        self.assertEqual(figures[0].source_method, "layout")

    def test_display_formula_absorbs_number_without_number_block(self):
        page = {
            "width": 200,
            "height": 100,
            "parsing_res_list": [
                {
                    "block_label": "display_formula",
                    "block_content": "$$B=I_1-I_0$$",
                    "block_bbox": [50, 40, 120, 55],
                    "block_id": 1,
                    "block_order": None,
                },
                {
                    "block_label": "formula_number",
                    "block_content": "(1)",
                    "block_bbox": [170, 40, 190, 55],
                    "block_id": 2,
                    "block_order": None,
                },
            ],
            "layout_det_res": {"boxes": []},
        }

        blocks = normalize_page_blocks(page, "doc", 1, 200, 100)

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].block_type, "formula")
        self.assertEqual(blocks[0].formula_no, "1")


if __name__ == "__main__":
    unittest.main()
