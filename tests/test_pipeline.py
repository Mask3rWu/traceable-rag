from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import fitz

from src.config import ParseConfig
from src.数据处理.pipeline import parse_pdf


class PipelineTest(unittest.TestCase):
    def test_reuses_detection_and_writes_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf_path = root / "sample.pdf"
            output = root / "out"
            output.mkdir()

            pdf = fitz.open()
            page = pdf.new_page(width=100, height=100)
            page.insert_text((10, 20), "A text layer with enough characters for detection")
            pdf.save(pdf_path)
            pdf.close()

            raw = [{
                "page_index": 0,
                "width": 50,
                "height": 50,
                "parsing_res_list": [
                    {"block_label": "paragraph_title", "block_content": "## 1 Intro", "block_bbox": [5, 5, 45, 10], "block_id": 0, "block_order": 0},
                    {"block_label": "image", "block_content": '<img src="imgs/a.jpg">', "block_bbox": [5, 12, 45, 35], "block_id": 1, "block_order": None},
                    {"block_label": "figure_title", "block_content": "图1 示例", "block_bbox": [5, 36, 45, 40], "block_id": 2, "block_order": None},
                    {"block_label": "text", "block_content": "如图1所示", "block_bbox": [5, 42, 45, 48], "block_id": 3, "block_order": 3},
                ],
            }]
            (output / "structurev3.json").write_text(
                json.dumps(raw, ensure_ascii=False), encoding="utf-8"
            )

            document = parse_pdf(
                pdf_path,
                config=ParseConfig(render_dpi=72),
                out_dir=output,
                reuse_detection=True,
            )

            self.assertEqual(document.total_pages, 1)
            self.assertEqual(document.block_count, 4)
            self.assertEqual(
                len({block.block_id for block in document.pages[0].blocks}), 4
            )
            figure = document.pages[0].blocks[1]
            paragraph = document.pages[0].blocks[3]
            self.assertEqual(figure.bbox_pixel, [10, 24, 90, 70])
            self.assertEqual(paragraph.references, [figure.block_id])
            self.assertEqual(paragraph.section_path, ["1"])
            self.assertTrue((output / "doc.json").is_file())


if __name__ == "__main__":
    unittest.main()
