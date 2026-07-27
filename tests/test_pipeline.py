from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import fitz

from src.config import ParseConfig
from src.data_processing.pipeline import parse_pdf, parse_pdfs


def _make_fixture(tmp: Path, name: str) -> Path:
    """在 tmp/out/{name} 下预置一份可复用的 structurev3.json，并返回对应 PDF 路径。

    走 reuse_detection=True，不依赖 paddle。fixture 与单篇测试同源。
    """
    pdf_path = tmp / f"{name}.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=100, height=100)
    page.insert_text((10, 20), "A text layer with enough characters for detection")
    pdf.save(pdf_path)
    pdf.close()

    doc_dir = tmp / "out" / name
    doc_dir.mkdir(parents=True, exist_ok=True)
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
    (doc_dir / "structurev3.json").write_text(
        json.dumps(raw, ensure_ascii=False), encoding="utf-8"
    )
    return pdf_path


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
            self.assertEqual(figure.crop_bbox_pixel, [0, 12, 100, 71])
            self.assertTrue(Path(figure.image_crop).is_file())
            self.assertTrue(figure.image_crop.endswith("p001_b01_figure.png"))
            self.assertTrue(figure.image_crop_raw.endswith("assets/imgs/a.jpg"))
            self.assertTrue(Path(figure.figure_crop).is_file())
            self.assertEqual(paragraph.references, [figure.block_id])
            self.assertEqual(paragraph.section_path, ["1"])
            self.assertTrue((output / "doc.json").is_file())
            self.assertTrue((output / "doc.md").is_file())

    def test_parse_pdfs_processes_skips_and_isolates_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            alpha = _make_fixture(root, "alpha")
            beta = _make_fixture(root, "beta")
            out_root = root / "out"
            config = ParseConfig(render_dpi=72)

            # 1) 首次批量：两篇都成功。
            s1 = parse_pdfs(
                [alpha, beta],
                config=config,
                reuse_detection=True,
                out_root=out_root,
                skip_existing=False,
            )
            self.assertEqual(s1["ok"], 2)
            self.assertEqual(s1["failed"], [])
            self.assertTrue((out_root / "alpha" / "doc.json").is_file())
            self.assertTrue((out_root / "beta" / "doc.json").is_file())

            # 2) skip_existing=True 时两篇都被跳过（断点续跑语义）。
            s2 = parse_pdfs(
                [alpha, beta],
                config=config,
                reuse_detection=True,
                out_root=out_root,
                skip_existing=True,
            )
            self.assertEqual(s2["ok"], 0)
            self.assertEqual(s2["skipped"], 2)

            # 3) 单篇失败被隔离，另一篇仍成功。
            import src.data_processing.pipeline as pl

            orig_parse = pl.parse_pdf
            calls = {"n": 0}

            def fake_parse(pdf_path, **kwargs):
                calls["n"] += 1
                if calls["n"] == 1:  # 首篇 alpha 抛错
                    raise RuntimeError("boom")
                return orig_parse(pdf_path, **kwargs)

            (out_root / "alpha" / "doc.json").unlink()
            (out_root / "beta" / "doc.json").unlink()
            pl.parse_pdf = fake_parse
            try:
                s3 = parse_pdfs(
                    [alpha, beta],
                    config=config,
                    reuse_detection=True,
                    out_root=out_root,
                    skip_existing=False,
                )
            finally:
                pl.parse_pdf = orig_parse
            self.assertEqual(s3["ok"], 1)
            self.assertEqual(len(s3["failed"]), 1)
            self.assertEqual(s3["failed"][0]["doc_id"], "alpha")
            self.assertIn("boom", s3["failed"][0]["error"])

            # 4) limit 截断数量。
            s4 = parse_pdfs(
                [alpha, beta],
                config=config,
                reuse_detection=True,
                out_root=out_root,
                skip_existing=False,
                limit=1,
            )
            self.assertEqual(s4["total"], 1)


if __name__ == "__main__":
    unittest.main()
