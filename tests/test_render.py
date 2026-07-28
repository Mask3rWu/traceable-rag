from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import fitz

from src.data_processing.render import render_pdf


def _write_pdf(path: Path, text: str = "A text layer with enough characters for detection") -> None:
    """写一个 100x100 pt 的单页 PDF，带文本层。"""
    pdf = fitz.open()
    page = pdf.new_page(width=100, height=100)
    page.insert_text((10, 20), text)
    pdf.save(path)
    pdf.close()


def _png_mtime(path: Path) -> float:
    return (path / "pages" / "p001.png").stat().st_mtime


class RenderCacheTest(unittest.TestCase):
    def test_renders_pages_and_writes_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "a.pdf"
            _write_pdf(pdf)
            out = root / "out"

            results = render_pdf(pdf, out, dpi=72)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["page"], 1)
            self.assertEqual(results[0]["width"], 100)  # 100pt @72dpi = 100px
            self.assertTrue(results[0]["has_text_layer"])
            self.assertTrue((out / "pages" / "p001.png").is_file())
            self.assertTrue((out / "pages" / "_render_meta.json").is_file())

    def test_cache_hit_skips_rerender(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "a.pdf"
            _write_pdf(pdf)
            out = root / "out"

            first = render_pdf(pdf, out, dpi=72)
            mtime_before = _png_mtime(out)

            # 同文件同 DPI 再跑：命中缓存，不应重写页图。
            second = render_pdf(pdf, out, dpi=72)
            mtime_after = _png_mtime(out)

            self.assertEqual(first, second)
            self.assertEqual(mtime_before, mtime_after)

    def test_cache_miss_on_dpi_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "a.pdf"
            _write_pdf(pdf)
            out = root / "out"

            low = render_pdf(pdf, out, dpi=72)
            high = render_pdf(pdf, out, dpi=144)

            # DPI 翻倍 -> 像素尺寸翻倍，判缓存失效并重渲染。
            self.assertEqual(low[0]["width"], 100)
            self.assertEqual(high[0]["width"], 200)

    def test_cache_miss_on_pdf_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "a.pdf"
            _write_pdf(pdf, "short text")
            out = root / "out"

            render_pdf(pdf, out, dpi=72)
            first_meta = (out / "pages" / "_render_meta.json").read_text(encoding="utf-8")

            # 覆盖源 PDF（内容与大小都变）：mtime/size 不再匹配 -> 失效重渲染。
            _write_pdf(pdf, "a much longer text body that changes the file size")
            render_pdf(pdf, out, dpi=72)
            second_meta = (out / "pages" / "_render_meta.json").read_text(encoding="utf-8")

            self.assertNotEqual(first_meta, second_meta)

    def test_cache_miss_on_missing_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "a.pdf"
            _write_pdf(pdf)
            out = root / "out"

            render_pdf(pdf, out, dpi=72)
            png = out / "pages" / "p001.png"
            self.assertTrue(png.is_file())
            png.unlink()  # 删掉一页图

            # 期望页图集合与现存不一致 -> 失效重渲染，页图被重新生成。
            results = render_pdf(pdf, out, dpi=72)
            self.assertTrue(png.is_file())
            self.assertEqual(len(results), 1)

    def test_corrupt_cache_falls_back_to_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "a.pdf"
            _write_pdf(pdf)
            out = root / "out"

            render_pdf(pdf, out, dpi=72)
            # 写坏缓存文件：不应抛错，应回退到正常渲染。
            (out / "pages" / "_render_meta.json").write_text("{not json", encoding="utf-8")

            results = render_pdf(pdf, out, dpi=72)
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0]["has_text_layer"])


if __name__ == "__main__":
    unittest.main()
