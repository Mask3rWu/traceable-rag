from __future__ import annotations

import json
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


def _write_scanlike_pdf(path: Path, page_pt=(2100, 2960), img_px=(2550, 3597)) -> None:
    """写一个"整页单图扫描件"PDF：大 MediaBox + 一张占满页的位图。

    模拟 GJB 扫描件：MediaBox 设成 2100x2960pt（≈29x41in），内嵌一张与
    页面同宽高比的位图占满整页，原生分辨率约 87 DPI（2550px / 29.2in）。
    """
    import io
    from PIL import Image

    pdf = fitz.open()
    page = pdf.new_page(width=page_pt[0], height=page_pt[1])
    img = Image.new("RGB", img_px, (220, 220, 220))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    page.insert_image(
        fitz.Rect(0, 0, page_pt[0], page_pt[1]),
        stream=buf.getvalue(),
        keep_proportion=False,
    )
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
            self.assertEqual(results[0]["render_dpi"], 72)
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

    def test_scan_page_capped_to_native_dpi(self):
        # 横向约 87 DPI、纵向约 87 DPI，按较低方向封顶，避免任一方向放大。
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "scan.pdf"
            image_size = (2550, 3597)
            _write_scanlike_pdf(pdf, img_px=image_size)
            out = root / "out"

            results = render_pdf(pdf, out, dpi=200)

            # 横向 DPI = 87.4、纵向 DPI = 87.5，向下封顶到 87。
            self.assertEqual(results[0]["render_dpi"], 87)
            self.assertLess(results[0]["width"], 3000)
            self.assertGreater(results[0]["width"], 2000)
            self.assertLessEqual(results[0]["width"], image_size[0])
            self.assertLessEqual(results[0]["height"], image_size[1])

    def test_scan_page_uses_lower_vertical_dpi(self):
        # 非等比放置的扫描图：横向约 87 DPI、纵向约 80 DPI，应按 80 封顶。
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "scan.pdf"
            image_size = (2550, 3300)
            _write_scanlike_pdf(pdf, img_px=image_size)

            results = render_pdf(pdf, root / "out", dpi=200)

            self.assertEqual(results[0]["render_dpi"], 80)
            self.assertLessEqual(results[0]["width"], image_size[0])
            self.assertLessEqual(results[0]["height"], image_size[1])

    def test_normal_pdf_not_capped(self):
        # 文本层页（无整页图）：即便请求低 DPI 也不触发封顶逻辑，
        # 按请求 DPI 渲染。
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "text.pdf"
            _write_pdf(pdf)  # 100x100pt 文本页
            out = root / "out"

            results = render_pdf(pdf, out, dpi=72)
            self.assertEqual(results[0]["width"], 100)  # 100pt @72dpi = 100px

    def test_stale_cache_without_schema_version_is_invalidated(self):
        # 旧缓存（无 render_schema 字段）应被判定失效并重渲染。
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "a.pdf"
            _write_pdf(pdf)
            out = root / "out"

            render_pdf(pdf, out, dpi=72)
            meta_path = out / "pages" / "_render_meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            # 抹掉 schema 字段并破坏页图。文件仍存在，因此只有 schema 校验
            # 能触发重渲染；仅校验页图文件名的实现会错误命中。
            meta.pop("render_schema", None)
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            png = out / "pages" / "p001.png"
            png.write_bytes(b"stale cache")

            # 旧缓存失效 -> 重渲染为有效 PNG。
            results = render_pdf(pdf, out, dpi=72)
            self.assertEqual(png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(len(results), 1)


if __name__ == "__main__":
    unittest.main()
