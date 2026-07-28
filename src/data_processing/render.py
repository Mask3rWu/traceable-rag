"""PDF -> 逐页图 + 文本层检查。

对应 pdf-parser.md §8 的渲染阶段。PyMuPDF 负责：
- 渲染每页为 PNG（供 PP-StructureV3 与 bbox 回溯）
- 检测每页是否有原生文本层（供 normalize 的 native/ocr 分流，见 §6.5）

按 数据处理.md 要求"按页分流"：不假设整份 PDF 一致。

渲染结果缓存（``pages/_render_meta.json``）：记录 DPI、源 PDF 的 mtime/size
与每页元数据。命中条件——DPI 一致、源 PDF 未改（mtime+size 不变）、缓存
列出的全部页图仍在且无多余页图——时跳过渲染，直接返回缓存元数据。这让
``--reuse-detection`` 重跑后处理时不必重复转图；由于版面检测读的是 PDF 本身
（不依赖渲染图），全量重检测时缓存同样安全。需强制重渲染时删 ``pages/``
目录或换 DPI 即可。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import fitz  # PyMuPDF


def has_text_layer(page: fitz.Page, min_chars: int = 20) -> bool:
    """页面是否有足够原生文本。

    扫描页 text 为空或极少；有文本层的页通常有大量文字。
    min_chars 阈值过滤仅含页码/页眉的"伪文本页"。
    """
    try:
        text = page.get_text("text") or ""
    except Exception:
        return False
    # 去除空白后字符数
    return len(text.strip()) >= min_chars


def native_scan_dpi(page: fitz.Page, coverage: float = 0.95) -> int | None:
    """整页扫描图的原生 DPI；非整页单图页返回 None。

    部分扫描件 PDF 把 MediaBox 设得很大（如 2100×2960pt≈29×41in），但内嵌
    的扫描图实际只有 2550×3300px（≈Letter@300DPI）。按请求 DPI 渲染会把源
    图放大数倍，纯插值无新细节，且页图/裁图偏大偏慢。这类页检测其原生
    分辨率（图像素 / 图显示英寸），渲染时据此封顶，避免无谓放大。

    判据：恰好一张图、且其显示矩形占页面 ≥ coverage（95%）才视作整页扫描。
    占满一部分的图（论文插图、页眉小图）一律不算，保持请求 DPI 不变。
    """
    images = page.get_images(full=True)
    if len(images) != 1:
        return None
    xref = images[0][0]
    rects = page.get_image_rects(xref)
    if not rects:
        return None
    rect = rects[0]
    mb = page.mediabox
    if mb.width <= 0 or mb.height <= 0:
        return None
    if rect.width / mb.width < coverage or rect.height / mb.height < coverage:
        return None
    info = page.parent.extract_image(xref)
    disp_w_in = rect.width / 72.0
    disp_h_in = rect.height / 72.0
    if disp_w_in <= 0 or disp_h_in <= 0:
        return None
    # PDF 中的扫描图可能被非等比拉伸。取两个方向中较低的原生 DPI，保证
    # 最终渲染不会在任一方向放大源位图。向下取整可避免四舍五入后轻微放大。
    return max(
        1,
        math.floor(
            min(info["width"] / disp_w_in, info["height"] / disp_h_in)
        ),
    )


def _rel_path(path: Path) -> str:
    """返回相对项目根的路径串；不在项目根下时回退绝对路径。"""
    from src.paths import PROJECT_ROOT

    try:
        stored = path.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        stored = path.resolve()
    return str(stored).replace("\\", "/")


def _render_meta_path(pages_dir: Path) -> Path:
    return pages_dir / "_render_meta.json"


# 渲染缓存结构版本：封顶/裁剪等渲染逻辑变化时 +1，使旧缓存自动失效重渲染。
# 当前版本：按横纵最低原生 DPI 封顶，并记录每页实际渲染 DPI。
_RENDER_SCHEMA = 3


def _try_load_cache(
    pages_dir: Path, pdf_path: Path, dpi: int
) -> list[dict] | None:
    """命中缓存则返回每页元数据，否则返回 None。

    任何异常（缓存文件缺失/损坏、字段缺失等）都视为未命中，回退到正常渲染。
    """
    meta_path = _render_meta_path(pages_dir)
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # 渲染逻辑版本不一致（如封顶规则改过）-> 旧缓存失效。
        if meta.get("render_schema") != _RENDER_SCHEMA:
            return None
        if meta.get("dpi") != dpi:
            return None
        stat = pdf_path.stat()
        if (
            meta.get("pdf_size") != stat.st_size
            or meta.get("pdf_mtime") != stat.st_mtime
        ):
            return None  # 源 PDF 已变（覆盖/重新生成等）

        cached = meta.get("pages") or []
        # 期望的页图文件名集合；与目录现存 p*.png 比对：
        # 缺图 -> 上次渲染未完成；多图 -> 页数变化（PDF 增删页），均判失效。
        expected = {f"p{int(p['page']):03d}.png" for p in cached}
        existing = {p.name for p in pages_dir.glob("p*.png")}
        if expected != existing:
            return None

        # 路径按当前项目布局重算（缓存里的 page_image 只作留底，不被读取），
        # 保证与一次全新渲染返回的路径串完全一致。
        results: list[dict] = []
        for p in cached:
            img_path = pages_dir / f"p{int(p['page']):03d}.png"
            results.append(
                {
                    "page": int(p["page"]),
                    "page_image": _rel_path(img_path),
                    "width": p["width"],
                    "height": p["height"],
                    "render_dpi": int(p["render_dpi"]),
                    "has_text_layer": p["has_text_layer"],
                }
            )
        return results
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_cache(
    pages_dir: Path, pdf_path: Path, dpi: int, pages: list[dict]
) -> None:
    """渲染完成后写缓存；写入失败不影响主流程。"""
    stat = pdf_path.stat()
    meta = {
        "render_schema": _RENDER_SCHEMA,
        "dpi": dpi,
        "pdf_name": pdf_path.name,
        "pdf_mtime": stat.st_mtime,
        "pdf_size": stat.st_size,
        "pages": pages,
    }
    try:
        _render_meta_path(pages_dir).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        pass


def render_pdf(
    pdf_path: Path,
    out_dir: Path,
    dpi: int = 200,
) -> list[dict]:
    """渲染 PDF 每页为 PNG，返回每页元数据。

    命中缓存（见模块文档）时跳过渲染，直接返回缓存的每页元数据。

    返回: [{"page": 1-based, "page_image": rel_path, "width": px, "height": px,
            "render_dpi": 实际渲染DPI, "has_text_layer": bool}, ...]
    """
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    cached = _try_load_cache(pages_dir, pdf_path, dpi)
    if cached is not None:
        return cached

    results: list[dict] = []

    capped_pages: list[int] = []  # 被原生分辨率封顶的页号（提示用）

    doc = fitz.open(str(pdf_path))
    try:
        for i, page in enumerate(doc):
            page_num = i + 1  # 1-based

            # 文本层检查（渲染前，在原始页面上做）
            text_layer = has_text_layer(page)

            # 整页扫描图：用其原生分辨率封顶，避免放大插值（见 native_scan_dpi）。
            page_dpi = dpi
            native = native_scan_dpi(page)
            if native is not None and native < dpi:
                page_dpi = native
                capped_pages.append(page_num)

            # 渲染
            zoom = page_dpi / 72.0  # PDF 默认 72 DPI
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img_name = f"p{page_num:03d}.png"
            img_path = pages_dir / img_name
            pix.save(str(img_path))

            results.append(
                {
                    "page": page_num,
                    "page_image": _rel_path(img_path),
                    "width": pix.width,
                    "height": pix.height,
                    "render_dpi": page_dpi,
                    "has_text_layer": text_layer,
                }
            )
    finally:
        doc.close()

    if capped_pages:
        # 不假设整份 PDF 一致：只报告被封顶的页。表明渲染分辨率被原生扫描
        # 分辨率限制，避免放大插值；版面检测仍读 PDF 本身，不受影响。
        print(
            f"  [render] {pdf_path.name}: {len(capped_pages)}/{len(results)} 页按"
            f"原生扫描分辨率渲染（请求 {dpi} DPI，扫描件未放大）"
        )

    _write_cache(pages_dir, pdf_path, dpi, results)
    return results
