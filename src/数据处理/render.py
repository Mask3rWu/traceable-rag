"""PDF -> 逐页图 + 文本层检查。

对应 pdf-parser.md §8 的渲染阶段。PyMuPDF 负责：
- 渲染每页为 PNG（供 PP-StructureV3 与 bbox 回溯）
- 检测每页是否有原生文本层（供 normalize 的 native/ocr 分流，见 §6.5）

按 数据处理.md 要求"按页分流"：不假设整份 PDF 一致。
"""
from __future__ import annotations

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


def render_pdf(
    pdf_path: Path,
    out_dir: Path,
    dpi: int = 200,
) -> list[dict]:
    """渲染 PDF 每页为 PNG，返回每页元数据。

    返回: [{"page": 1-based, "page_image": rel_path, "width": px, "height": px,
            "has_text_layer": bool}, ...]
    """
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    doc_id = pages_dir.parent.name
    results: list[dict] = []

    doc = fitz.open(str(pdf_path))
    try:
        zoom = dpi / 72.0  # PDF 默认 72 DPI
        matrix = fitz.Matrix(zoom, zoom)
        for i, page in enumerate(doc):
            page_num = i + 1  # 1-based

            # 文本层检查（渲染前，在原始页面上做）
            text_layer = has_text_layer(page)

            # 渲染
            pix = page.get_pixmap(matrix=matrix)
            img_name = f"p{page_num:03d}.png"
            img_path = pages_dir / img_name
            pix.save(str(img_path))

            # 相对项目根的路径（便于 doc.json 中记录与跨机迁移）
            from src.paths import PROJECT_ROOT

            try:
                stored_path = img_path.resolve().relative_to(PROJECT_ROOT.resolve())
            except ValueError:
                stored_path = img_path.resolve()
            rel = str(stored_path).replace("\\", "/")

            results.append(
                {
                    "page": page_num,
                    "page_image": rel,
                    "width": pix.width,
                    "height": pix.height,
                    "has_text_layer": text_layer,
                }
            )
    finally:
        doc.close()

    return results
