"""PDF 解析层端到端编排与命令行入口。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.config import ParseConfig
from src.paths import PROJECT_ROOT, doc_id_from_path, doc_out_dir
from src.schema import Document, Page
from src.数据处理.detect import detect_pdf
from src.数据处理.normalize import normalize_page_blocks
from src.数据处理.relations import build_relations
from src.数据处理.render import render_pdf


def _load_detection(out_dir: Path) -> list[dict]:
    path = out_dir / "structurev3.json"
    if not path.is_file():
        raise FileNotFoundError(f"缺少可复用的检测结果: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"检测结果顶层应为页面列表: {path}")
    return data


def _project_relative(path: Path) -> str:
    try:
        path = path.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return str(path.resolve()).replace("\\", "/")
    return str(path).replace("\\", "/")


def parse_pdf(
    pdf_path: Path,
    *,
    config: ParseConfig | None = None,
    out_dir: Path | None = None,
    reuse_detection: bool = False,
) -> Document:
    """执行渲染、版面检测、归一化和关系构建，写出 ``doc.json``。

    ``reuse_detection`` 复用已有 structurev3.json，适合只重跑后处理和测试。
    """
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        raise FileNotFoundError(f"PDF 不存在: {pdf_path}")

    config = config or ParseConfig()
    document_id = doc_id_from_path(pdf_path)
    out_dir = Path(out_dir) if out_dir is not None else doc_out_dir(document_id)
    (out_dir / "pages").mkdir(parents=True, exist_ok=True)
    (out_dir / "assets").mkdir(parents=True, exist_ok=True)

    rendered_pages = render_pdf(pdf_path, out_dir, dpi=config.render_dpi)
    if reuse_detection:
        detected_pages = _load_detection(out_dir)
    else:
        detect_pdf(pdf_path, out_dir, config)
        detected_pages = _load_detection(out_dir)

    detected_by_index = {
        int(page.get("page_index", index)): page
        for index, page in enumerate(detected_pages)
    }
    if len(rendered_pages) != len(detected_pages):
        raise ValueError(
            "渲染页数与检测页数不一致: "
            f"{len(rendered_pages)} != {len(detected_pages)}"
        )

    pages: list[Page] = []
    all_blocks = []
    for rendered in rendered_pages:
        page_num = rendered["page"]
        detected = detected_by_index.get(page_num - 1)
        if detected is None:
            raise ValueError(f"检测结果缺少第 {page_num} 页")
        blocks = normalize_page_blocks(
            detected,
            document_id,
            page_num,
            rendered["width"],
            rendered["height"],
        )
        for block in blocks:
            if block.image_crop:
                block.image_crop = _project_relative(out_dir / block.image_crop)
        all_blocks.extend(blocks)
        pages.append(Page(**rendered, document_id=document_id, blocks=blocks))

    build_relations(all_blocks)
    document = Document(
        document_id=document_id,
        source_file=pdf_path.name,
        total_pages=len(pages),
        pages=pages,
    )
    (out_dir / "doc.json").write_text(
        json.dumps(document.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="解析 PDF 为可溯源结构化块")
    parser.add_argument("pdf", type=Path, nargs="+", help="一个或多个 PDF 路径")
    parser.add_argument(
        "--reuse-detection",
        action="store_true",
        help="复用输出目录中的 structurev3.json，仅重跑后处理",
    )
    parser.add_argument("--dpi", type=int, default=200, help="页图渲染 DPI（默认 200）")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ParseConfig(render_dpi=args.dpi)
    for pdf_path in args.pdf:
        document = parse_pdf(
            pdf_path,
            config=config,
            reuse_detection=args.reuse_detection,
        )
        output = doc_out_dir(document.document_id) / "doc.json"
        print(
            f"{document.source_file}: {document.total_pages} 页, "
            f"{document.block_count} 块 -> {output}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
