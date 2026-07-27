"""PDF 解析层端到端编排与命令行入口。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

from src.config import ParseConfig
from src.paths import (
    PROJECT_ROOT,
    doc_id_from_path,
    doc_out_dir,
    list_gjb_pdfs,
    list_input_pdfs,
    list_paper_pdfs,
)
from src.schema import Document, Page
from src.data_processing.crop import crop_visual_blocks
from src.data_processing.detect import build_pipeline, detect_pdf
from src.data_processing.markdown import write_document_markdown
from src.data_processing.normalize import normalize_page_blocks
from src.data_processing.relations import build_relations
from src.data_processing.render import render_pdf


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
    pipeline=None,
) -> Document:
    """执行渲染、版面检测、归一化和关系构建，写出 ``doc.json``。

    ``reuse_detection`` 复用已有 structurev3.json，适合只重跑后处理和测试。
    ``pipeline`` 可传入已构造的 PPStructureV3 产线在多篇间复用（批量解析时
    整批只建一次模型）；仅 ``not reuse_detection`` 时用到。
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
        detect_pdf(pdf_path, out_dir, config, pipeline=pipeline)
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
            config.layout_visual_fallback_min_score,
        )
        for block in blocks:
            if block.image_crop:
                raw_crop = _project_relative(out_dir / block.image_crop)
                block.image_crop_raw = raw_crop
                block.image_crop = raw_crop
        all_blocks.extend(blocks)
        pages.append(Page(**rendered, document_id=document_id, blocks=blocks))

    build_relations(all_blocks)
    crop_visual_blocks(pages, out_dir, config)
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
    write_document_markdown(document, out_dir)
    return document


def parse_pdfs(
    pdf_paths: Sequence[Path],
    *,
    config: ParseConfig | None = None,
    skip_existing: bool = True,
    limit: int | None = None,
    reuse_detection: bool = False,
    out_root: Path | str | None = None,
) -> dict:
    """批量解析多份 PDF。

    - 整批只构造一次 PPStructureV3 模型并在多篇间复用（非复用检测模式）。
    - ``skip_existing``：跳过已存在 ``doc.json`` 的文档，支持断点续跑。
    - 单篇失败被隔离并记录，不中断其余文档。
    - ``limit``：最多处理 N 篇（冒烟用）。
    - ``out_root``：输出根目录，默认 ``processed/parsed``；传入可重定向（测试用）。

    返回汇总 ``{total, ok, skipped, failed}``，``failed`` 为
    ``[{doc_id, source, error}]``。
    """
    config = config or ParseConfig()
    paths = [Path(p) for p in pdf_paths]
    if limit is not None:
        paths = paths[: limit]

    # 非复用模式才需要模型；复用检测模式只重跑后处理，不碰 paddle。
    pipe = None if reuse_detection else build_pipeline(config)

    def _doc_dir(doc_id: str) -> Path:
        if out_root is not None:
            return Path(out_root) / doc_id
        return doc_out_dir(doc_id)

    total = len(paths)
    ok = 0
    skipped = 0
    failed: list[dict] = []
    batch_started = time.perf_counter()

    for index, pdf_path in enumerate(paths, start=1):
        document_id = doc_id_from_path(pdf_path)
        doc_json = _doc_dir(document_id) / "doc.json"
        started = time.perf_counter()

        if skip_existing and doc_json.is_file():
            skipped += 1
            print(f"[{index}/{total}] {document_id}: skipped (doc.json 已存在)")
            continue

        try:
            document = parse_pdf(
                pdf_path,
                config=config,
                out_dir=_doc_dir(document_id),
                reuse_detection=reuse_detection,
                pipeline=pipe,
            )
            elapsed = time.perf_counter() - started
            print(
                f"[{index}/{total}] {document_id}: ok "
                f"({document.total_pages} 页, {document.block_count} 块, "
                f"{elapsed:.1f}s) -> {_project_relative(doc_json)}"
            )
            ok += 1
        except Exception as exc:  # 隔离单篇失败，继续下一篇
            elapsed = time.perf_counter() - started
            failed.append(
                {"doc_id": document_id, "source": pdf_path.name, "error": repr(exc)}
            )
            print(
                f"[{index}/{total}] {document_id}: FAILED after {elapsed:.1f}s "
                f"- {exc!r}"
            )

    elapsed_total = time.perf_counter() - batch_started
    print(
        f"\n批量完成: {ok} ok, {skipped} skipped, {len(failed)} failed, "
        f"共 {total} 篇, 耗时 {elapsed_total:.1f}s"
    )
    if failed:
        print("失败列表:")
        for item in failed:
            print(f"  - {item['doc_id']} ({item['source']}): {item['error']}")
    return {"total": total, "ok": ok, "skipped": skipped, "failed": failed}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="解析 PDF 为可溯源结构化块")
    parser.add_argument(
        "pdf",
        type=Path,
        nargs="*",
        help="一个或多个 PDF 路径（可与枚举开关混用）",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="枚举 资料/ 下全部 PDF（国军标 + 论文）",
    )
    parser.add_argument(
        "--papers-only",
        action="store_true",
        help="仅枚举 资料/论文/ 下的 PDF",
    )
    parser.add_argument(
        "--gjb-only",
        action="store_true",
        help="仅枚举 资料/国军标/ 下的 PDF",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="跳过已存在 doc.json 的文档（默认开启；--no-skip-existing 关闭）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="最多处理 N 篇（冒烟用）",
    )
    parser.add_argument(
        "--reuse-detection",
        action="store_true",
        help="复用输出目录中的 structurev3.json，仅重跑后处理（不加载模型）",
    )
    parser.add_argument("--dpi", type=int, default=200, help="页图渲染 DPI（默认 200）")
    parser.add_argument(
        "--crop-padding-x",
        type=float,
        default=0.02,
        help="视觉块左右扩边比例（默认 0.02）",
    )
    parser.add_argument(
        "--crop-padding-top",
        type=float,
        default=0.02,
        help="视觉块上方扩边比例（默认 0.02）",
    )
    parser.add_argument(
        "--crop-padding-bottom",
        type=float,
        default=0.08,
        help="视觉块下方扩边比例（默认 0.08）",
    )
    parser.add_argument(
        "--crop-padding-min-px",
        type=int,
        default=12,
        help="各方向最小扩边像素（默认 12）",
    )
    parser.add_argument(
        "--layout-fallback-min-score",
        type=float,
        default=0.90,
        help="补回 layout-only 图片候选的最低置信度（默认 0.90）",
    )
    return parser


def _resolve_pdfs(args) -> list[Path]:
    """合并显式路径与枚举开关，去重保序。"""
    pdfs: list[Path] = [Path(p) for p in args.pdf]
    if args.all:
        pdfs.extend(list_input_pdfs())
    if args.papers_only:
        pdfs.extend(list_paper_pdfs())
    if args.gjb_only:
        pdfs.extend(list_gjb_pdfs())
    seen: set[str] = set()
    deduped: list[Path] = []
    for p in pdfs:
        key = str(Path(p).resolve())
        if key not in seen:
            seen.add(key)
            deduped.append(Path(p))
    return deduped


def main(argv: Sequence[str] | None = None) -> int:
    # Windows 控制台默认 cp936，中文进度行会乱码；尽量切 UTF-8（不影响测试）。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = build_parser()
    args = parser.parse_args(argv)
    pdfs = _resolve_pdfs(args)
    if not pdfs:
        parser.error("未指定 PDF：请给出路径或使用 --all/--papers-only/--gjb-only")

    config = ParseConfig(
        render_dpi=args.dpi,
        crop_padding_x_ratio=args.crop_padding_x,
        crop_padding_top_ratio=args.crop_padding_top,
        crop_padding_bottom_ratio=args.crop_padding_bottom,
        crop_padding_min_px=args.crop_padding_min_px,
        layout_visual_fallback_min_score=args.layout_fallback_min_score,
    )
    summary = parse_pdfs(
        pdfs,
        config=config,
        skip_existing=args.skip_existing,
        limit=args.limit,
        reuse_detection=args.reuse_detection,
    )
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
