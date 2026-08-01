"""PP-StructureV3 包装。

对应 pdf-parser.md §4。负责调用 PP-StructureV3 产线，落盘原始结果。
单次调用处理整份 PDF（产线内部逐页处理），输出：
- structurev3.json  全部页的原始解析结果（留底，便于重跑 normalize）
- structurev3.md     Markdown（图表配对参考，normalize 用）
- imgs/              图片子图（PP-StructureV3 自动裁剪，喂 MLLM）

实测输出结构（PP-StructureV3，format_block_content=True）：
  result.json['res'] = {
    'input_path', 'page_index'(0-based), 'page_count', 'width', 'height',
    'model_settings', 'parsing_res_list': [
      {'block_label','block_content','block_bbox','block_id','block_order'}, ...
    ],
    'doc_preprocessor_res', 'layout_det_res', 'overall_ocr_res'
  }
  - 图片块 block_content 为 HTML: <img src="imgs/img_in_image_box_x1_y1_x2_y2.jpg">
  - figure_title 即图表标题；image 即图/图表区域；formula 公式；formula_number 公式编号
  - block_order 可能为 None（如 figure_title/image 不在主阅读流）
"""
from __future__ import annotations

import json
import re
import shutil
import warnings
from pathlib import Path
from typing import Sequence

from src.config import ParseConfig, apply_env

# 抑制 paddle 的海量日志，便于看清业务输出
warnings.filterwarnings("ignore")


def build_pipeline(config: ParseConfig):
    """构造 PPStructureV3 产线（import 即触发模型加载，需在 apply_env 后）。

    批量解析时只调用一次，把返回的产线在多篇 PDF 间复用，避免每篇重载模型。
    """
    apply_env(config)
    from paddleocr import PPStructureV3

    return PPStructureV3(format_block_content=True, **config.to_pipeline_kwargs())


# 向后兼容：旧调用方仍可用私有名。
_build_pipeline = build_pipeline


def detect_pdf(
    pdf_path: Path,
    out_dir: Path,
    config: ParseConfig | None = None,
    *,
    pipeline=None,
    rendered_pages: Sequence[dict] | None = None,
) -> dict:
    """对单 PDF 跑 PP-StructureV3，落盘原始结果，返回元数据。

    ``pipeline`` 可传入一个已构造的 PPStructureV3 产线以复用（批量解析时
    整批只建一次模型）；为 None 时自行 ``build_pipeline``，保持单篇旧行为。

    返回: {"pages": [ {page_index, width, height, block_count}, ... ],
           "structurev3_json": rel_path, "structurev3_md": rel_path,
           "imgs_dir": rel_path}
    """
    config = config or ParseConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "assets").mkdir(exist_ok=True)
    # 通过兼容别名构造，保留既有调用方和测试对 _build_pipeline 的替换能力。
    pipe = pipeline if pipeline is not None else _build_pipeline(config)

    # 1. 调用产线（整 PDF 一次）。未检测到目标水印时 prediction_input 就是
    # 原始 pdf_path，确保其他文档的推理输入和既有行为完全不变。
    prediction_input = pdf_path
    watermarks = {}
    watermark_metadata_path = None
    if config.use_watermark_preprocessing and rendered_pages:
        from src.data_processing.watermark import prepare_watermark_input

        prediction_input, watermarks, watermark_metadata_path = prepare_watermark_input(
            pdf_path, out_dir, rendered_pages
        )
    # 必须物化结果：既用于 JSON，也用于 Markdown，避免整份 PDF 推理两次。
    try:
        if prediction_input == pdf_path:
            results = list(pipe.predict(str(pdf_path), **config.to_predict_kwargs()))
        else:
            # Preserve untouched pages exactly on the original prediction path.
            # The cleaned document contributes only confirmed watermark pages.
            original_results = list(
                pipe.predict(str(pdf_path), **config.to_predict_kwargs())
            )
            cleaned_results = list(
                pipe.predict(str(prediction_input), **config.to_predict_kwargs())
            )
            cleaned_by_index = {
                int(result.json["res"].get("page_index", index)): result
                for index, result in enumerate(cleaned_results)
            }
            results = []
            for index, result in enumerate(original_results):
                page_index = int(result.json["res"].get("page_index", index))
                if page_index + 1 in watermarks:
                    cleaned = cleaned_by_index.get(page_index)
                    if cleaned is None:
                        raise ValueError(f"清洗检测结果缺少第 {page_index + 1} 页")
                    results.append(cleaned)
                else:
                    results.append(result)
    finally:
        if prediction_input != pdf_path:
            prediction_input.unlink(missing_ok=True)

    # 2. 汇总所有页的 json，落盘 structurev3.json
    all_pages = []
    for r in results:
        res = r.json["res"]
        page_index = int(res.get("page_index", len(all_pages)))
        watermark = watermarks.get(page_index + 1)
        if watermark is not None:
            res["watermark"] = {
                "detected": True,
                "watermark_type": watermark.watermark_type,
                "mask_ratio": watermark.mask_ratio,
                "bbox": watermark.bbox,
                "template_similarity": watermark.template_similarity,
                "cleaned_image": watermark.cleaned_image,
            }
            # Raw output should continue to identify the actual source document.
            res["input_path"] = str(pdf_path)
        all_pages.append(res)
    all_pages.sort(key=lambda page: int(page.get("page_index", 0)))

    from src.paths import PROJECT_ROOT

    sv3_json = out_dir / "structurev3.json"
    sv3_json.write_text(
        json.dumps(all_pages, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 3. 落盘 Markdown：用产线的 save_to_markdown，自动跨页合并 + 图片相对路径处理
    md_path = out_dir / "structurev3.md"
    # save_to_markdown 写到指定目录，文件名固定；先存到临时目录再改名
    tmp_md = out_dir / "_md_tmp"
    shutil.rmtree(tmp_md, ignore_errors=True)
    tmp_md.mkdir(exist_ok=True)
    for r in results:
        try:
            r.save_to_markdown(save_path=str(tmp_md))
        except Exception:
            pass
    # save_to_markdown 在目录下生成 "<stem>.md"（可能多个文件），合并到目标
    # Paddle 文件名通常带页号；字符串排序会得到 0,1,10,2。
    def natural_key(path: Path) -> list[object]:
        return [
            int(part) if part.isdigit() else part.casefold()
            for part in re.split(r"(\d+)", path.name)
        ]

    md_files = sorted(tmp_md.glob("*.md"), key=natural_key)
    if md_files and not watermarks:
        merged = "\n\n".join(f.read_text(encoding="utf-8") for f in md_files)
        md_path.write_text(merged, encoding="utf-8")
    else:
        # 水印页来自清洗 PDF、其他页来自原 PDF，文件名前缀不同，不能依赖
        # save_to_markdown 的文件名排序。直接按原始页序生成可审查 Markdown。
        fallback_pages = []
        for page in all_pages:
            contents = [
                block.get("block_content", "")
                for block in page.get("parsing_res_list", [])
                if isinstance(block.get("block_content"), str)
            ]
            fallback_pages.append("\n\n".join(contents))
        md_path.write_text("\n\n".join(fallback_pages), encoding="utf-8")
    # save_to_markdown 可能同时写出 imgs/ 到 save 目录，迁移到 assets/
    src_imgs = tmp_md / "imgs"
    dst_imgs = out_dir / "assets" / "imgs"
    if src_imgs.exists():
        dst_imgs.parent.mkdir(parents=True, exist_ok=True)
        if dst_imgs.exists():
            shutil.rmtree(dst_imgs)
        shutil.move(str(src_imgs), str(dst_imgs))
    shutil.rmtree(tmp_md, ignore_errors=True)

    # 重写 md 中的图片引用路径：imgs/xxx -> assets/imgs/xxx（相对项目根可访问）
    text = md_path.read_text(encoding="utf-8")
    text = re.sub(
        r"src=([\"'])imgs/",
        lambda match: f"src={match.group(1)}assets/imgs/",
        text,
    )
    text = text.replace("](imgs/", "](assets/imgs/")
    md_path.write_text(text, encoding="utf-8")

    def stored_path(path: Path) -> str:
        try:
            value = path.resolve().relative_to(PROJECT_ROOT.resolve())
        except ValueError:
            value = path.resolve()
        return str(value).replace("\\", "/")

    return {
        "pages": [
            {
                "page_index": p.get("page_index", 0),  # 0-based
                "width": p.get("width"),
                "height": p.get("height"),
                "block_count": len(p.get("parsing_res_list", [])),
            }
            for p in all_pages
        ],
        "structurev3_json": stored_path(sv3_json),
        "structurev3_md": stored_path(md_path),
        "imgs_dir": stored_path(dst_imgs) if dst_imgs.exists() else None,
        "watermarks": stored_path(watermark_metadata_path)
        if watermark_metadata_path is not None
        else None,
        "watermark_pages": sorted(watermarks),
    }
