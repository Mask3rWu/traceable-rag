"""从高分辨率页图重裁视觉块，并为检测框添加受约束的冗余边界。"""
from __future__ import annotations

import math
import shutil
from pathlib import Path

from PIL import Image

from src.config import ParseConfig
from src.paths import PROJECT_ROOT
from src.schema import Block, Page

VISUAL_BLOCK_TYPES = {"figure", "table", "formula"}


def crop_visual_blocks(
    pages: list[Page],
    out_dir: Path,
    config: ParseConfig,
) -> int:
    """为 figure/table/formula 生成高分辨率扩边裁图，返回生成数量。

    原始 ``bbox_pixel`` 不变，实际使用的扩边框写入 ``crop_bbox_pixel``。
    已配对 caption 会限制对应方向的扩张，避免把独立标题裁进视觉块。
    """
    crops_dir = out_dir / "assets" / "crops"
    figures_dir = out_dir / "assets" / "figures"
    _validate_config(config)
    shutil.rmtree(crops_dir, ignore_errors=True)
    shutil.rmtree(figures_dir, ignore_errors=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated = 0

    for page in pages:
        page_image = _resolve_stored_path(page.page_image)
        if page_image is None or not page_image.is_file():
            raise FileNotFoundError(f"缺少第 {page.page} 页图: {page.page_image}")

        captions_by_target: dict[str, list[Block]] = {}
        for block in page.blocks:
            if block.block_type == "caption" and block.caption_of:
                captions_by_target.setdefault(block.caption_of, []).append(block)

        with Image.open(page_image) as image:
            for block in page.blocks:
                if block.block_type not in VISUAL_BLOCK_TYPES:
                    continue
                crop_box = _expanded_crop_box(
                    block,
                    captions_by_target.get(block.block_id, []),
                    image.width,
                    image.height,
                    config,
                )
                if crop_box is None:
                    continue

                suffix = _block_suffix(block.block_id)
                filename = f"p{page.page:03d}_{suffix}_{block.block_type}.png"
                crop_path = crops_dir / filename
                image.crop(tuple(crop_box)).save(crop_path, format="PNG")

                block.crop_bbox_pixel = crop_box
                block.image_crop = _stored_path(crop_path)

                captions = captions_by_target.get(block.block_id, [])
                if block.block_type in {"figure", "table"} and captions:
                    context_box = _figure_context_box(
                        block,
                        captions,
                        image.width,
                        image.height,
                        config,
                    )
                    context_path = figures_dir / filename
                    image.crop(tuple(context_box)).save(context_path, format="PNG")
                    block.figure_crop_bbox_pixel = context_box
                    block.figure_crop = _stored_path(context_path)
                generated += 1

    return generated


def _expanded_crop_box(
    block: Block,
    captions: list[Block],
    page_width: int,
    page_height: int,
    config: ParseConfig,
) -> list[int] | None:
    if len(block.bbox_pixel) != 4:
        return None
    x1, y1, x2, y2 = block.bbox_pixel
    if x2 <= x1 or y2 <= y1:
        return None

    width = x2 - x1
    height = y2 - y1
    pad_x = max(config.crop_padding_min_px, round(width * config.crop_padding_x_ratio))
    pad_top = max(
        config.crop_padding_min_px,
        round(height * config.crop_padding_top_ratio),
    )
    pad_bottom = max(
        config.crop_padding_min_px,
        round(height * config.crop_padding_bottom_ratio),
    )

    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_top)
    crop_x2 = min(page_width, x2 + pad_x)
    crop_y2 = min(page_height, y2 + pad_bottom)

    target_center_y = (y1 + y2) / 2
    for caption in captions:
        if len(caption.bbox_pixel) != 4:
            continue
        _, cap_y1, _, cap_y2 = caption.bbox_pixel
        caption_center_y = (cap_y1 + cap_y2) / 2
        if caption_center_y > target_center_y and cap_y1 < crop_y2:
            # 图注可能已侵入模型检测框；此时允许把原框本身缩回来。
            crop_y2 = max(crop_y1 + 1, cap_y1 - config.crop_caption_gap_px)
        elif caption_center_y < target_center_y and cap_y2 > crop_y1:
            crop_y1 = min(crop_y2 - 1, cap_y2 + config.crop_caption_gap_px)

    return [crop_x1, crop_y1, crop_x2, crop_y2]


def _figure_context_box(
    block: Block,
    captions: list[Block],
    page_width: int,
    page_height: int,
    config: ParseConfig,
) -> list[int]:
    """生成包含图片主体及全部中英文图注的人工复核框。"""
    boxes = [block.bbox_pixel] + [
        caption.bbox_pixel for caption in captions if len(caption.bbox_pixel) == 4
    ]
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    pad = config.crop_padding_min_px
    return [
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(page_width, x2 + pad),
        min(page_height, y2 + pad),
    ]


def _resolve_stored_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _stored_path(path: Path) -> str:
    try:
        value = path.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        value = path.resolve()
    return str(value).replace("\\", "/")


def _block_suffix(block_id: str) -> str:
    suffix = block_id.rsplit("_", 1)[-1].lower()
    if len(suffix) >= 2 and suffix[0] in {"b", "l"} and suffix[1:].isdigit():
        return suffix
    return "block"


def _validate_config(config: ParseConfig) -> None:
    ratios = (
        config.crop_padding_x_ratio,
        config.crop_padding_top_ratio,
        config.crop_padding_bottom_ratio,
    )
    if any(not math.isfinite(value) or value < 0 for value in ratios):
        raise ValueError("裁剪扩边比例必须是非负有限数")
    if config.crop_padding_min_px < 0 or config.crop_caption_gap_px < 0:
        raise ValueError("裁剪扩边像素和 caption 间距不能为负数")
