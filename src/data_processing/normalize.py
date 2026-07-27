"""原始输出 -> 归一 block。

对应 pdf-parser.md §4.2（标签归一）+ §6.4（坐标归一）+ §5.2（block_id）。
把 PP-StructureV3 的 parsing_res_list 转为 schema.Block。

标签 -> block_type 映射（实测 label 取值）：
  doc_title / paragraph_title    -> heading
  text / abstract                -> paragraph
  list / algorithm               -> list
  table                          -> table
  formula                        -> formula
  image / chart                  -> figure
  figure_title / table_title / figure_caption -> caption
  reference / reference_content  -> appendix（或单独 reference）
  footnote                       -> footnote
  page_number / header / footer / header_image / footer_image -> （丢弃）
"""
from __future__ import annotations

import re

from src.schema import Block

# PP-StructureV3 label -> 归一 block_type
LABEL_TO_TYPE = {
    "doc_title": "heading",
    "paragraph_title": "heading",
    "text": "paragraph",
    "abstract": "paragraph",
    "list": "list",
    "algorithm": "list",
    "table": "table",
    "formula": "formula",
    "display_formula": "formula",
    "inline_formula": "formula",
    "image": "figure",
    "chart": "figure",
    "figure_title": "caption",
    "table_title": "caption",
    "figure_caption": "caption",
    "reference": "appendix",
    "reference_content": "appendix",
    "footnote": "footnote",
}

# 丢弃的 label（页眉页脚页码等噪声）
DROP_LABELS = {"page_number", "number", "header", "footer", "header_image", "footer_image", "aside_text"}

# 图片 HTML 路径提取
_IMG_SRC_RE = re.compile(r'src=["\']([^"\']+)["\']')
# 剥离 HTML 标签，取纯文本（caption 的 block_content 被 <div> 包裹）
_TAG_RE = re.compile(r"<[^>]+>")


def _extract_image_path(block_content) -> str | None:
    """从 image 块的 block_content(HTML) 提取图片相对路径，并补 assets/ 前缀。

    PP-StructureV3 把图存到 out_dir/assets/imgs/，但 HTML 引用写的是 imgs/xxx。
    统一改为 assets/imgs/xxx（相对项目根可访问）。
    """
    if not isinstance(block_content, str):
        return None
    m = _IMG_SRC_RE.search(block_content)
    if not m:
        return None
    p = m.group(1)
    if p.startswith("imgs/"):
        p = "assets/imgs/" + p[len("imgs/"):]
    return p


def _strip_html(s: str) -> str:
    """剥离 HTML 标签，取纯文本。"""
    return _TAG_RE.sub("", s).strip() if s else s




def _extract_label_no(text: str) -> str | None:
    """从图表标题抽取编号，如 '图3 毁伤...' -> '3'，'Fig.1 xxx' -> '1'。"""
    if not text:
        return None
    m = re.match(
        r"^\s*(?:图|表|Fig\.?|Figure|Table)\s*"
        r"(\d+(?:[.\-][0-9A-Za-z]+)*|[A-Za-z]+(?:[.\-]?\d+)*)",
        text,
        re.IGNORECASE,
    )
    return m.group(1) if m else None


def _caption_language(text: str) -> str:
    """按图注前缀标记语言，避免混合正文干扰判断。"""
    stripped = text.lstrip()
    if re.match(r"^(?:图|表)", stripped):
        return "zh"
    if re.match(r"^(?:Fig\.?|Figure|Table)\b", stripped, re.IGNORECASE):
        return "en"
    return "unknown"


def normalize_page_blocks(
    page_res: dict,
    document_id: str,
    page_num: int,  # 1-based
    page_width: int,
    page_height: int,
    layout_visual_fallback_min_score: float = 0.90,
) -> list[Block]:
    """转换单页 parsing_res_list -> schema.Block 列表。

    ``page_width``/``page_height`` 是最终页图的像素尺寸。PP-StructureV3
    解析 PDF 时可能使用另一 DPI，因此先把检测坐标缩放到页图坐标系，再做
    [0, 1] 归一化。原始坐标仍完整保留在 structurev3.json 中。
    """
    blocks_raw = page_res.get("parsing_res_list", [])
    out: list[Block] = []

    detector_width = page_res.get("width") or page_width
    detector_height = page_res.get("height") or page_height
    scale_x = page_width / detector_width if detector_width else 1.0
    scale_y = page_height / detector_height if detector_height else 1.0

    for source_index, b in enumerate(blocks_raw):
        label = b.get("block_label", "")
        if label == "formula_number":
            # 编号会挂到同页相邻公式的 formula_no，不单独生成可裁剪块。
            continue
        if label in DROP_LABELS:
            continue
        block_type = LABEL_TO_TYPE.get(label)
        if block_type is None:
            # 未知 label 默认归为 paragraph，保留 raw_label 以便审查
            block_type = "paragraph"

        bbox_raw = list(b.get("block_bbox", [0, 0, 0, 0]))
        if len(bbox_raw) == 4:
            x1, y1, x2, y2 = bbox_raw
            bbox_pixel = [
                round(x1 * scale_x),
                round(y1 * scale_y),
                round(x2 * scale_x),
                round(y2 * scale_y),
            ]
        else:
            bbox_pixel = [0, 0, 0, 0]
        # 归一化坐标 [0,1]
        if page_width and page_height and len(bbox_pixel) == 4:
            x1, y1, x2, y2 = bbox_pixel
            bbox = [x1 / page_width, y1 / page_height, x2 / page_width, y2 / page_height]
        else:
            bbox = [0.0, 0.0, 0.0, 0.0]

        content = b.get("block_content", "")
        raw_text = content if isinstance(content, str) else str(content)

        # 图片：提取裁剪路径
        image_crop = None
        if block_type == "figure":
            image_crop = _extract_image_path(content)

        # caption / heading / paragraph 文本剥 HTML 取纯文本
        text = _strip_html(raw_text) if block_type in ("caption", "heading", "paragraph", "list", "footnote", "appendix") else raw_text

        # 图表标题：抽取编号
        label_no = _extract_label_no(text) if block_type == "caption" else None

        order = b.get("block_order")  # 可能为 None

        # block_order 对图片/caption 常为空，且会与其他 raw block_id 撞号。
        # 页面原始 block_id（或原始列表下标）才是页内稳定唯一标识。
        source_id = b.get("block_id", source_index)
        source_id = source_id if isinstance(source_id, int) else source_index
        block_id = f"{document_id}_P{page_num:03d}_B{source_id:02d}"

        blk = Block(
            block_id=block_id,
            document_id=document_id,
            page=page_num,
            block_type=block_type,
            order=order,
            bbox=[round(v, 5) for v in bbox],
            bbox_pixel=[int(v) for v in bbox_pixel],
            text=text,
            source_method="ocr",
            confidence=0.0,  # parsing_res_list 不带 per-block score
            image_crop=image_crop,
            image_crop_raw=image_crop,
            label_no=label_no,
            raw_label=label,
        )
        if block_type == "formula":
            blk.formula_no = _nearest_formula_no(bbox_raw, blocks_raw)
        if block_type == "caption":
            blk.label_norm = text.strip()
            blk.caption_language = _caption_language(text)
        out.append(blk)

    _append_ocr_caption_fallbacks(
        page_res,
        out,
        document_id,
        page_num,
        page_width,
        page_height,
        scale_x,
        scale_y,
    )
    _append_layout_figure_fallbacks(
        page_res,
        blocks_raw,
        out,
        document_id,
        page_num,
        page_width,
        page_height,
        scale_x,
        scale_y,
        layout_visual_fallback_min_score,
    )
    return out


def _nearest_formula_no(formula_bbox: list, blocks_raw: list[dict]) -> str | None:
    """查找与公式同一水平带、通常位于右侧的公式编号。"""
    if len(formula_bbox) != 4:
        return None
    _, y1, _, y2 = formula_bbox
    center_y = (y1 + y2) / 2
    candidates: list[tuple[float, str]] = []
    for block in blocks_raw:
        if block.get("block_label") != "formula_number":
            continue
        bbox = list(block.get("block_bbox", []))
        if len(bbox) != 4:
            continue
        number_center_y = (bbox[1] + bbox[3]) / 2
        if abs(number_center_y - center_y) > max(y2 - y1, bbox[3] - bbox[1]) * 1.5:
            continue
        text = str(block.get("block_content", "")).strip()
        match = re.search(r"\(?\s*([0-9A-Za-z.\-]+)\s*\)?", text)
        if match:
            candidates.append((abs(number_center_y - center_y), match.group(1)))
    return min(candidates, default=(0.0, None), key=lambda item: item[0])[1]


def _append_ocr_caption_fallbacks(
    page_res: dict,
    out: list[Block],
    document_id: str,
    page_num: int,
    page_width: int,
    page_height: int,
    scale_x: float,
    scale_y: float,
) -> None:
    """补回整体 OCR 已识别、但 parsing_res_list 漏掉的图表标题。"""
    existing = {
        (block.label_no.casefold(), block.caption_language)
        for block in out
        if block.block_type == "caption" and block.label_no
    }
    ocr = page_res.get("overall_ocr_res", {})
    texts = ocr.get("rec_texts", []) or []
    boxes = ocr.get("rec_boxes", []) or []
    scores = ocr.get("rec_scores", []) or []

    for index, text_value in enumerate(texts):
        text = str(text_value).strip()
        label_no = _extract_label_no(text)
        if not label_no or index >= len(boxes):
            continue
        language = _caption_language(text)
        key = (label_no.casefold(), language)
        if key in existing:
            continue
        raw_box = list(boxes[index])
        if len(raw_box) != 4:
            continue
        x1, y1, x2, y2 = raw_box
        bbox_pixel = [
            round(x1 * scale_x),
            round(y1 * scale_y),
            round(x2 * scale_x),
            round(y2 * scale_y),
        ]
        bbox = [
            round(bbox_pixel[0] / page_width, 5),
            round(bbox_pixel[1] / page_height, 5),
            round(bbox_pixel[2] / page_width, 5),
            round(bbox_pixel[3] / page_height, 5),
        ]
        confidence = float(scores[index]) if index < len(scores) else 0.0
        out.append(
            Block(
                block_id=f"{document_id}_P{page_num:03d}_O{index:02d}",
                document_id=document_id,
                page=page_num,
                block_type="caption",
                order=None,
                bbox=bbox,
                bbox_pixel=bbox_pixel,
                text=text,
                source_method="ocr",
                confidence=confidence,
                label_norm=text,
                label_no=label_no,
                caption_language=language,
                raw_label="ocr_caption_fallback",
            )
        )
        existing.add(key)


def _append_layout_figure_fallbacks(
    page_res: dict,
    blocks_raw: list[dict],
    out: list[Block],
    document_id: str,
    page_num: int,
    page_width: int,
    page_height: int,
    scale_x: float,
    scale_y: float,
    min_score: float,
) -> None:
    """补回 layout 已检出、但 parsing_res_list 漏掉的高置信度图片。"""
    parsed_figure_boxes = [
        list(block.get("block_bbox", []))
        for block in blocks_raw
        if LABEL_TO_TYPE.get(block.get("block_label", "")) == "figure"
    ]
    layout_boxes = page_res.get("layout_det_res", {}).get("boxes", [])

    for layout_index, candidate in enumerate(layout_boxes):
        if candidate.get("label") not in {"image", "chart"}:
            continue
        score = float(candidate.get("score", 0.0))
        coordinate = list(candidate.get("coordinate", []))
        if len(coordinate) != 4:
            continue
        if any(_bbox_iou(coordinate, box) >= 0.80 for box in parsed_figure_boxes):
            continue

        raw_box = [int(value) for value in coordinate]
        x1, y1, x2, y2 = raw_box
        bbox_pixel = [
            round(x1 * scale_x),
            round(y1 * scale_y),
            round(x2 * scale_x),
            round(y2 * scale_y),
        ]
        bbox = [
            round(bbox_pixel[0] / page_width, 5),
            round(bbox_pixel[1] / page_height, 5),
            round(bbox_pixel[2] / page_width, 5),
            round(bbox_pixel[3] / page_height, 5),
        ]
        caption_supported = any(
            block.block_type == "caption"
            and _spatial_bbox_distance(bbox, block.bbox) <= 0.15
            for block in out
        )
        if score < min_score and not (
            score >= max(0.0, min_score - 0.15) and caption_supported
        ):
            continue
        image_path = (
            "assets/imgs/img_in_image_box_"
            f"{x1}_{y1}_{x2}_{y2}.jpg"
        )
        block_id = f"{document_id}_P{page_num:03d}_L{layout_index:02d}"
        out.append(
            Block(
                block_id=block_id,
                document_id=document_id,
                page=page_num,
                block_type="figure",
                order=_infer_layout_order(bbox_pixel, out),
                bbox=bbox,
                bbox_pixel=bbox_pixel,
                text="",
                source_method="layout",
                confidence=score,
                image_crop=image_path,
                image_crop_raw=image_path,
                raw_label=candidate.get("label"),
            )
        )


def _spatial_bbox_distance(first: list[float], second: list[float]) -> float:
    """与 caption 配对一致的距离度量：垂直为主，跨栏增加惩罚。"""
    if first[3] < second[1]:
        vertical_gap = second[1] - first[3]
    elif second[3] < first[1]:
        vertical_gap = first[1] - second[3]
    else:
        vertical_gap = 0.0
    horizontal_gap = max(0.0, max(first[0], second[0]) - min(first[2], second[2]))
    return vertical_gap + horizontal_gap * 2


def _bbox_iou(first: list, second: list) -> float:
    if len(first) != 4 or len(second) != 4:
        return 0.0
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _infer_layout_order(bbox_pixel: list[int], blocks: list[Block]) -> int | None:
    """用同栏、位于候选上方的最近有序块作为阅读顺序锚点。"""
    x1, y1, x2, _ = bbox_pixel
    anchors: list[tuple[int, int]] = []
    for block in blocks:
        if block.order is None or block.block_type in {"figure", "table", "formula"}:
            continue
        bx1, _, bx2, by2 = block.bbox_pixel
        horizontal_overlap = min(x2, bx2) - max(x1, bx1)
        if horizontal_overlap > 0 and by2 <= y1:
            anchors.append((by2, block.order))
    return max(anchors, default=(0, None), key=lambda item: item[0])[1]
