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
        r"^\s*(?:图|表|Fig\.?|Table)\s*([0-9A-Za-z\-]+)",
        text,
        re.IGNORECASE,
    )
    return m.group(1) if m else None


def normalize_page_blocks(
    page_res: dict,
    document_id: str,
    page_num: int,  # 1-based
    page_width: int,
    page_height: int,
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
            label_no=label_no,
            raw_label=label,
        )
        if block_type == "caption":
            blk.label_norm = text.strip()
        out.append(blk)

    return out
