"""数据模型（pydantic v2）。

Block schema 对齐 数据处理.md 与 pdf-parser.md §5.2。
block_type 统一：heading / paragraph / list / table / formula / figure / caption / appendix / footnote
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# block_type 取值（统一归一后的类型）
BLOCK_TYPES = {
    "heading",
    "paragraph",
    "list",
    "table",
    "formula",
    "figure",
    "caption",
    "appendix",
    "footnote",
}


class Block(BaseModel):
    """内容块：解析层的最小可寻址单元。"""

    block_id: str  # 全局稳定 ID: {document_id}_P{page:03d}_B{order:02d}
    document_id: str
    page: int  # 1-based 页码

    block_type: str  # 见 BLOCK_TYPES
    order: Optional[int] = None  # 页内阅读顺序（0-based）

    # 坐标（同时保留像素与归一化，便于回溯与裁剪）
    bbox: list[float]  # 归一化 [xmin,ymin,xmax,ymax] in [0,1]
    bbox_pixel: list[int]  # 原始像素坐标
    polygon_points: Optional[list[list[int]]] = None  # 实例分割掩码点

    # 内容
    text: str = ""  # OCR文本/表格HTML/公式LaTeX
    source_method: str = "ocr"  # ocr / native / layout / unreadable
    confidence: float = 0.0

    # 文档结构
    section_path: list[str] = Field(default_factory=list)  # 所属标题编号层级
    is_appendix: bool = False
    appendix_type: Optional[str] = None  # 规范性附录 / 资料性附录

    # 图片：裁剪出的子图路径（相对项目根），用于 MLLM 与回显
    image_crop: Optional[str] = None
    image_crop_raw: Optional[str] = None  # Paddle 按原始检测框生成的裁图
    crop_bbox_pixel: Optional[list[int]] = None  # 扩边后在 page_image 上的实际裁剪框
    figure_crop: Optional[str] = None  # 图片主体 + 全部图注的人工复核裁图
    figure_crop_bbox_pixel: Optional[list[int]] = None

    # 图表标题相关（见 relations.py）
    label_norm: Optional[str] = None  # 如 "图3 毁伤评估流程图"
    label_no: Optional[str] = None  # 如 "3"
    caption_of: Optional[str] = None  # caption 指向的 figure/table block_id
    caption_ids: list[str] = Field(default_factory=list)  # figure/table 的全部图注
    caption_language: Optional[str] = None  # zh / en / unknown
    formula_no: Optional[str] = None  # 公式编号，如 "1"

    # 跨栏/跨页逻辑续接。原始块不合并，最终视图按关系拼接。
    continuation_of: Optional[str] = None
    continues_to: Optional[str] = None
    quality_flags: list[str] = Field(default_factory=list)

    # 交叉引用：本块正文引用的图表 block_id 列表
    references: list[str] = Field(default_factory=list)

    # 原始标签（PP-StructureV3 label，便于回溯/重跑）
    raw_label: Optional[str] = None


class Page(BaseModel):
    """单页解析结果。"""

    document_id: str
    page: int  # 1-based
    width: int  # 像素宽
    height: int  # 像素高
    render_dpi: Optional[int] = None  # 页图实际 DPI；扫描页可能低于请求值
    has_text_layer: bool = False  # 是否有原生文本层
    page_image: Optional[str] = None  # pages/pXXX.png 路径（相对项目根）
    watermark_detected: bool = False
    watermark_type: Optional[str] = None
    watermark_ratio: float = 0.0
    watermark_bbox: Optional[list[float]] = None
    blocks: list[Block] = Field(default_factory=list)


class Document(BaseModel):
    """单文档解析结果（解析层最终产物 doc.json）。"""

    document_id: str
    source_file: str  # 原 PDF 文件名
    total_pages: int
    pages: list[Page] = Field(default_factory=list)

    # 统计
    @property
    def block_count(self) -> int:
        return sum(len(p.blocks) for p in self.pages)
