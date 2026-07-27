"""块间关系：caption配对、section_path、交叉引用。

对应 pdf-parser.md §6：
- §6.1 caption <-> figure/table 配对
- §6.2 标题编号层级 section_path（沿阅读顺序传递）
- §6.3 交叉引用索引（正文"如图3所示"-> figure block_id）

所有关系在全文范围建立（跨页：正文在第3页引用第5页的图也能命中）。
"""
from __future__ import annotations

import re

from src.schema import Block

# ---------- §6.1 caption <-> figure/table 配对 ----------

# 标题编号索引：label_no(如"3") -> caption block_id
# 注意：图和表编号独立（图3、表3 各算各的），用 ("fig"/"table", no) 作 key
_CAP_LABEL_NO_RE = re.compile(r"^(图|表|Fig\.?|Table)\s*([0-9A-Za-z\-]+)", re.IGNORECASE)


def _caption_kind(text: str) -> str | None:
    """判断 caption 是图还是表。"""
    if not text:
        return None
    if re.match(r"^表|^Table", text, re.IGNORECASE):
        return "table"
    if re.match(r"^图|^Fig", text, re.IGNORECASE):
        return "figure"
    return None


def build_caption_index(blocks: list[Block]) -> dict:
    """建 {(kind, label_no): caption_block_id} 索引，并回填 caption.label_no。

    label_no 已在 normalize 抽取；这里补建索引。
    """
    idx: dict[tuple[str, str], str] = {}
    for b in blocks:
        if b.block_type != "caption":
            continue
        if not b.label_no:
            m = _CAP_LABEL_NO_RE.match(b.text)
            if m:
                b.label_no = m.group(2)
        kind = _caption_kind(b.text)
        if b.label_no and kind:
            idx[(kind, b.label_no)] = b.block_id
    return idx


def pair_captions(blocks: list[Block]) -> None:
    """把 caption 配到最近的 figure/table（按空间相邻 + 阅读顺序）。

    策略（pdf-parser.md §6.1 取法B）：
    caption 通常紧邻 figure/table（正上方或正下方）。
    按阅读顺序遍历，遇到 caption 就向前后找同页最近的 figure/table。
    """
    # 按页分组 + order 排序（order=None 的用 block_id 兜底排序）
    by_page: dict[int, list[Block]] = {}
    for b in blocks:
        by_page.setdefault(b.page, []).append(b)
    for page, pblocks in by_page.items():
        pblocks.sort(key=lambda b: (b.order if b.order is not None else 9999, b.block_id))
        n = len(pblocks)
        for i, b in enumerate(pblocks):
            if b.block_type != "caption":
                continue
            # 向前后各找最近的 figure/table（同页，空间重叠或紧邻）
            target = _find_nearest_figure(pblocks, i)
            if target:
                b.caption_of = target.block_id
                # 回填 figure 的 label_no/label_norm（若它没有）
                if not target.label_no and b.label_no:
                    target.label_no = b.label_no
                if b.label_norm:
                    target.label_norm = b.label_norm


def _find_nearest_figure(pblocks: list[Block], idx: int) -> Block | None:
    """在排序后的同页 blocks 中，找距 idx 最近的 figure/table。"""
    cap = pblocks[idx]
    cap_box = cap.bbox_pixel
    best: Block | None = None
    best_dist = float("inf")
    for j, b in enumerate(pblocks):
        if b.block_type not in ("figure", "table"):
            continue
        # 空间距离：用 bbox 中心点的距离（caption 在图正上/下方时很小）
        cb = b.bbox_pixel
        cx1, cy1 = (cap_box[0] + cap_box[2]) / 2, (cap_box[1] + cap_box[3]) / 2
        bx1, by1 = (cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2
        dist = abs(cx1 - bx1) + abs(cy1 - by1)
        # 水平重叠优先（caption 应在图正上/下方，x 范围重叠）
        x_overlap = min(cap_box[2], cb[2]) - max(cap_box[0], cb[0])
        if x_overlap > 0:  # 有水平重叠，距离打折
            dist *= 0.5
        if dist < best_dist:
            best_dist = dist
            best = b
    return best


# ---------- §6.2 标题编号层级 section_path ----------

# 编号正则：纯数字层级(5.3.2) / 附录A / A.4
_SECTION_RE = re.compile(r"^(\d+(?:\.\d+)*|附录\s*[A-Z]|[A-Z]\.\d+(?:\.\d+)*)")


def build_section_paths(blocks: list[Block]) -> None:
    """沿阅读顺序维护标题栈，给每个 block 写 section_path。

    遍历顺序：按 page 升序，页内按 order。遇到 heading 抽取编号压栈/弹栈，
    后续所有 block 继承当前栈。
    """
    stack: list[str] = []  # 当前标题编号栈
    sorted_blocks = sorted(
        blocks,
        key=lambda b: (b.page, b.order if b.order is not None else 9999, b.block_id),
    )
    for b in sorted_blocks:
        if b.block_type == "heading":
            no = _extract_heading_no(b.text)
            if no:
                _push_stack(stack, no)
        b.section_path = list(stack)


def _extract_heading_no(text: str) -> str | None:
    if not text:
        return None
    m = _SECTION_RE.match(text.strip())
    return m.group(1) if m else None


def _push_stack(stack: list[str], no: str) -> None:
    """按编号深度压栈：5.3.2 比 5.3 深，压入；比 5.3 浅，弹到同级再压。"""
    depth = no.count(".") + 1 if no[0].isdigit() else 1
    # 弹出比当前深的（同级或更浅的先弹）
    while stack:
        top = stack[-1]
        top_depth = top.count(".") + 1 if top[0].isdigit() else 1
        if top_depth >= depth:
            stack.pop()
        else:
            break
    stack.append(no)


# ---------- §6.3 交叉引用索引（正文 -> 图表/附录） ----------

# 文中引用标记：图3 / 表2 / 公式(2) / Fig.1 / Table 3 / 见附录A / 见第3.2节
_REF_PATTERNS = [
    (re.compile(r"图\s?([0-9A-Za-z\-]+)"), "figure"),
    (re.compile(r"表\s?([0-9A-Za-z\-]+)"), "table"),
    (re.compile(r"公式\s?\(?\s*([0-9A-Za-z\-]+)\s*\)?"), "formula"),
    (re.compile(r"Fig\.?\s*([0-9A-Za-z\-]+)", re.IGNORECASE), "figure"),
    (re.compile(r"Table\s+([0-9A-Za-z\-]+)", re.IGNORECASE), "table"),
    (re.compile(r"见附录\s*([A-Z])"), "appendix"),
    (re.compile(r"第\s*([\d\.]+)\s*节"), "section"),
]


def extract_references(blocks: list[Block]) -> None:
    """扫描正文块的文字，匹配引用标记，回填 references 字段。

    需 caption 索引建好后调用。引用命中时写入被引 figure/table 的 block_id。
    """
    cap_idx = build_caption_index(blocks)
    # 也给 figure/table 建一个 {(kind, no): block_id}，覆盖 caption 之外的 figure 自身编号
    fig_idx = {}
    for b in blocks:
        if b.block_type in ("figure", "table") and b.label_no:
            kind = b.block_type
            fig_idx[(kind, b.label_no)] = b.block_id
    ref_idx = {**cap_idx, **fig_idx}  # 优先用 caption 的指向（更全）

    for b in blocks:
        if b.block_type not in ("paragraph", "heading", "list", "footnote"):
            continue
        if not b.text:
            continue
        refs: list[str] = []
        seen = set()
        for pat, kind in _REF_PATTERNS:
            for m in pat.finditer(b.text):
                no = m.group(1).strip()
                key = (kind, no) if kind in ("figure", "table") else None
                if key and key in ref_idx:
                    bid = ref_idx[key]
                    if bid not in seen:
                        refs.append(bid)
                        seen.add(bid)
        b.references = refs


# ---------- 入口：一次跑完所有关系 ----------

def build_relations(blocks: list[Block]) -> None:
    """构建全部块间关系（原地修改 blocks）。顺序：caption -> section -> references。"""
    pair_captions(blocks)
    build_section_paths(blocks)
    extract_references(blocks)
