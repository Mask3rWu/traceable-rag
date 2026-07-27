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
_LABEL_NO = r"[0-9A-Za-z]+(?:[.\-][0-9A-Za-z]+)*"
_CAP_LABEL_NO_RE = re.compile(
    rf"^\s*(图|表|Fig\.?|Figure|Table)\s*({_LABEL_NO})", re.IGNORECASE
)


def _caption_kind(text: str) -> str | None:
    """判断 caption 是图还是表。"""
    if not text:
        return None
    text = text.lstrip()
    if re.match(r"^表|^Table", text, re.IGNORECASE):
        return "table"
    if re.match(r"^图|^Fig|^Figure", text, re.IGNORECASE):
        return "figure"
    return None


def _label_key(label_no: str) -> str:
    return label_no.strip().rstrip(".").casefold()


def build_caption_index(blocks: list[Block]) -> dict[tuple[str, str], str]:
    """建 ``{(kind, label_no): target_block_id}`` 索引。

    caption 已配对时索引必须指向 figure/table，而不是 caption 自己。
    未配对 caption 不进入索引，避免生成无法展开的伪关系。
    """
    idx: dict[tuple[str, str], str] = {}
    by_id = {b.block_id: b for b in blocks}
    for b in blocks:
        if b.block_type != "caption":
            continue
        if not b.label_no:
            m = _CAP_LABEL_NO_RE.match(b.text)
            if m:
                b.label_no = m.group(2)
        kind = _caption_kind(b.text)
        target = by_id.get(b.caption_of or "")
        if (
            b.label_no
            and kind
            and target is not None
            and target.block_type == kind
        ):
            idx[(kind, _label_key(b.label_no))] = target.block_id
    return idx


def pair_captions(blocks: list[Block]) -> None:
    """把 caption 配到最近的 figure/table（编号分组 + 一对一空间匹配）。

    同一图的中英文 caption 会共享编号，先合成一组。不同编号的 caption
    不能占用同一个图块；按组到图块的最小距离贪心匹配，避免缺图时把相邻
    图的标题误连到现存图块。
    """
    # 按页分组 + order 排序（order=None 的用 block_id 兜底排序）
    by_page: dict[int, list[Block]] = {}
    for b in blocks:
        by_page.setdefault(b.page, []).append(b)
    for pblocks in by_page.values():
        pblocks.sort(key=lambda b: (b.order if b.order is not None else 9999, b.block_id))
        targets = [b for b in pblocks if b.block_type in ("figure", "table")]
        groups: dict[tuple[str, str], list[Block]] = {}
        for caption in (b for b in pblocks if b.block_type == "caption"):
            kind = _caption_kind(caption.text)
            if not kind:
                continue
            group_no = _label_key(caption.label_no) if caption.label_no else caption.block_id
            groups.setdefault((kind, group_no), []).append(caption)

        candidates: list[tuple[float, tuple[str, str], Block]] = []
        for group_key, captions in groups.items():
            for target in targets:
                if target.block_type != group_key[0]:
                    continue
                distance = min(_spatial_distance(caption, target) for caption in captions)
                if distance <= 0.15:
                    candidates.append((distance, group_key, target))

        assigned_groups: set[tuple[str, str]] = set()
        assigned_targets: set[str] = set()
        for _, group_key, target in sorted(
            candidates, key=lambda item: (item[0], item[1], item[2].block_id)
        ):
            if group_key in assigned_groups or target.block_id in assigned_targets:
                continue
            captions = groups[group_key]
            for caption in captions:
                caption.caption_of = target.block_id
            primary = captions[0]
            target.label_no = primary.label_no or target.label_no
            target.label_norm = primary.label_norm or target.label_norm
            assigned_groups.add(group_key)
            assigned_targets.add(target.block_id)


def _spatial_distance(caption: Block, target: Block) -> float:
    """归一化空间距离；垂直间距为主，无水平重叠时增加列间惩罚。"""
    cap = caption.bbox
    box = target.bbox
    if cap[3] < box[1]:
        vertical_gap = box[1] - cap[3]
    elif box[3] < cap[1]:
        vertical_gap = cap[1] - box[3]
    else:
        vertical_gap = 0.0
    horizontal_gap = max(0.0, max(cap[0], box[0]) - min(cap[2], box[2]))
    return vertical_gap + horizontal_gap * 2


# ---------- §6.2 标题编号层级 section_path ----------

# 编号正则：纯数字层级(5.3.2) / 附录A / A.4
_SECTION_RE = re.compile(
    r"^(\d+(?:\.\d+)*|附录\s*[A-Z]|[A-Z]\.\d+(?:\.\d+)*)",
    re.IGNORECASE,
)
_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s*")
_APPENDIX_KIND_RE = re.compile(r"(规范性附录|资料性附录)")
_BLOCK_ID_SUFFIX_RE = re.compile(r"_B(\d+)$")


def build_section_paths(blocks: list[Block]) -> None:
    """沿阅读顺序维护标题栈，给每个 block 写 section_path。

    遍历顺序：按 page 升序，页内按 order。遇到 heading 抽取编号压栈/弹栈，
    后续所有 block 继承当前栈。
    """
    stack: list[str] = []  # 当前标题编号栈
    in_appendix = False
    appendix_type: str | None = None
    sorted_blocks = _reading_order(blocks)
    for b in sorted_blocks:
        if b.block_type == "heading":
            no = _extract_heading_no(b.text)
            if no:
                _push_stack(stack, no)
                if _is_appendix_no(no):
                    in_appendix = True
            kind = _APPENDIX_KIND_RE.search(b.text)
            if kind:
                appendix_type = kind.group(1)
                in_appendix = True
        b.section_path = list(stack)
        b.is_appendix = in_appendix
        b.appendix_type = appendix_type if in_appendix else None


def _reading_order(blocks: list[Block]) -> list[Block]:
    """将无 order 的图表插入相邻有序块之间，而不是统一放到页尾。"""
    by_page: dict[int, list[Block]] = {}
    for block in blocks:
        by_page.setdefault(block.page, []).append(block)

    result: list[Block] = []
    for page in sorted(by_page):
        page_blocks = by_page[page]
        anchors = sorted(
            (
                (source_id, block.order)
                for block in page_blocks
                if block.order is not None
                if (source_id := _source_id(block)) is not None
            ),
            key=lambda item: item[0],
        )

        def effective_order(block: Block) -> tuple[float, str]:
            if block.order is not None:
                return float(block.order), block.block_id
            source_id = _source_id(block)
            if source_id is None or not anchors:
                return 9999.0 + block.bbox[1], block.block_id
            previous = next((item for item in reversed(anchors) if item[0] < source_id), None)
            following = next((item for item in anchors if item[0] > source_id), None)
            if previous and following:
                span = following[0] - previous[0]
                ratio = (source_id - previous[0]) / span
                value = previous[1] + ratio * (following[1] - previous[1])
            elif previous:
                value = previous[1] + (source_id - previous[0]) * 0.01
            else:
                value = following[1] - (following[0] - source_id) * 0.01
            return float(value), block.block_id

        result.extend(sorted(page_blocks, key=effective_order))
    return result


def _source_id(block: Block) -> int | None:
    match = _BLOCK_ID_SUFFIX_RE.search(block.block_id)
    return int(match.group(1)) if match else None


def _extract_heading_no(text: str) -> str | None:
    if not text:
        return None
    cleaned = _MARKDOWN_HEADING_RE.sub("", text.strip())
    m = _SECTION_RE.match(cleaned)
    if not m:
        return None
    remainder = cleaned[m.end() :].lstrip()
    if remainder.startswith((")", "）", "、")):
        return None
    return re.sub(r"\s+", "", m.group(1))


def _is_appendix_no(no: str) -> bool:
    return no.upper().startswith("附录") or bool(re.match(r"^[A-Z](?:\.|$)", no, re.I))


def _push_stack(stack: list[str], no: str) -> None:
    """按编号深度压栈：5.3.2 比 5.3 深，压入；比 5.3 浅，弹到同级再压。"""
    if _is_appendix_no(no) and not any(_is_appendix_no(item) for item in stack):
        stack.clear()
    if no.startswith("附录"):
        depth = 1
    elif no[0].isalpha():
        depth = no.count(".") + 1
    else:
        depth = no.count(".") + 1
    # 弹出比当前深的（同级或更浅的先弹）
    while stack:
        top = stack[-1]
        top_depth = top.count(".") + 1
        if top_depth >= depth:
            stack.pop()
        else:
            break
    stack.append(no)


# ---------- §6.3 交叉引用索引（正文 -> 图表/附录） ----------

# 文中引用标记：图3 / 表2 / 公式(2) / Fig.1 / Table 3 / 见附录A / 见第3.2节
_REF_PATTERNS = [
    (re.compile(rf"图\s*({_LABEL_NO})"), "figure"),
    (re.compile(rf"表\s*({_LABEL_NO})"), "table"),
    (re.compile(rf"公式\s*\(?\s*({_LABEL_NO})\s*\)?"), "formula"),
    (re.compile(rf"(?:Fig\.?|Figure)\s*({_LABEL_NO})", re.IGNORECASE), "figure"),
    (re.compile(rf"Table\s*({_LABEL_NO})", re.IGNORECASE), "table"),
    (re.compile(r"见附录\s*([A-Z])"), "appendix"),
    (re.compile(r"第\s*([\d\.]+)\s*节"), "section"),
]


def extract_references(blocks: list[Block]) -> None:
    """扫描正文块的文字，匹配引用标记，回填 references 字段。

    需 caption 索引建好后调用。引用命中时写入被引 figure/table 的 block_id。
    """
    ref_idx = build_caption_index(blocks)
    # 也给已有自身编号的图表建立索引。
    for b in blocks:
        if b.block_type in ("figure", "table") and b.label_no:
            ref_idx[(b.block_type, _label_key(b.label_no))] = b.block_id

    for b in blocks:
        if b.block_type == "heading":
            no = _extract_heading_no(b.text)
            if no:
                ref_idx[("section", no.casefold())] = b.block_id
                if no.startswith("附录"):
                    ref_idx[("appendix", no[2:].casefold())] = b.block_id
                else:
                    appendix = re.match(r"^([A-Z])(?:\.|$)", no, re.I)
                    if appendix:
                        ref_idx.setdefault(
                            ("appendix", appendix.group(1).casefold()), b.block_id
                        )

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
                key = (kind, _label_key(no))
                if key in ref_idx:
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
