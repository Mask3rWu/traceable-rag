"""从增强后的 Document 生成与 doc.json 一致的人工审阅 Markdown。"""
from __future__ import annotations

import os
import re
from pathlib import Path

from src.paths import PROJECT_ROOT
from src.schema import Block, Document
from src.data_processing.relations import reading_order_blocks

# 剥离 PP-StructureV3 给标题加的 # 前缀；层级改由 section_path 深度决定，
# 避免 PP 把括号编号(1）/2）)误标为同级 ## 造成层级倒挂。
_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s*")


def write_document_markdown(document: Document, out_dir: Path) -> Path:
    """写出最终 ``doc.md``；原始 ``structure.md`` 继续作为模型留底。"""
    out_dir = Path(out_dir)
    blocks = [block for page in document.pages for block in page.blocks]
    by_id = {block.block_id: block for block in blocks}
    rendered: list[str] = []
    current_page: int | None = None

    for block in reading_order_blocks(blocks):
        if block.continuation_of:
            continue
        if block.block_type == "caption" and block.caption_of:
            continue
        if block.page != current_page:
            rendered.append(f"<!-- page: {block.page} -->")
            current_page = block.page

        chain = _continuation_chain(block, by_id)
        piece = _render_chain(chain, out_dir, by_id)
        if piece:
            rendered.append(piece)

    path = out_dir / "doc.md"
    path.write_text("\n\n".join(rendered).rstrip() + "\n", encoding="utf-8")
    return path


def _continuation_chain(root: Block, by_id: dict[str, Block]) -> list[Block]:
    chain = [root]
    seen = {root.block_id}
    current = root
    while current.continues_to:
        following = by_id.get(current.continues_to)
        if following is None or following.block_id in seen:
            break
        chain.append(following)
        seen.add(following.block_id)
        current = following
    return chain


def _render_chain(
    chain: list[Block],
    out_dir: Path,
    by_id: dict[str, Block],
) -> str:
    block = chain[0]
    if block.block_type in {"paragraph", "appendix", "footnote"}:
        text = chain[0].text.strip()
        for following in chain[1:]:
            text = _join_text(text, following.text.strip())
        comments = []
        if len(chain) > 1:
            sources = ", ".join(item.block_id for item in chain)
            comments.append(f"<!-- continued-sources: {sources} -->")
        flags = sorted({flag for item in chain for flag in item.quality_flags})
        if flags:
            comments.append(f"<!-- quality-flags: {', '.join(flags)} -->")
        comments.append(text)
        return "\n".join(comments)

    if block.block_type == "heading":
        text = _MARKDOWN_HEADING_RE.sub("", block.text.strip())
        if not text:
            return ""
        # 层级 = section_path 深度 + 1（文档标题深度 0 -> #，4.2.1 深度 3 -> ####，
        # 其下括号编号 1） 深度 4 -> #####），不信任 PP-StructureV3 给的 # 数量。
        level = max(1, min(len(block.section_path) + 1, 6))
        return f"{'#' * level} {text}"
    if block.block_type == "list":
        return block.text.strip()
    if block.block_type == "formula":
        parts = [block.text.strip()]
        if block.formula_no:
            parts.append(f'<div align="right">({block.formula_no})</div>')
        return "\n\n".join(parts)
    if block.block_type in {"figure", "table"}:
        if not block.image_crop:
            return ""
        alt = block.label_norm or f"{block.block_type} {block.label_no or ''}".strip()
        parts = [f"![{alt}]({_relative_asset(block.image_crop, out_dir)})"]
        for caption_id in block.caption_ids:
            caption = by_id.get(caption_id)
            if caption is not None:
                parts.append(_render_caption(caption))
        return "\n\n".join(parts)
    if block.block_type == "caption":
        return _render_caption(block)
    return block.text.strip()


def _join_text(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    separator = (
        " "
        if left[-1].isascii()
        and right[0].isascii()
        and left[-1] != "$"
        and right[0] != "$"
        else ""
    )
    return left + separator + right


def _render_caption(block: Block) -> str:
    language = f' lang="{block.caption_language}"' if block.caption_language else ""
    return f'<div align="center"{language}>{block.text.strip()}</div>'


def _relative_asset(stored_path: str, out_dir: Path) -> str:
    path = Path(stored_path)
    absolute = path if path.is_absolute() else PROJECT_ROOT / path
    relative = os.path.relpath(absolute.resolve(), out_dir.resolve())
    return relative.replace("\\", "/")
