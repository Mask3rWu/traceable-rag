"""Deterministic cleanup rules for parsed document blocks.

The raw PP-StructureV3 result is retained separately. These rules only remove
non-content blocks from the normalized document view used by Markdown and
downstream retrieval.
"""
from __future__ import annotations

import re
from collections import defaultdict

from src.schema import Block, Document, Page


_INLINE_METADATA_RE = re.compile(
    r"^(?:\*?\s*)?(?:"
    r"doi\s*[:：]|收稿日期|修回日期|录用日期|网络出版(?:时间|地址)?|"
    r"作者简介|第一作者|通信作者|通讯作者|基金项目|本文网址|期刊网址|引用本文|"
    r"received\s*[:：]|revised\s*[:：]|accepted\s*[:：]|"
    r"published\s+online\s*[:：]|foundation\s+items\s*[:：]"
    r")",
    re.IGNORECASE,
)
_MARKDOWN_PREFIX_RE = re.compile(r"^[#\s\d.]+")
_MEANINGFUL_CHAR_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")


def is_table_of_contents_noise(block: Block) -> bool:
    """Return whether a layout ``content`` block is a table of contents."""
    return block.raw_label == "content"


def is_footnote_metadata_noise(block: Block) -> bool:
    """Return whether a publisher footnote should be excluded from body text."""
    return block.block_type == "footnote"


def is_inline_publication_metadata_noise(block: Block) -> bool:
    """Return whether a body block contains publication rather than subject metadata."""
    return bool(_INLINE_METADATA_RE.match(block.text.strip()))


def is_symbol_fragment_noise(block: Block) -> bool:
    """Return whether text has no Chinese, Latin, or numeric content.

    Figure labels such as ``a)`` and table values are intentionally preserved.
    """
    if block.block_type not in {"paragraph", "heading", "caption"}:
        return False
    text = _MARKDOWN_PREFIX_RE.sub("", block.text.strip()).strip()
    return bool(text) and not _MEANINGFUL_CHAR_RE.search(text)


def is_isolated_unscored_ocr_fragment(block: Block, page: Page) -> bool:
    """Return whether the only unscored page block is a short OCR artifact."""
    text = block.text.strip()
    return (
        len(page.blocks) == 1
        and block.block_type == "paragraph"
        and block.confidence == 0.0
        and bool(re.fullmatch(r"[A-Z]{2,4}\d*", text))
    )


def find_running_header_noise_ids(pages: list[Page]) -> set[str]:
    """Find repeated, alternating-margin running headers.

    A repeated sentence near a page edge is not enough: body text may happen to
    recur there. Running headers are additionally required to occupy one narrow
    vertical band and alternate between left and right page margins.
    """
    candidates: dict[str, list[Block]] = defaultdict(list)
    for page in pages:
        for block in page.blocks:
            text = " ".join(block.text.split())
            center_y = (block.bbox[1] + block.bbox[3]) / 2
            if text and len(text) <= 120 and center_y >= 0.84:
                candidates[text].append(block)

    noise_ids: set[str] = set()
    for blocks in candidates.values():
        if len({block.page for block in blocks}) < 3:
            continue
        centers_y = [(block.bbox[1] + block.bbox[3]) / 2 for block in blocks]
        centers_x = [(block.bbox[0] + block.bbox[2]) / 2 for block in blocks]
        if max(centers_y) - min(centers_y) > 0.025:
            continue
        if max(centers_x) - min(centers_x) < 0.3:
            continue
        noise_ids.update(block.block_id for block in blocks)
    return noise_ids


def filter_document_noise(document: Document) -> dict[str, int]:
    """Remove deterministic non-content blocks from a normalized document.

    Returns counts by category for logging and tests. Reference blocks are not
    deleted here; callers can exclude ``appendix`` blocks from their retrieval
    index while retaining bibliographic provenance in ``doc.json``.
    """
    running_header_ids = find_running_header_noise_ids(document.pages)
    counts = {
        "table_of_contents": 0,
        "footnote_metadata": 0,
        "running_header": 0,
        "inline_publication_metadata": 0,
        "symbol_fragment": 0,
        "isolated_unscored_fragment": 0,
    }

    for page in document.pages:
        retained: list[Block] = []
        for block in page.blocks:
            category = None
            if is_table_of_contents_noise(block):
                category = "table_of_contents"
            elif is_footnote_metadata_noise(block):
                category = "footnote_metadata"
            elif block.block_id in running_header_ids:
                category = "running_header"
            elif is_inline_publication_metadata_noise(block):
                category = "inline_publication_metadata"
            elif is_symbol_fragment_noise(block):
                category = "symbol_fragment"
            elif is_isolated_unscored_ocr_fragment(block, page):
                category = "isolated_unscored_fragment"

            if category is None:
                retained.append(block)
            else:
                counts[category] += 1
        page.blocks = retained
    return counts
