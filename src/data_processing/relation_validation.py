"""Validate parser relations and emit non-blocking, auditable warnings."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.schema import Block


REPORT_NAME = "relation_validation.jsonl"
RELATION_FIELDS = (
    "references",
    "caption_of",
    "caption_ids",
    "continuation_of",
    "continues_to",
)
VISUAL_TYPES = {"figure", "table"}


def _targets(block: Block, field: str) -> list[str]:
    value = getattr(block, field)
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def _add_flag(block: Block, code: str) -> None:
    if code not in block.quality_flags:
        block.quality_flags.append(code)


def _compatible_sections(left: Block, right: Block) -> bool:
    """Unknown paths are allowed; two known, different paths are not."""
    return not (left.section_path and right.section_path and left.section_path != right.section_path)


def validate_relations(blocks: Iterable[Block]) -> list[dict[str, object]]:
    """Return and annotate recoverable relation defects without dropping blocks."""
    values = list(blocks)
    by_id = {block.block_id: block for block in values}
    issues: list[dict[str, object]] = []
    reported: set[tuple[str, ...]] = set()

    def report(code: str, source: Block, target_id: str | None = None) -> None:
        _add_flag(source, code)
        item: dict[str, object] = {
            "code": code,
            "severity": "warning",
            "source_block_id": source.block_id,
            "source_page": source.page,
        }
        if target_id:
            item["target_block_id"] = target_id
            target = by_id.get(target_id)
            if target is not None:
                item["target_page"] = target.page
                _add_flag(target, code)
        relation_key = (code, *sorted((source.block_id, target_id or "")))
        if relation_key in reported:
            return
        reported.add(relation_key)
        issues.append(item)

    for block in values:
        for field in RELATION_FIELDS:
            for target_id in _targets(block, field):
                if target_id not in by_id:
                    report(f"invalid_relation:{field}", block, target_id)

        if block.block_type == "caption" and block.caption_of:
            target = by_id.get(block.caption_of)
            if target is None:
                continue
            if target.block_type not in VISUAL_TYPES:
                report("caption_target_type_mismatch", block, target.block_id)
            elif block.block_id not in target.caption_ids:
                report("caption_backlink_missing", block, target.block_id)
            if not _compatible_sections(block, target):
                report("cross_section_caption", block, target.block_id)

        if block.block_type in VISUAL_TYPES:
            for caption_id in block.caption_ids:
                caption = by_id.get(caption_id)
                if caption is None:
                    continue
                if caption.block_type != "caption":
                    report("caption_member_type_mismatch", block, caption_id)
                elif caption.caption_of != block.block_id:
                    report("caption_backlink_missing", block, caption_id)
                if not _compatible_sections(block, caption):
                    report("cross_section_caption", block, caption_id)

        for target_id in (block.continuation_of, block.continues_to):
            target = by_id.get(target_id or "")
            if target is not None and not _compatible_sections(block, target):
                report("cross_section_continuation", block, target_id)

    return issues


def write_relation_validation_report(path: Path, issues: Iterable[dict[str, object]]) -> None:
    """Write one machine-readable warning per line, including an empty report."""
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for issue in issues:
            stream.write(json.dumps(issue, ensure_ascii=False) + "\n")
