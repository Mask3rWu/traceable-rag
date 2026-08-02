"""Build traceable retrieval chunks from the parser's ``doc.json`` output.

Chunking is deliberately downstream of parsing.  It never mutates ``doc.json``
and treats optional MLLM output as retrieval-only enrichment.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.paths import PROJECT_ROOT
from src.schema import Block, Chunk, ChunkVisualAsset, Document


OUTPUT_NAME = "chunks.jsonl"
VISUAL_OUTPUT_NAME = "visual_enrichment.json"
VISUAL_TYPES = {"figure", "table"}
TEXT_OVERLAP_TYPES = {"paragraph", "list", "appendix", "footnote"}
RELATION_FIELDS = (
    "references",
    "caption_of",
    "caption_ids",
    "continuation_of",
    "continues_to",
)


@dataclass(frozen=True)
class ChunkConfig:
    """Character-based P0 limits; structured units may exceed ``max_chars``."""

    target_chars: int = 600
    max_chars: int = 800
    overlap_chars: int = 80

    def __post_init__(self) -> None:
        if self.target_chars < 1:
            raise ValueError("target_chars must be positive")
        if self.max_chars < self.target_chars:
            raise ValueError("max_chars must be greater than or equal to target_chars")
        if self.overlap_chars < 0:
            raise ValueError("overlap_chars must be non-negative")


@dataclass
class _Unit:
    blocks: list[Block]
    fragments: list[str]

    @property
    def section_path(self) -> list[str]:
        return list(self.blocks[0].section_path) if self.blocks else []

    @property
    def text(self) -> str:
        return "\n\n".join(part.strip() for part in self.fragments if part.strip())


class _DisjointSet:
    def __init__(self, ids: Iterable[str]):
        self.parent = {value: value for value in ids}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            value, self.parent[value] = self.parent[value], root
        return root

    def union(self, left: str, right: str) -> None:
        if left not in self.parent or right not in self.parent:
            return
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _all_blocks(document: Document) -> list[Block]:
    return sorted(
        (block for page in document.pages for block in page.blocks),
        key=lambda block: (
            block.page,
            block.order is None,
            block.order if block.order is not None else 0,
            block.block_id,
        ),
    )


def _relation_targets(block: Block, field: str) -> list[str]:
    value = getattr(block, field)
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def validate_document(document: Document) -> dict[str, list[str]]:
    """Return block-scoped warnings instead of rejecting recoverable defects."""
    warnings: dict[str, list[str]] = {}
    blocks = _all_blocks(document)
    counts: dict[str, int] = {}
    for block in blocks:
        counts[block.block_id] = counts.get(block.block_id, 0) + 1
    known = set(counts)

    for block in blocks:
        flags = warnings.setdefault(block.block_id, [])
        if counts[block.block_id] > 1:
            flags.append("duplicate_block_id")
        if block.document_id != document.document_id:
            flags.append("document_id_mismatch")
        if block.page < 1 or len(block.bbox) != 4 or len(block.bbox_pixel) != 4:
            flags.append("invalid_source_location")
        for field in RELATION_FIELDS:
            if any(target not in known for target in _relation_targets(block, field)):
                flags.append(f"invalid_relation:{field}")
        if not flags:
            warnings.pop(block.block_id, None)
    return warnings


def _load_visual_enrichment(path: Path | None) -> tuple[dict[str, dict[str, Any]], bool]:
    if path is None or not path.is_file():
        return {}, False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, False
    items = payload.get("items", [])
    if not isinstance(items, list):
        return {}, False
    return {
        str(item["block_id"]): item
        for item in items
        if isinstance(item, dict) and item.get("block_id")
    }, True


def _load_document(doc_json: Path) -> tuple[Document, list[str]]:
    """Validate blocks independently so one malformed block cannot drop a document."""
    payload = json.loads(doc_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"doc.json top level must be an object: {doc_json}")
    pages = payload.get("pages")
    if not isinstance(pages, list):
        raise ValueError(f"doc.json pages must be a list: {doc_json}")

    warnings: list[str] = []
    clean_pages: list[dict[str, Any]] = []
    for page_index, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            warnings.append(f"invalid_page_skipped:{page_index}")
            continue
        clean_page = dict(page)
        clean_blocks: list[Block] = []
        raw_blocks = page.get("blocks", [])
        if not isinstance(raw_blocks, list):
            warnings.append(f"invalid_page_blocks:{page.get('page', page_index)}")
            raw_blocks = []
        for block_index, raw_block in enumerate(raw_blocks, start=1):
            try:
                clean_blocks.append(Block.model_validate(raw_block))
            except Exception:
                identity = (
                    raw_block.get("block_id")
                    if isinstance(raw_block, dict)
                    else f"page{page.get('page', page_index)}_block{block_index}"
                )
                warnings.append(f"invalid_block_skipped:{identity}")
        clean_page["blocks"] = clean_blocks
        clean_pages.append(clean_page)
    clean_payload = dict(payload)
    clean_payload["pages"] = clean_pages
    return Document.model_validate(clean_payload), warnings


def _resolve_asset(doc_json: Path, value: str) -> Path | None:
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [doc_json.parent / raw, PROJECT_ROOT / raw]
    return next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)


def _build_units(blocks: list[Block]) -> list[_Unit]:
    """Bind relations whose members must not be separated across chunks."""
    by_id = {block.block_id: block for block in blocks}
    groups = _DisjointSet(by_id)
    for block in blocks:
        for target in (block.continuation_of, block.continues_to):
            if target:
                groups.union(block.block_id, target)
        if block.block_type in VISUAL_TYPES:
            for caption_id in block.caption_ids:
                groups.union(block.block_id, caption_id)
        if block.block_type == "caption" and block.caption_of:
            groups.union(block.block_id, block.caption_of)

    # A heading carries the first semantic unit of its own section.
    for index, block in enumerate(blocks):
        if block.block_type != "heading":
            continue
        for following in blocks[index + 1 :]:
            if following.block_type == "heading":
                break
            if following.section_path == block.section_path:
                groups.union(block.block_id, following.block_id)
                break

    members: dict[str, list[Block]] = {}
    for block in blocks:
        members.setdefault(groups.find(block.block_id), []).append(block)
    position = {block.block_id: index for index, block in enumerate(blocks)}
    result = []
    for values in members.values():
        values.sort(key=lambda block: position[block.block_id])
        result.append(_Unit(values, [block.text for block in values if block.text.strip()]))
    result.sort(key=lambda unit: min(position[block.block_id] for block in unit.blocks))
    return result


_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;])|(?<=\.)\s+")


def _split_long_unit(unit: _Unit, max_chars: int) -> list[_Unit]:
    if len(unit.blocks) != 1 or unit.blocks[0].block_type not in TEXT_OVERLAP_TYPES:
        return [unit]
    text = unit.text
    if len(text) <= max_chars:
        return [unit]
    sentences = [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]
    fragments: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                fragments.append(current)
                current = ""
            fragments.extend(sentence[i : i + max_chars] for i in range(0, len(sentence), max_chars))
        elif current and len(current) + len(sentence) > max_chars:
            fragments.append(current)
            current = sentence
        else:
            current += sentence
    if current:
        fragments.append(current)
    return [_Unit(unit.blocks, [fragment]) for fragment in fragments]


def _same_section(left: _Unit, right: _Unit) -> bool:
    return left.section_path == right.section_path


def _pack_units(units: list[_Unit], config: ChunkConfig) -> list[list[_Unit]]:
    expanded = [part for unit in units for part in _split_long_unit(unit, config.max_chars)]
    packed: list[list[_Unit]] = []
    current: list[_Unit] = []
    current_length = 0
    for unit in expanded:
        addition = len(unit.text) + (2 if current and unit.text else 0)
        crosses_section = bool(current) and not _same_section(current[0], unit)
        exceeds_limit = bool(current) and current_length + addition > config.max_chars
        if crosses_section or exceeds_limit:
            packed.append(current)
            current, current_length = [], 0
            addition = len(unit.text)
        current.append(unit)
        current_length += addition
        if current_length >= config.target_chars:
            packed.append(current)
            current, current_length = [], 0
    if current:
        packed.append(current)
    return packed


def _overlap_for(units: list[_Unit], limit: int) -> tuple[str, list[str]]:
    if limit <= 0:
        return "", []
    candidates: list[tuple[str, str]] = []
    for unit in reversed(units):
        if len(unit.blocks) == 1 and unit.blocks[0].block_type in TEXT_OVERLAP_TYPES:
            candidates.append((unit.blocks[0].block_id, unit.text.strip()))
            continue
        for block in reversed(unit.blocks):
            if block.block_type in TEXT_OVERLAP_TYPES and block.text.strip():
                candidates.append((block.block_id, block.text.strip()))
    if not candidates:
        return "", []
    block_id, text = candidates[0]
    tail = text[-limit:]
    boundary = min((pos for pos in (tail.find("。"), tail.find("；")) if pos >= 0), default=-1)
    if boundary >= 0 and boundary + 1 < len(tail):
        tail = tail[boundary + 1 :].lstrip()
    return tail, [block_id] if tail else []


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _unique_blocks(blocks: Iterable[Block]) -> list[Block]:
    result: list[Block] = []
    seen: set[str] = set()
    for block in blocks:
        if block.block_id not in seen:
            seen.add(block.block_id)
            result.append(block)
    return result


def build_chunks(
    document: Document,
    *,
    config: ChunkConfig | None = None,
    visual_items: dict[str, dict[str, Any]] | None = None,
    visual_file_available: bool = False,
    available_assets: set[str] | None = None,
    document_quality_flags: Iterable[str] = (),
) -> list[Chunk]:
    """Create section-bounded chunks with complete parser provenance."""
    config = config or ChunkConfig()
    visual_items = visual_items or {}
    blocks = _all_blocks(document)
    by_id = {block.block_id: block for block in blocks}
    warnings = validate_document(document)
    packed = _pack_units(_build_units(blocks), config)
    chunks: list[Chunk] = []
    previous_units: list[_Unit] = []

    for index, units in enumerate(packed, start=1):
        primary_blocks = _unique_blocks(block for unit in units for block in unit.blocks)
        text = "\n\n".join(unit.text for unit in units if unit.text)
        references = _unique(ref for block in primary_blocks for ref in block.references)
        flags = _unique(
            flag
            for block in primary_blocks
            for flag in [*block.quality_flags, *warnings.get(block.block_id, [])]
        )
        visual_assets: list[ChunkVisualAsset] = []
        descriptions: list[str] = []
        linked_visuals: list[tuple[Block, str]] = [
            (block, "contained")
            for block in primary_blocks
            if block.block_type in VISUAL_TYPES
        ]
        contained_visual_ids = {block.block_id for block, _ in linked_visuals}
        linked_visuals.extend(
            (by_id[reference], "referenced")
            for reference in references
            if reference in by_id
            and by_id[reference].block_type in VISUAL_TYPES
            and reference not in contained_visual_ids
        )
        for block, relation in linked_visuals:
            item = visual_items.get(block.block_id, {})
            description = item.get("description") if item.get("status") == "ok" else None
            crop_available = bool(block.image_crop) and (
                available_assets is None or block.image_crop in available_assets
            )
            status = "ok" if description and crop_available else "unavailable"
            if not crop_available or not visual_file_available or not description:
                flags.append("visual_unavailable")
            if description:
                descriptions.append(str(description).strip())
            visual_assets.append(
                ChunkVisualAsset(
                    block_id=block.block_id,
                    block_type=block.block_type,
                    page=block.page,
                    relation=relation,
                    image_crop=block.image_crop,
                    description=description,
                    status=status,
                )
            )
        overlap_text, overlap_ids = _overlap_for(previous_units, config.overlap_chars)
        chunks.append(
            Chunk(
                chunk_id=f"{document.document_id}_C{index:05d}",
                document_id=document.document_id,
                text=text,
                visual_text="\n".join(_unique(descriptions)),
                overlap_text=overlap_text,
                block_ids=_unique(block.block_id for block in primary_blocks),
                overlap_block_ids=overlap_ids,
                page_start=min(block.page for block in primary_blocks),
                page_end=max(block.page for block in primary_blocks),
                section_path=list(units[0].section_path),
                references=references,
                source_file=document.source_file,
                visual_assets=visual_assets,
                quality_flags=_unique(flags),
            )
        )
        previous_units = units
    if chunks:
        chunks[0].quality_flags = _unique(
            [*document_quality_flags, *chunks[0].quality_flags]
        )
    return chunks


def chunk_document(
    doc_json: Path,
    *,
    output_path: Path | None = None,
    visual_path: Path | None = None,
    config: ChunkConfig | None = None,
) -> list[Chunk]:
    """Load one parsed document and atomically replace its JSONL chunk view."""
    doc_json = Path(doc_json)
    document, document_warnings = _load_document(doc_json)
    visual_path = visual_path or doc_json.parent / VISUAL_OUTPUT_NAME
    visual_items, visual_available = _load_visual_enrichment(visual_path)
    available_assets = {
        block.image_crop
        for block in _all_blocks(document)
        if block.image_crop and _resolve_asset(doc_json, block.image_crop) is not None
    }
    chunks = build_chunks(
        document,
        config=config,
        visual_items=visual_items,
        visual_file_available=visual_available,
        available_assets=available_assets,
        document_quality_flags=document_warnings,
    )
    output_path = output_path or doc_json.parent / OUTPUT_NAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as stream:
        for chunk in chunks:
            stream.write(json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False) + "\n")
    temp_path.replace(output_path)
    return chunks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从 doc.json 生成章节约束的可溯源 chunks.jsonl")
    parser.add_argument("doc", nargs="*", type=Path, help="一个或多个 doc.json")
    parser.add_argument("--all", action="store_true", help="处理 parsed-root 下全部 doc.json")
    parser.add_argument("--parsed-root", type=Path, default=Path("processed/parsed"))
    parser.add_argument("--target-chars", type=int, default=600)
    parser.add_argument("--max-chars", type=int, default=800)
    parser.add_argument("--overlap-chars", type=int, default=80)
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    args = build_parser().parse_args(argv)
    docs = list(args.doc)
    if args.all:
        docs.extend(sorted(args.parsed_root.glob("*/doc.json")))
    docs = list(dict.fromkeys(path.resolve() for path in docs))
    if args.limit is not None:
        docs = docs[: args.limit]
    if not docs:
        raise SystemExit("请指定 doc.json 或使用 --all")
    config = ChunkConfig(args.target_chars, args.max_chars, args.overlap_chars)
    failures = 0
    for index, path in enumerate(docs, 1):
        try:
            chunks = chunk_document(path, config=config)
            flagged = sum(bool(chunk.quality_flags) for chunk in chunks)
            print(f"[{index}/{len(docs)}] {path.parent.name}: {len(chunks)} chunks, {flagged} flagged")
        except Exception as exc:
            failures += 1
            print(f"[{index}/{len(docs)}] {path}: FAILED - {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
