"""Run a small vision model over parsed figure/table crops.

The output is deliberately kept outside ``doc.json``.  Parser output remains the
source of truth, while ``visual_enrichment.json`` is a replaceable retrieval aid.
The client speaks the OpenAI-compatible API exposed by LM Studio.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


OUTPUT_NAME = "visual_enrichment.json"
SYSTEM_PROMPT = """你是技术文档图片描述器。
请根据图片及给定上下文，只提取可确认的视觉内容。不要猜测图片中不可见或无法辨认的内容，
不要根据常识补充结论。只输出图片描述正文，不要输出 JSON、Markdown、标题或额外说明。
用 1-3 句话描述图片中明确可见的对象、文字、箭头、坐标轴、图例或表格结构，最多120字。
优先说明关键对象及其可见关系；图注只作辅助，不要仅复述图注。
"""


@dataclass(frozen=True)
class VisionConfig:
    base_url: str = "http://localhost:1234/v1"
    model: str = ""
    timeout: float = 120.0
    max_tokens: int = 300


class VisionClient:
    def __init__(self, config: VisionConfig):
        if not config.model:
            raise ValueError("LM Studio model is required (use --model)")
        self.config = config

    def describe(self, image_path: Path, context: str) -> str:
        mime = _mime_type(image_path)
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload = {
            "model": self.config.model,
            "temperature": 0,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": context + "\n\n请输出图片描述正文。"},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                    ],
                },
            ],
        }
        request = urllib.request.Request(
            self.config.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LM Studio HTTP {exc.code}: {body[:500]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"cannot reach LM Studio: {exc.reason}") from exc

        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("LM Studio response has no choices[0].message.content") from exc
        return parse_model_output(content)


def parse_model_output(content: str) -> str:
    """Normalize a model response into one description string.

    Old JSON-shaped responses remain readable so interrupted batch jobs can be
    resumed after the output format change.
    """
    text = str(content).strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    if text.startswith("{"):
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict) and isinstance(value.get("description"), str):
            text = value["description"].strip()
    if not text:
        raise ValueError("model returned an empty description")
    return _trim_description(text)


def _trim_description(description: str, limit: int = 150) -> str:
    """Enforce the storage limit without leaving a partial sentence when possible."""
    text = description.strip()
    if len(text) <= limit:
        return text
    prefix = text[:limit]
    boundary = max(prefix.rfind(mark) for mark in "。！？.!?")
    if boundary >= 0:
        return prefix[:boundary + 1].strip()
    return prefix.rstrip("，、；：,;: ")


def _mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(suffix, "image/png")


def _all_blocks(doc: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for page in doc.get("pages", []):
        blocks.extend(page.get("blocks", []))
    return sorted(blocks, key=lambda b: (int(b.get("page", 0)), b.get("order") is None, b.get("order") or 0, b.get("block_id", "")))


def _resolve_asset(doc_json: Path, value: str | None) -> Path | None:
    if not value:
        return None
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [doc_json.parent / raw, doc_json.parent.parents[2] / raw]
    return next((p.resolve() for p in candidates if p.is_file()), None)


def _context_for(block: dict[str, Any], blocks: list[dict[str, Any]], by_id: dict[str, dict[str, Any]]) -> str:
    captions = [by_id[cid].get("text", "") for cid in block.get("caption_ids", []) if cid in by_id]
    section = " / ".join(block.get("section_path") or []) or "未标注章节"
    refs = [b.get("text", "") for b in blocks if block.get("block_id") in (b.get("references") or [])]
    same_section = [b for b in blocks if b.get("section_path") == block.get("section_path")]
    index = next((i for i, item in enumerate(same_section) if item.get("block_id") == block.get("block_id")), len(same_section))
    nearby = [
        b.get("text", "")
        for b in same_section[max(0, index - 2): index + 3]
        if b.get("block_type") in {"paragraph", "list", "appendix"} and b.get("text")
    ]
    parts = [f"所属章节：{section}"]
    if captions:
        parts.append("图注：" + "；".join(captions))
    if refs:
        parts.append("引用正文：" + "；".join(refs))
    if nearby:
        parts.append("相邻正文：" + "；".join(nearby))
    if block.get("text"):
        parts.append("已有块文本：" + str(block["text"]))
    return "\n".join(parts)[:3000]


def enrich_document(doc_json: Path, client: VisionClient, *, force: bool = False) -> dict[str, Any]:
    doc = json.loads(doc_json.read_text(encoding="utf-8"))
    blocks = _all_blocks(doc)
    by_id = {b.get("block_id"): b for b in blocks}
    output_path = doc_json.parent / OUTPUT_NAME
    existing: dict[str, dict[str, Any]] = {}
    if output_path.is_file() and not force:
        old = json.loads(output_path.read_text(encoding="utf-8"))
        existing = {item["block_id"]: item for item in old.get("items", []) if item.get("block_id")}

    items: list[dict[str, Any]] = []
    for block in blocks:
        if block.get("block_type") not in {"figure", "table"}:
            continue
        block_id = block.get("block_id")
        if block_id in existing and existing[block_id].get("status") == "ok" and not force:
            items.append(existing[block_id])
            continue
        image_path = _resolve_asset(doc_json, block.get("image_crop"))
        item: dict[str, Any] = {
            "document_id": doc.get("document_id"),
            "block_id": block_id,
            "block_type": block.get("block_type"),
            "page": block.get("page"),
            "image_path": block.get("image_crop"),
            "status": "error",
        }
        if image_path is None:
            item["error"] = "image_crop does not exist"
        else:
            try:
                item["description"] = client.describe(image_path, _context_for(block, blocks, by_id))
                item["status"] = "ok"
            except Exception as exc:  # keep one bad image from stopping a document
                item["error"] = str(exc)
        items.append(item)

    output = {
        "schema_version": 1,
        "document_id": doc.get("document_id"),
        "source_doc": str(doc_json).replace("\\", "/"),
        "items": items,
    }
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _doc_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(value) for value in args.doc]
    if args.all:
        root = Path(args.parsed_root)
        paths.extend(sorted(root.glob("*/doc.json")))
    seen: set[Path] = set()
    result = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(path)
    return result[: args.limit] if args.limit else result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用 LM Studio 为解析后的图表生成视觉描述")
    parser.add_argument("doc", nargs="*", help="doc.json 路径")
    parser.add_argument("--all", action="store_true", help="处理 parsed_root 下所有文档")
    parser.add_argument("--parsed-root", default="processed/parsed")
    parser.add_argument("--base-url", default="http://localhost:1234/v1")
    parser.add_argument("--model", required=True, help="LM Studio 中加载的视觉模型标识")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true", help="重新处理已有成功结果")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    docs = _doc_paths(args)
    if not docs:
        parser.error("请指定 doc.json 或使用 --all")
    client = VisionClient(VisionConfig(args.base_url, args.model, args.timeout, args.max_tokens))
    failed = 0
    for index, doc_path in enumerate(docs, 1):
        try:
            result = enrich_document(doc_path, client, force=args.force)
            ok = sum(item.get("status") == "ok" for item in result["items"])
            errors = len(result["items"]) - ok
            print(f"[{index}/{len(docs)}] {result['document_id']}: {ok} ok, {errors} error -> {doc_path.parent / OUTPUT_NAME}")
        except Exception as exc:
            failed += 1
            print(f"[{index}/{len(docs)}] {doc_path}: FAILED - {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
