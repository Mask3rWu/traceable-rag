"""Batch-run MLLM visual parsing for parsed chart/table documents.

The parser writes one ``visual_enrichment.json`` next to each ``doc.json``.
Successful blocks are skipped on later runs by :func:`enrich_document`, so the
script can be interrupted and resumed without re-paying for completed calls.

Examples (from the project root)::

    conda run -n dba-py311 python scripts/batch_mllm_charts.py --model qwen2.5-vl
    conda run -n dba-py311 python scripts/batch_mllm_charts.py \
        --doc processed/parsed/2-电子系统/doc.json --model qwen2.5-vl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.visual_enrichment import (  # noqa: E402
    VisionClient,
    VisionConfig,
    enrich_document,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch parse figure/table crops with an OpenAI-compatible MLLM."
    )
    parser.add_argument(
        "--parsed-root",
        type=Path,
        default=PROJECT_ROOT / "processed" / "parsed",
        help="Root containing one directory per parsed document (default: processed/parsed).",
    )
    parser.add_argument(
        "--doc",
        action="append",
        type=Path,
        default=[],
        help="A doc.json to process; repeat for multiple documents. If omitted, scan --parsed-root.",
    )
    parser.add_argument(
        "--model",
        default="qwen/qwen3.5-9b",
        help="MLLM model name exposed by the server (default: qwen/qwen3.5-9b).",
    )
    parser.add_argument("--base-url", default="http://localhost:1234/v1")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--limit", type=int, help="Process at most this many documents.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess blocks whose previous status is ok.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Retry a document after an unexpected document-level failure (default: 0).",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=2.0,
        help="Seconds between document retries (default: 2).",
    )
    parser.add_argument(
        "--failure-log",
        type=Path,
        help="Optional JSON file for document-level failures.",
    )
    return parser


def _configure_utf8_streams() -> None:
    """Keep document names readable in the Windows console."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass


def discover_docs(args: argparse.Namespace) -> list[Path]:
    """Resolve, de-duplicate, and validate the requested doc.json paths."""
    parsed_root = args.parsed_root
    if not parsed_root.is_absolute():
        parsed_root = PROJECT_ROOT / parsed_root
    candidates = args.doc or sorted(parsed_root.glob("*/doc.json"))
    result: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        path = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            result.append(path)
        else:
            print(f"Skip missing doc.json: {path}", file=sys.stderr)
    if args.limit is not None:
        result = result[: args.limit]
    return result


def _run_document(
    doc_path: Path,
    client: VisionClient,
    *,
    force: bool,
    retries: int,
    retry_delay: float,
) -> dict:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return enrich_document(doc_path, client, force=force)
        except Exception as exc:  # document-level errors are retried by the caller
            last_error = exc
            if attempt < retries:
                time.sleep(retry_delay)
    assert last_error is not None
    raise last_error


def main(argv: Iterable[str] | None = None) -> int:
    _configure_utf8_streams()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.retries < 0:
        parser.error("--retries must be non-negative")
    if args.retry_delay < 0:
        parser.error("--retry-delay must be non-negative")

    docs = discover_docs(args)
    if not docs:
        parser.error("No doc.json files found; pass --doc or check --parsed-root")

    client = VisionClient(
        VisionConfig(
            base_url=args.base_url,
            model=args.model,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
        )
    )
    failures: list[dict[str, str]] = []
    total_ok = 0
    total_errors = 0
    print(f"Processing {len(docs)} document(s) with model {args.model!r}")
    for index, doc_path in enumerate(docs, start=1):
        try:
            output = _run_document(
                doc_path,
                client,
                force=args.force,
                retries=args.retries,
                retry_delay=args.retry_delay,
            )
            ok = sum(item.get("status") == "ok" for item in output.get("items", []))
            errors = len(output.get("items", [])) - ok
            total_ok += ok
            total_errors += errors
            print(
                f"[{index}/{len(docs)}] {output.get('document_id', doc_path.parent.name)}: "
                f"{ok} ok, {errors} error -> {doc_path.parent / 'visual_enrichment.json'}"
            )
        except Exception as exc:
            failures.append({"doc": str(doc_path), "error": str(exc)})
            print(f"[{index}/{len(docs)}] FAILED {doc_path}: {exc}", file=sys.stderr)

    if args.failure_log:
        failure_log = args.failure_log
        if not failure_log.is_absolute():
            failure_log = PROJECT_ROOT / failure_log
        failure_log.parent.mkdir(parents=True, exist_ok=True)
        failure_log.write_text(
            json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(
        f"Finished: {len(docs) - len(failures)} documents, "
        f"{total_ok} successful blocks, {total_errors} block errors, "
        f"{len(failures)} document failures."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
