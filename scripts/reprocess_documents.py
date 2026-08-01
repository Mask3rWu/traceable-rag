"""Rebuild parsed documents after post-processing rules change.

By default this reads each document's stored ``structure.json`` and rebuilds
``doc.json`` and ``doc.md`` without rendering the PDF. ``--full`` runs the
complete rendering, detection, and post-processing pipeline instead.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ParseConfig
from src.data_processing.pipeline import parse_pdfs
from src.paths import (
    PARSED_ROOT,
    doc_id_from_path,
    list_gjb_pdfs,
    list_input_pdfs,
    list_paper_pdfs,
)


def _configure_utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild parsed outputs and overwrite doc.json/doc.md."
    )
    parser.add_argument(
        "pdf",
        nargs="*",
        type=Path,
        help="Specific PDF paths. Defaults to all source PDFs.",
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--papers-only",
        action="store_true",
        help="Process only source PDFs in the papers directory.",
    )
    scope.add_argument(
        "--gjb-only",
        action="store_true",
        help="Process only source PDFs in the GJB directory.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run rendering, PP-StructureV3 detection, and post-processing again.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many PDFs.",
    )
    return parser


def _resolve_pdfs(args: argparse.Namespace) -> list[Path]:
    if args.pdf:
        paths = args.pdf
    elif args.papers_only:
        paths = list_paper_pdfs()
    elif args.gjb_only:
        paths = list_gjb_pdfs()
    else:
        paths = list_input_pdfs()

    deduplicated: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduplicated.append(resolved)
    return deduplicated


def _missing_detection_paths(paths: Sequence[Path]) -> list[Path]:
    return [
        path
        for path in paths
        if not (PARSED_ROOT / doc_id_from_path(path) / "structure.json").is_file()
    ]


def main(argv: Sequence[str] | None = None) -> int:
    _configure_utf8_streams()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    pdfs = _resolve_pdfs(args)
    if args.limit is not None:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        parser.error("No PDFs matched the selected scope")

    if not args.full:
        missing = _missing_detection_paths(pdfs)
        if missing:
            examples = "\n".join(f"  - {path}" for path in missing[:5])
            more = "" if len(missing) <= 5 else f"\n  ... and {len(missing) - 5} more"
            parser.error(
                "Cannot reuse detection because structure.json is missing for:\n"
                f"{examples}{more}\nUse --full to regenerate it."
            )

    mode = "full pipeline" if args.full else "structure-only rebuild"
    print(f"Reprocessing {len(pdfs)} PDFs ({mode}); existing outputs will be overwritten.")
    summary = parse_pdfs(
        pdfs,
        config=ParseConfig(),
        skip_existing=False,
        reuse_detection=not args.full,
        skip_render=not args.full,
    )
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
