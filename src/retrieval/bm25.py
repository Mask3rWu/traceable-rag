"""Persistent Chinese BM25 index over traceable chunks."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import unicodedata
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import bm25s
import jieba

from src.paths import PROCESSED_ROOT
from src.retrieval.contracts import SearchResult
from src.retrieval.indexing import SourceChunk, discover_chunk_files, load_chunks


DEFAULT_INDEX_ROOT = PROCESSED_ROOT / "retrieval" / "bm25"
MANIFEST_NAME = "manifest.json"
INDEX_SCHEMA_VERSION = 1
_HTML_TAG_RE = re.compile(r"<[^>]+>")

jieba.setLogLevel(logging.WARNING)


def tokenize(text: str) -> list[str]:
    """Normalize text and apply Jieba search-mode tokenization."""
    normalized = unicodedata.normalize("NFKC", _HTML_TAG_RE.sub(" ", text)).lower()
    return [
        token.strip()
        for token in jieba.lcut_for_search(normalized)
        if token.strip() and any(character.isalnum() for character in token)
    ]


def lexical_text(source: SourceChunk) -> str:
    chunk = source.chunk
    fields = [chunk.source_file, chunk.document_id, *chunk.heading_path, chunk.text]
    return "\n".join(field for field in fields if field)


def _corpus_hash(sources: Sequence[SourceChunk]) -> str:
    value = "\n".join(
        f"{source.chunk.chunk_id}:{source.content_hash}" for source in sources
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_bm25_index(
    sources: Sequence[SourceChunk], output_dir: Path = DEFAULT_INDEX_ROOT
) -> dict:
    if not sources:
        raise ValueError("Cannot build a BM25 index without chunks")
    output_dir.mkdir(parents=True, exist_ok=True)
    retriever = bm25s.BM25(method="lucene")
    retriever.index(
        [tokenize(lexical_text(source)) for source in sources], show_progress=True
    )
    corpus = [
        {
            "chunk_id": source.chunk.chunk_id,
            "content_hash": source.content_hash,
        }
        for source in sources
    ]
    retriever.save(output_dir, corpus=corpus, show_progress=False)
    manifest = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "lucene",
        "tokenizer": "jieba-search",
        "chunk_count": len(sources),
        "corpus_hash": _corpus_hash(sources),
    }
    (output_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


class BM25Retriever:
    def __init__(self, index_dir: Path = DEFAULT_INDEX_ROOT) -> None:
        manifest_path = index_dir / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"BM25 index not found at {index_dir}; run scripts/build_bm25.py first"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != INDEX_SCHEMA_VERSION:
            raise RuntimeError("Unsupported BM25 index schema version")
        self._retriever = bm25s.BM25.load(
            index_dir, load_corpus=True, mmap=True, show_progress=False
        )
        self._chunk_count = int(self.manifest["chunk_count"])

    def search(self, query: str, *, limit: int = 50) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("BM25 retrieval query must not be blank")
        if limit <= 0:
            raise ValueError("BM25 retrieval limit must be greater than zero")
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        result = self._retriever.retrieve(
            [query_tokens],
            k=min(limit, self._chunk_count),
            return_as="tuple",
            show_progress=False,
        )
        hits: list[SearchResult] = []
        for document, score in zip(
            result.documents[0], result.scores[0], strict=True
        ):
            numeric_score = float(score)
            if numeric_score <= 0:
                break
            rank = len(hits) + 1
            hits.append(
                SearchResult(
                    chunk_id=document["chunk_id"],
                    content_hash=document["content_hash"],
                    bm25_rank=rank,
                    bm25_score=numeric_score,
                    final_rank=rank,
                )
            )
        return hits

    def search_many(
        self, queries: Sequence[str], *, limit: int = 50
    ) -> list[list[SearchResult]]:
        return [self.search(query, limit=limit) for query in queries]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the persistent Chinese BM25 index")
    parser.add_argument("paths", nargs="*", type=Path, help="Specific chunks.jsonl files")
    parser.add_argument("--all", action="store_true", help="Index all parsed chunks")
    parser.add_argument("--output", type=Path, default=DEFAULT_INDEX_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.paths) == bool(args.all):
        raise SystemExit("Provide one or more chunks.jsonl paths, or use --all")
    files = discover_chunk_files(args.paths if args.paths else None)
    manifest = build_bm25_index(load_chunks(files), args.output)
    print(
        f"BM25 index complete: {manifest['chunk_count']} chunks at {args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
