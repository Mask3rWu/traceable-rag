from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.retrieval.indexing import load_chunks
from src.schema import Chunk


def chunk(*, page_end: int = 1) -> Chunk:
    return Chunk(
        chunk_id="doc_C00001",
        document_id="doc",
        text="Body",
        embedding_text="Document: doc\nContent: Body",
        block_ids=["block-1"],
        page_start=1,
        page_end=page_end,
        source_file="doc.pdf",
    )


class IndexingTest(unittest.TestCase):
    def write_chunks(self, directory: str, chunks: list[Chunk], name: str) -> Path:
        path = Path(directory) / name
        path.write_text(
            "".join(item.model_dump_json() + "\n" for item in chunks),
            encoding="utf-8",
        )
        return path

    def test_hash_changes_when_traceability_metadata_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = self.write_chunks(tmp, [chunk(page_end=1)], "first.jsonl")
            second = self.write_chunks(tmp, [chunk(page_end=2)], "second.jsonl")

            first_hash = load_chunks([first])[0].content_hash
            second_hash = load_chunks([second])[0].content_hash

        self.assertNotEqual(first_hash, second_hash)

    def test_rejects_duplicate_chunk_ids_across_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = self.write_chunks(tmp, [chunk()], "first.jsonl")
            second = self.write_chunks(tmp, [chunk()], "second.jsonl")

            with self.assertRaisesRegex(ValueError, "Duplicate chunk_id"):
                load_chunks([first, second])
