from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.retrieval.bm25 import BM25Retriever, build_bm25_index, tokenize
from src.retrieval.indexing import SourceChunk
from src.schema import Chunk


def source(chunk_id: str, text: str, content_hash: str) -> SourceChunk:
    return SourceChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id="doc",
            source_file="doc.pdf",
            text=text,
            embedding_text=text,
            block_ids=[f"block-{chunk_id}"],
            page_start=1,
            page_end=1,
            heading_path=["1 测试章节"],
        ),
        content_hash=content_hash,
    )


class BM25RetrieverTest(unittest.TestCase):
    def test_chinese_index_round_trip(self):
        sources = [
            source("tank", "坦克履带断裂后无法继续行驶", "hash-tank"),
            source("aircraft", "军用飞机结构强度要求", "hash-aircraft"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            index_dir = Path(tmp) / "bm25"
            manifest = build_bm25_index(sources, index_dir)
            results = BM25Retriever(index_dir).search("履带断裂", limit=2)

        self.assertEqual(manifest["chunk_count"], 2)
        self.assertEqual(results[0].chunk_id, "tank")
        self.assertEqual(results[0].content_hash, "hash-tank")
        self.assertGreater(results[0].bm25_score, 0)

    def test_tokenizer_normalizes_identifiers_and_drops_markup(self):
        tokens = tokenize("<td>GJB 67.11A 坦克履带</td>")
        self.assertIn("gjb", tokens)
        self.assertIn("坦克", tokens)
        self.assertNotIn("td", tokens)
