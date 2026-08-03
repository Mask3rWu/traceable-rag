from __future__ import annotations

import unittest

from src.config import EmbeddingConfig
from src.retrieval.contracts import SearchResult
from src.retrieval.dense import DenseRetriever


class _FakeEmbeddingClient:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[float(len(text)), 0.0] for text in texts]


class _FakeVectorStore:
    def __init__(self) -> None:
        self.calls = []

    def search_many(self, vectors, *, embedding_model, limit):
        self.calls.append((vectors, embedding_model, limit))
        return [
            [
                SearchResult(
                    chunk_id=f"chunk-{index}",
                    content_hash=f"hash-{index}",
                    dense_rank=1,
                    dense_score=0.9,
                    final_rank=1,
                )
            ]
            for index, _ in enumerate(vectors)
        ]


class DenseRetrieverTest(unittest.TestCase):
    def test_batches_embeddings_and_delegates_vector_search(self):
        config = EmbeddingConfig(
            model="test-model",
            base_url="https://example.test/v1",
            api_key="secret",
            dimension=2,
            batch_size=2,
        )
        client = _FakeEmbeddingClient()
        store = _FakeVectorStore()
        retriever = DenseRetriever(config, client=client, store=store)

        results = retriever.search_many(["a", "bb", "ccc"], limit=7)

        self.assertEqual(client.calls, [["a", "bb"], ["ccc"]])
        self.assertEqual(store.calls[0][1:], ("test-model", 7))
        self.assertEqual([items[0].chunk_id for items in results], ["chunk-0", "chunk-1", "chunk-2"])

    def test_rejects_blank_query(self):
        config = EmbeddingConfig("model", "https://example.test", "secret", 2, 2)
        retriever = DenseRetriever(
            config, client=_FakeEmbeddingClient(), store=_FakeVectorStore()
        )
        with self.assertRaisesRegex(ValueError, "must not be blank"):
            retriever.search_many(["ok", " "])
