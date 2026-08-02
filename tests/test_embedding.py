from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.config import EmbeddingConfig
from src.retrieval.embedding import EmbeddingClient


class _FakeEmbeddings:
    def __init__(self, data):
        self.data = data
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(data=self.data)


class EmbeddingClientTest(unittest.TestCase):
    def config(self, dimension: int = 3) -> EmbeddingConfig:
        return EmbeddingConfig(
            model="test-model",
            base_url="https://example.test/v1",
            api_key="test-secret",
            dimension=dimension,
            batch_size=2,
        )

    def test_preserves_input_order_using_response_indices(self):
        embeddings = _FakeEmbeddings(
            [
                SimpleNamespace(index=1, embedding=[4.0, 5.0, 6.0]),
                SimpleNamespace(index=0, embedding=[1.0, 2.0, 3.0]),
            ]
        )
        client = EmbeddingClient(
            self.config(), client=SimpleNamespace(embeddings=embeddings)
        )

        result = client.embed(["first", "second"])

        self.assertEqual(result, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertEqual(embeddings.calls[0]["model"], "test-model")
        self.assertEqual(embeddings.calls[0]["input"], ["first", "second"])

    def test_rejects_dimension_mismatch(self):
        embeddings = _FakeEmbeddings(
            [SimpleNamespace(index=0, embedding=[1.0, 2.0])]
        )
        client = EmbeddingClient(
            self.config(), client=SimpleNamespace(embeddings=embeddings)
        )

        with self.assertRaisesRegex(RuntimeError, "dimension mismatch"):
            client.embed(["text"])

    def test_empty_batch_does_not_call_endpoint(self):
        embeddings = _FakeEmbeddings([])
        client = EmbeddingClient(
            self.config(), client=SimpleNamespace(embeddings=embeddings)
        )

        self.assertEqual(client.embed([]), [])
        self.assertEqual(embeddings.calls, [])
