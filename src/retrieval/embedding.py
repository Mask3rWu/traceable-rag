"""OpenAI-compatible embedding client with strict dimension validation."""
from __future__ import annotations

from collections.abc import Sequence

from openai import OpenAI

from src.config import EmbeddingConfig


class EmbeddingClient:
    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        *,
        client: OpenAI | None = None,
    ) -> None:
        self.config = config or EmbeddingConfig.from_env()
        self._client = client or OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=60.0,
            max_retries=3,
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed one non-empty batch and preserve its input order."""
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("Embedding input must not contain empty text")

        response = self._client.embeddings.create(
            model=self.config.model,
            input=list(texts),
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        if len(ordered) != len(texts):
            raise RuntimeError(
                f"Embedding endpoint returned {len(ordered)} items for {len(texts)} inputs"
            )

        embeddings = [item.embedding for item in ordered]
        invalid_dimensions = {
            len(vector) for vector in embeddings if len(vector) != self.config.dimension
        }
        if invalid_dimensions:
            returned = ", ".join(str(value) for value in sorted(invalid_dimensions))
            raise RuntimeError(
                f"Embedding dimension mismatch: configured {self.config.dimension}, "
                f"returned {returned}"
            )
        return embeddings
