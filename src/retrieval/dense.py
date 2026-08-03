"""Dense retrieval over the existing pgvector chunk index."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg import sql

from src.config import DatabaseConfig, EmbeddingConfig
from src.retrieval.contracts import SearchResult
from src.retrieval.database import connect
from src.retrieval.embedding import EmbeddingClient
from src.retrieval.indexing import TABLE_NAME


class VectorSearchStore(Protocol):
    def search_many(
        self,
        vectors: Sequence[Sequence[float]],
        *,
        embedding_model: str,
        limit: int,
    ) -> list[list[SearchResult]]: ...


class PgVectorSearchStore:
    """Execute cosine searches while keeping database details out of the retriever."""

    def __init__(self, database_config: DatabaseConfig | None = None) -> None:
        self.database_config = database_config

    def search_many(
        self,
        vectors: Sequence[Sequence[float]],
        *,
        embedding_model: str,
        limit: int,
    ) -> list[list[SearchResult]]:
        if limit <= 0:
            raise ValueError("Dense retrieval limit must be greater than zero")
        statement = sql.SQL(
            """
            SELECT chunk_id, content_hash, 1 - (embedding <=> %s) AS score
            FROM {}
            WHERE embedding_model = %s
            ORDER BY embedding <=> %s
            LIMIT %s
            """
        ).format(sql.Identifier(TABLE_NAME))
        batches: list[list[SearchResult]] = []
        with connect(self.database_config) as connection:
            register_vector(connection)
            for vector in vectors:
                value = Vector(vector)
                rows = connection.execute(
                    statement, (value, embedding_model, value, limit)
                ).fetchall()
                batches.append(
                    [
                        SearchResult(
                            chunk_id=chunk_id,
                            content_hash=content_hash,
                            dense_rank=rank,
                            dense_score=float(score),
                            final_rank=rank,
                        )
                        for rank, (chunk_id, content_hash, score) in enumerate(
                            rows, start=1
                        )
                    ]
                )
        return batches


class DenseRetriever:
    def __init__(
        self,
        embedding_config: EmbeddingConfig | None = None,
        *,
        client: EmbeddingClient | None = None,
        store: VectorSearchStore | None = None,
    ) -> None:
        self.config = embedding_config or EmbeddingConfig.from_env()
        self.client = client or EmbeddingClient(self.config)
        self.store = store or PgVectorSearchStore()

    def search(self, query: str, *, limit: int = 50) -> list[SearchResult]:
        return self.search_many([query], limit=limit)[0]

    def search_many(
        self, queries: Sequence[str], *, limit: int = 50
    ) -> list[list[SearchResult]]:
        if any(not query.strip() for query in queries):
            raise ValueError("Dense retrieval queries must not be blank")
        if not queries:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(queries), self.config.batch_size):
            batch = queries[start : start + self.config.batch_size]
            vectors.extend(self.client.embed(batch))
        return self.store.search_many(
            vectors, embedding_model=self.config.model, limit=limit
        )
