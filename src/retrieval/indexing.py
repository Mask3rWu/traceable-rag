"""Incrementally embed traceable chunks and store them in pgvector."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.types.json import Jsonb

from src.config import DatabaseConfig, EmbeddingConfig
from src.paths import PARSED_ROOT
from src.retrieval.database import connect
from src.retrieval.embedding import EmbeddingClient
from src.schema import Chunk

TABLE_NAME = "chunk_embeddings"


@dataclass(frozen=True)
class SourceChunk:
    chunk: Chunk
    content_hash: str


@dataclass(frozen=True)
class IndexResult:
    discovered: int
    embedded: int
    unchanged: int
    deleted: int


def discover_chunk_files(paths: Sequence[Path] | None = None) -> list[Path]:
    if paths:
        files = [path.resolve() for path in paths]
    else:
        files = sorted(PARSED_ROOT.glob("*/chunks.jsonl"))
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Chunk file not found: {missing[0]}")
    return files


def load_chunks(files: Iterable[Path]) -> list[SourceChunk]:
    chunks: list[SourceChunk] = []
    seen_ids: set[str] = set()
    for path in files:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    chunk = Chunk.model_validate_json(line)
                except Exception as exc:
                    raise ValueError(f"Invalid chunk at {path}:{line_number}: {exc}") from exc
                if chunk.chunk_id in seen_ids:
                    raise ValueError(f"Duplicate chunk_id across inputs: {chunk.chunk_id}")
                if not chunk.embedding_text.strip():
                    raise ValueError(f"Chunk has empty embedding_text: {chunk.chunk_id}")
                seen_ids.add(chunk.chunk_id)
                canonical = json.dumps(
                    chunk.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                chunks.append(SourceChunk(chunk=chunk, content_hash=digest))
    return chunks


def initialize_schema(connection: psycopg.Connection, dimension: int) -> None:
    """Create the dimension-specific table and cosine HNSW index."""
    table = sql.Identifier(TABLE_NAME)
    connection.execute(
        sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {} (
                chunk_id TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                document_id TEXT NOT NULL,
                source_file TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding_text TEXT NOT NULL,
                content_hash CHAR(64) NOT NULL,
                metadata JSONB NOT NULL,
                embedding VECTOR({}) NOT NULL,
                indexed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (chunk_id, embedding_model)
            )
            """
        ).format(table, sql.Literal(dimension))
    )
    actual_type = connection.execute(
        """
        SELECT format_type(attribute.atttypid, attribute.atttypmod)
        FROM pg_attribute AS attribute
        JOIN pg_class AS relation ON relation.oid = attribute.attrelid
        WHERE relation.relname = %s AND attribute.attname = 'embedding'
        """,
        (TABLE_NAME,),
    ).fetchone()
    expected_type = f"vector({dimension})"
    if actual_type is None or actual_type[0] != expected_type:
        found = actual_type[0] if actual_type else "missing"
        raise RuntimeError(
            f"Existing {TABLE_NAME}.embedding type is {found}; expected {expected_type}"
        )
    connection.execute(
        sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (document_id)").format(
            sql.Identifier(f"{TABLE_NAME}_document_id_idx"), table
        )
    )
    connection.execute(
        sql.SQL(
            "CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw "
            "(embedding vector_cosine_ops)"
        ).format(sql.Identifier(f"{TABLE_NAME}_embedding_hnsw_idx"), table)
    )
    connection.commit()


def _existing_hashes(
    connection: psycopg.Connection, model: str
) -> dict[str, str]:
    rows = connection.execute(
        sql.SQL("SELECT chunk_id, content_hash FROM {} WHERE embedding_model = %s").format(
            sql.Identifier(TABLE_NAME)
        ),
        (model,),
    ).fetchall()
    return {chunk_id: content_hash for chunk_id, content_hash in rows}


def _metadata(chunk: Chunk) -> dict:
    return chunk.model_dump(
        mode="json",
        exclude={"chunk_id", "document_id", "source_file", "text", "embedding_text"},
    )


def _upsert_batch(
    connection: psycopg.Connection,
    model: str,
    sources: Sequence[SourceChunk],
    embeddings: Sequence[list[float]],
) -> None:
    statement = sql.SQL(
        """
        INSERT INTO {} (
            chunk_id, embedding_model, document_id, source_file, text,
            embedding_text, content_hash, metadata, embedding, indexed_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (chunk_id, embedding_model) DO UPDATE SET
            document_id = EXCLUDED.document_id,
            source_file = EXCLUDED.source_file,
            text = EXCLUDED.text,
            embedding_text = EXCLUDED.embedding_text,
            content_hash = EXCLUDED.content_hash,
            metadata = EXCLUDED.metadata,
            embedding = EXCLUDED.embedding,
            indexed_at = CURRENT_TIMESTAMP
        """
    ).format(sql.Identifier(TABLE_NAME))
    rows = []
    for source, embedding in zip(sources, embeddings, strict=True):
        chunk = source.chunk
        rows.append(
            (
                chunk.chunk_id,
                model,
                chunk.document_id,
                chunk.source_file,
                chunk.text,
                chunk.embedding_text,
                source.content_hash,
                Jsonb(_metadata(chunk)),
                Vector(embedding),
            )
        )
    with connection.transaction():
        with connection.cursor() as cursor:
            cursor.executemany(statement, rows)


def _delete_stale(
    connection: psycopg.Connection,
    model: str,
    sources: Sequence[SourceChunk],
) -> int:
    current_by_document: dict[str, list[str]] = {}
    for source in sources:
        current_by_document.setdefault(source.chunk.document_id, []).append(
            source.chunk.chunk_id
        )

    deleted = 0
    statement = sql.SQL(
        "DELETE FROM {} WHERE embedding_model = %s AND document_id = %s "
        "AND NOT (chunk_id = ANY(%s))"
    ).format(sql.Identifier(TABLE_NAME))
    with connection.transaction():
        for document_id, chunk_ids in current_by_document.items():
            cursor = connection.execute(statement, (model, document_id, chunk_ids))
            deleted += cursor.rowcount
    return deleted


def build_index(
    files: Sequence[Path],
    *,
    embedding_config: EmbeddingConfig | None = None,
    database_config: DatabaseConfig | None = None,
) -> IndexResult:
    embedding_config = embedding_config or EmbeddingConfig.from_env()
    sources = load_chunks(files)
    client = EmbeddingClient(embedding_config)

    with connect(database_config) as connection:
        initialize_schema(connection, embedding_config.dimension)
        register_vector(connection)
        existing = _existing_hashes(connection, embedding_config.model)
        pending = [
            source
            for source in sources
            if existing.get(source.chunk.chunk_id) != source.content_hash
        ]

        completed = 0
        for start in range(0, len(pending), embedding_config.batch_size):
            batch = pending[start : start + embedding_config.batch_size]
            embeddings = client.embed([source.chunk.embedding_text for source in batch])
            _upsert_batch(connection, embedding_config.model, batch, embeddings)
            completed += len(batch)
            print(f"Embedded {completed}/{len(pending)} changed chunks", flush=True)

        deleted = _delete_stale(connection, embedding_config.model, sources)
        connection.execute(
            sql.SQL("ANALYZE {}").format(sql.Identifier(TABLE_NAME))
        )
        connection.commit()

    return IndexResult(
        discovered=len(sources),
        embedded=len(pending),
        unchanged=len(sources) - len(pending),
        deleted=deleted,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed chunks.jsonl records and incrementally store them in pgvector"
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Specific chunks.jsonl files")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Index every processed/parsed/*/chunks.jsonl file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.paths) == bool(args.all):
        raise SystemExit("Provide one or more chunks.jsonl paths, or use --all")
    files = discover_chunk_files(args.paths if args.paths else None)
    result = build_index(files)
    print(
        f"Index complete: {result.discovered} discovered, {result.embedded} embedded, "
        f"{result.unchanged} unchanged, {result.deleted} stale deleted"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
