"""PostgreSQL/pgvector connection and readiness checks."""
from __future__ import annotations

from dataclasses import dataclass

import psycopg

from src.config import DatabaseConfig


@dataclass(frozen=True)
class DatabaseStatus:
    database: str
    user: str
    server_version: str
    vector_version: str


def connect(config: DatabaseConfig | None = None) -> psycopg.Connection:
    """Open a connection using project environment settings."""
    resolved = config or DatabaseConfig.from_env()
    return psycopg.connect(**resolved.connection_kwargs())


def check_pgvector(config: DatabaseConfig | None = None) -> DatabaseStatus:
    """Verify PostgreSQL connectivity and the required vector extension."""
    with connect(config) as connection:
        database, user, server_version = connection.execute(
            "SELECT current_database(), current_user, current_setting('server_version')"
        ).fetchone()
        extension = connection.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()

    if extension is None:
        raise RuntimeError("PostgreSQL is reachable but the vector extension is not enabled")
    return DatabaseStatus(
        database=database,
        user=user,
        server_version=server_version,
        vector_version=extension[0],
    )


def main() -> int:
    status = check_pgvector()
    print(
        f"Connected to {status.database} as {status.user}; "
        f"PostgreSQL {status.server_version}, pgvector {status.vector_version}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
