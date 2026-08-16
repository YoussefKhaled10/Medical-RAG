import json
from collections.abc import Mapping, Sequence
from typing import Any

import asyncpg
from pgvector.asyncpg import register_vector

from ..VectorDBInterface import (
    VectorDBInterface,
    VectorDocument,
    VectorSearchResult,
)


class PGVector_Provider(VectorDBInterface):
    def __init__(
        self,
        database_url: str,
        embedding_dimension: int,
        table_name: str = "vector_documents",
        min_pool_size: int = 1,
        max_pool_size: int = 10,
    ) -> None:
        if not database_url:
            raise ValueError("database_url must not be empty")
        if embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be greater than zero")
        if not table_name.replace("_", "").isalnum():
            raise ValueError("table_name contains invalid characters")
        if min_pool_size < 1 or max_pool_size < min_pool_size:
            raise ValueError("Invalid connection pool size")

        self._database_url = database_url.replace(
            "postgresql+asyncpg://", "postgresql://", 1
        )
        self._embedding_dimension = embedding_dimension
        self._table_name = table_name
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._pool: asyncpg.Pool | None = None

    async def _initialize_connection(self, connection: asyncpg.Connection) -> None:
        await register_vector(connection)

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PGVector_Provider is not initialized")
        return self._pool

    def _validate_embedding(self, embedding: Sequence[float]) -> list[float]:
        vector = [float(value) for value in embedding]
        if len(vector) != self._embedding_dimension:
            raise ValueError(
                f"Expected {self._embedding_dimension} dimensions, got {len(vector)}"
            )
        return vector

    async def initialize(self) -> None:
        if self._pool is not None:
            return

        connection = await asyncpg.connect(self._database_url)
        try:
            await connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        finally:
            await connection.close()

        self._pool = await asyncpg.create_pool(
            dsn=self._database_url,
            min_size=self._min_pool_size,
            max_size=self._max_pool_size,
            init=self._initialize_connection,
        )

        pool = await self._get_pool()
        async with pool.acquire() as connection:
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._table_name} (
                    id TEXT PRIMARY KEY,
                    asset_id INTEGER NULL REFERENCES assets(id) ON DELETE CASCADE,
                    text TEXT NOT NULL,
                    embedding vector({self._embedding_dimension}) NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await connection.execute(
                f"CREATE INDEX IF NOT EXISTS ix_{self._table_name}_asset_id "
                f"ON {self._table_name} (asset_id)"
            )
            await connection.execute(
                f"CREATE INDEX IF NOT EXISTS ix_{self._table_name}_embedding_hnsw "
                f"ON {self._table_name} USING hnsw "
                f"(embedding vector_cosine_ops)"
            )
            await connection.execute(
                f"CREATE INDEX IF NOT EXISTS ix_{self._table_name}_metadata_gin "
                f"ON {self._table_name} USING gin (metadata)"
            )

    async def upsert(self, documents: Sequence[VectorDocument]) -> None:
        if not documents:
            return

        rows = []
        for document in documents:
            metadata = dict(document.metadata)
            asset_id_value = metadata.get("asset_id")
            asset_id = int(asset_id_value) if asset_id_value is not None else None
            rows.append(
                (
                    document.id,
                    asset_id,
                    document.text,
                    self._validate_embedding(document.embedding),
                    json.dumps(metadata),
                )
            )

        pool = await self._get_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                await connection.executemany(
                    f"""
                    INSERT INTO {self._table_name}
                        (id, asset_id, text, embedding, metadata)
                    VALUES ($1, $2, $3, $4, $5::jsonb)
                    ON CONFLICT (id) DO UPDATE SET
                        asset_id = EXCLUDED.asset_id,
                        text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                    """,
                    rows,
                )

    async def similarity_search(
        self,
        query_embedding: Sequence[float],
        limit: int = 5,
        filters: Mapping[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        vector = self._validate_embedding(query_embedding)
        metadata_filter = json.dumps(dict(filters or {}))
        pool = await self._get_pool()

        async with pool.acquire() as connection:
            records = await connection.fetch(
                f"""
                SELECT id, text, metadata,
                       1 - (embedding <=> $1) AS score
                FROM {self._table_name}
                WHERE metadata @> $2::jsonb
                ORDER BY embedding <=> $1
                LIMIT $3
                """,
                vector,
                metadata_filter,
                limit,
            )

        return [
            VectorSearchResult(
                id=record["id"],
                text=record["text"],
                score=float(record["score"]),
                metadata=(
                    json.loads(record["metadata"])
                    if isinstance(record["metadata"], str)
                    else dict(record["metadata"])
                ),
            )
            for record in records
        ]

    async def delete_by_ids(self, document_ids: Sequence[str]) -> int:
        if not document_ids:
            return 0
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            status = await connection.execute(
                f"DELETE FROM {self._table_name} WHERE id = ANY($1::text[])",
                list(document_ids),
            )
        return self._affected_rows(status)

    async def delete_by_asset_id(self, asset_id: int) -> int:
        pool = await self._get_pool()
        async with pool.acquire() as connection:
            status = await connection.execute(
                f"DELETE FROM {self._table_name} WHERE asset_id = $1", asset_id
            )
        return self._affected_rows(status)

    async def health_check(self) -> bool:
        try:
            pool = await self._get_pool()
            async with pool.acquire() as connection:
                return await connection.fetchval("SELECT 1") == 1
        except (asyncpg.PostgresError, OSError, RuntimeError):
            return False

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @staticmethod
    def _affected_rows(command_status: str) -> int:
        try:
            return int(command_status.rsplit(" ", maxsplit=1)[-1])
        except (ValueError, IndexError):
            return 0
