import re
from collections.abc import Sequence
from typing import Any

from sqlalchemy import delete, func, literal, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.db_schemes.medical_rag import Asset, Chunk
from src.schemas.ingestion import SemanticChunk


class ChunkModel:
    @staticmethod
    def estimate_token_count(text: str) -> int:
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))

    @classmethod
    async def bulk_create(
        cls,
        session: AsyncSession,
        asset_id: int,
        chunks: Sequence[SemanticChunk],
        metadata: dict[str, Any] | None = None,
        commit: bool = True,
    ) -> list[Chunk]:
        if asset_id <= 0:
            raise ValueError("asset_id must be greater than zero")
        records = [
            Chunk(
                asset_id=asset_id,
                chunk_id=chunk.chunk_id,
                document_name=chunk.document_name,
                section_title=chunk.section_title,
                page_number=chunk.page_number,
                page_end=chunk.page_number,
                text=chunk.text,
                chunk_index=index,
                token_count=cls.estimate_token_count(chunk.text),
                chunk_metadata=dict(metadata or {}),
            )
            for index, chunk in enumerate(chunks, start=1)
        ]
        session.add_all(records)
        if commit:
            await session.commit()
            for record in records:
                await session.refresh(record)
        else:
            await session.flush()
        return records

    @classmethod
    async def replace_asset_chunks(
        cls,
        session: AsyncSession,
        asset_id: int,
        chunks: Sequence[SemanticChunk],
        metadata: dict[str, Any] | None = None,
        commit: bool = True,
    ) -> list[Chunk]:
        try:
            await session.execute(delete(Chunk).where(Chunk.asset_id == asset_id))
            records = await cls.bulk_create(
                session, asset_id, chunks, metadata, commit=False
            )
            if commit:
                await session.commit()
                for record in records:
                    await session.refresh(record)
            return records
        except Exception:
            await session.rollback()
            raise

    @staticmethod
    async def get_by_asset(
        session: AsyncSession,
        asset_id: int,
    ) -> Sequence[Chunk]:
        result = await session.execute(
            select(Chunk)
            .where(Chunk.asset_id == asset_id)
            .order_by(Chunk.chunk_index.asc())
        )
        return result.scalars().all()

    @staticmethod
    async def get_by_chunk_id(
        session: AsyncSession,
        asset_id: int,
        chunk_id: str,
    ) -> Chunk | None:
        result = await session.execute(
            select(Chunk).where(
                Chunk.asset_id == asset_id,
                Chunk.chunk_id == chunk_id,
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def count_by_asset(session: AsyncSession, asset_id: int) -> int:
        result = await session.execute(
            select(func.count(Chunk.id)).where(Chunk.asset_id == asset_id)
        )
        return int(result.scalar_one())

    @staticmethod
    async def delete_by_asset(
        session: AsyncSession,
        asset_id: int,
        commit: bool = True,
    ) -> int:
        result = await session.execute(
            delete(Chunk).where(Chunk.asset_id == asset_id)
        )
        if commit:
            await session.commit()
        return int(result.rowcount or 0)

    @staticmethod
    async def keyword_search(
        session: AsyncSession,
        query: str,
        limit: int = 20,
        project_id: int | None = None,
        asset_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """PostgreSQL full-text search over section titles and chunk text."""
        normalized_query = query.strip()
        if not normalized_query:
            return []

        searchable_text = func.concat_ws(
            " ",
            func.coalesce(Chunk.section_title, ""),
            func.coalesce(Chunk.text, ""),
        )
        document_vector = func.to_tsvector("english", searchable_text)
        query_vector = func.websearch_to_tsquery("english", normalized_query)
        keyword_score = func.ts_rank_cd(
            document_vector,
            query_vector,
            32,
        ).label("keyword_score")

        statement = (
            select(
                Chunk.asset_id,
                Asset.project_id,
                Chunk.chunk_id,
                Chunk.document_name,
                Chunk.section_title,
                Chunk.page_number,
                Chunk.text,
                keyword_score,
            )
            .join(Asset, Asset.id == Chunk.asset_id)
            .where(document_vector.op("@@")(query_vector))
        )
        if project_id is not None:
            statement = statement.where(Asset.project_id == project_id)
        if asset_id is not None:
            statement = statement.where(Chunk.asset_id == asset_id)

        statement = statement.order_by(
            keyword_score.desc(),
            Chunk.id.asc(),
        ).limit(max(1, min(limit, 100)))

        result = await session.execute(statement)
        return [dict(row._mapping) for row in result.all()]
