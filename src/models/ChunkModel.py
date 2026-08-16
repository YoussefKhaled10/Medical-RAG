import re
from collections.abc import Sequence
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.db_schemes.medical_rag import Chunk
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

        records: list[Chunk] = []
        for chunk_index, semantic_chunk in enumerate(chunks, start=1):
            record = Chunk(
                asset_id=asset_id,
                chunk_id=semantic_chunk.chunk_id,
                document_name=semantic_chunk.document_name,
                section_title=semantic_chunk.section_title,
                page_number=semantic_chunk.page_number,
                page_end=semantic_chunk.page_number,
                text=semantic_chunk.text,
                chunk_index=chunk_index,
                token_count=cls.estimate_token_count(semantic_chunk.text),
                chunk_metadata=dict(metadata or {}),
            )
            records.append(record)

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
            await session.execute(
                delete(Chunk).where(Chunk.asset_id == asset_id)
            )
            records = await cls.bulk_create(
                session=session,
                asset_id=asset_id,
                chunks=chunks,
                metadata=metadata,
                commit=False,
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
    async def count_by_asset(
        session: AsyncSession,
        asset_id: int,
    ) -> int:
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
