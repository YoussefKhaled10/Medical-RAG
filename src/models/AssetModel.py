from collections.abc import Sequence
from datetime import datetime, timezone

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.db_schemes.medical_rag import Asset


class AssetModel:
    VALID_STATUSES = {"PENDING", "PROCESSING", "COMPLETED", "FAILED"}

    @staticmethod
    async def create(
        session: AsyncSession,
        project_id: int,
        document_name: str,
        file_name: str,
        file_path: str,
        file_type: str | None = None,
        file_size: int | None = None,
        file_checksum: str | None = None,
        commit: bool = True,
    ) -> Asset:
        asset = Asset(
            project_id=project_id,
            document_name=document_name.strip(),
            file_name=file_name.strip(),
            file_path=file_path,
            file_type=file_type,
            file_size=file_size,
            file_checksum=file_checksum,
            processing_status="PENDING",
        )
        session.add(asset)

        if commit:
            await session.commit()
            await session.refresh(asset)
        else:
            await session.flush()

        return asset

    @staticmethod
    async def get_by_id(
        session: AsyncSession,
        asset_id: int,
    ) -> Asset | None:
        return await session.get(Asset, asset_id)

    @staticmethod
    async def get_by_checksum(
        session: AsyncSession,
        project_id: int,
        file_checksum: str,
    ) -> Asset | None:
        result = await session.execute(
            select(Asset).where(
                Asset.project_id == project_id,
                Asset.file_checksum == file_checksum,
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def list_by_project(
        session: AsyncSession,
        project_id: int,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[Asset]:
        result = await session.execute(
            select(Asset)
            .where(Asset.project_id == project_id)
            .order_by(Asset.id.asc())
            .offset(max(offset, 0))
            .limit(max(1, min(limit, 500)))
        )
        return result.scalars().all()

    @classmethod
    async def update_status(
        cls,
        session: AsyncSession,
        asset_id: int,
        status: str,
        error_message: str | None = None,
        commit: bool = True,
    ) -> Asset | None:
        normalized_status = status.strip().upper()
        if normalized_status not in cls.VALID_STATUSES:
            raise ValueError(f"Unsupported asset status: {status}")

        asset = await session.get(Asset, asset_id)
        if asset is None:
            return None

        asset.processing_status = normalized_status
        asset.error_message = error_message

        if normalized_status == "COMPLETED":
            asset.parsed_at = datetime.now(timezone.utc)
        elif normalized_status in {"PENDING", "PROCESSING"}:
            asset.parsed_at = None

        if commit:
            await session.commit()
            await session.refresh(asset)
        else:
            await session.flush()

        return asset

    @staticmethod
    async def mark_completed(
        session: AsyncSession,
        asset_id: int,
        total_pages: int,
        total_chunks: int,
        commit: bool = True,
    ) -> Asset | None:
        asset = await session.get(Asset, asset_id)
        if asset is None:
            return None

        asset.processing_status = "COMPLETED"
        asset.error_message = None
        asset.total_pages = total_pages
        asset.total_chunks = total_chunks
        asset.parsed_at = datetime.now(timezone.utc)

        if commit:
            await session.commit()
            await session.refresh(asset)
        else:
            await session.flush()

        return asset

    @staticmethod
    async def mark_failed(
        session: AsyncSession,
        asset_id: int,
        error_message: str,
        commit: bool = True,
    ) -> Asset | None:
        asset = await session.get(Asset, asset_id)
        if asset is None:
            return None

        asset.processing_status = "FAILED"
        asset.error_message = error_message[:5000]

        if commit:
            await session.commit()
            await session.refresh(asset)
        else:
            await session.flush()

        return asset

    @staticmethod
    async def delete(
        session: AsyncSession,
        asset_id: int,
        commit: bool = True,
    ) -> bool:
        result = await session.execute(
            delete(Asset).where(Asset.id == asset_id)
        )
        if commit:
            await session.commit()
        return bool(result.rowcount)
