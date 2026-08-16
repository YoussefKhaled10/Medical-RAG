from collections.abc import Sequence

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.db_schemes.medical_rag import Project


class ProjectModel:
    @staticmethod
    async def create(
        session: AsyncSession,
        name: str,
        description: str | None = None,
        source_name: str | None = None,
        source_url: str | None = None,
        source_version: str | None = None,
        license_name: str | None = None,
        status: str = "ACTIVE",
        commit: bool = True,
    ) -> Project:
        project = Project(
            name=name.strip(),
            description=description,
            source_name=source_name,
            source_url=source_url,
            source_version=source_version,
            license_name=license_name,
            status=status,
        )
        session.add(project)

        if commit:
            await session.commit()
            await session.refresh(project)
        else:
            await session.flush()

        return project

    @staticmethod
    async def get_by_id(
        session: AsyncSession,
        project_id: int,
    ) -> Project | None:
        return await session.get(Project, project_id)

    @staticmethod
    async def get_by_name(
        session: AsyncSession,
        name: str,
    ) -> Project | None:
        result = await session.execute(
            select(Project).where(Project.name == name.strip())
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def list_all(
        session: AsyncSession,
        offset: int = 0,
        limit: int = 100,
    ) -> Sequence[Project]:
        result = await session.execute(
            select(Project)
            .order_by(Project.id.asc())
            .offset(max(offset, 0))
            .limit(max(1, min(limit, 500)))
        )
        return result.scalars().all()

    @staticmethod
    async def update(
        session: AsyncSession,
        project_id: int,
        commit: bool = True,
        **changes: object,
    ) -> Project | None:
        project = await session.get(Project, project_id)
        if project is None:
            return None

        allowed_fields = {
            "name",
            "description",
            "source_name",
            "source_url",
            "source_version",
            "license_name",
            "status",
        }
        for field, value in changes.items():
            if field not in allowed_fields:
                raise ValueError(f"Project field cannot be updated: {field}")
            setattr(project, field, value)

        if commit:
            await session.commit()
            await session.refresh(project)
        else:
            await session.flush()

        return project

    @staticmethod
    async def delete(
        session: AsyncSession,
        project_id: int,
        commit: bool = True,
    ) -> bool:
        result = await session.execute(
            delete(Project).where(Project.id == project_id)
        )
        if commit:
            await session.commit()
        return bool(result.rowcount)
