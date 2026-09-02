from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    create_async_engine,
)

from src.helpers.config import settings


engine: AsyncEngine = create_async_engine(
    settings.POSTGRES_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    echo=False,
)

try:
    from sqlalchemy.ext.asyncio import async_sessionmaker
    AsyncSessionLocal = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        autoflush=False,
        expire_on_commit=False,
    )
except ImportError:
    from sqlalchemy.orm import sessionmaker
    AsyncSessionLocal = sessionmaker(
        bind=engine,
        class_=AsyncSession,
        autoflush=False,
        expire_on_commit=False,
    )


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db_tables() -> None:
    from src.models.db_schemes.medical_rag import SQLAlchemyBase
    async with engine.begin() as conn:
        await conn.run_sync(SQLAlchemyBase.metadata.create_all)


async def close_database() -> None:
    await engine.dispose()
