from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.helpers.config import settings
from src.models import close_database
from src.routes import (
    base_router,
    export_router,
    hybrid_retrieval_router,
    ingestion_router,
    rag_router,
    retrieval_pipeline_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close_database()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.include_router(base_router)
app.include_router(ingestion_router)
app.include_router(export_router)
app.include_router(hybrid_retrieval_router)
app.include_router(retrieval_pipeline_router)
app.include_router(rag_router)
