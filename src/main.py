from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    """Release database resources when the application shuts down."""
    yield
    await close_database()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Evidence-grounded API for alcohol recovery information, "
        "retrieval, ingestion, and document analysis."
    ),
    lifespan=lifespan,
)


# Comma-separated production origins can be supplied through CORS_ORIGINS.
# Example:
# CORS_ORIGINS=https://recoverypath-ai.streamlit.app,http://localhost:8501
configured_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]

local_origins = [
    "http://localhost:8501",
    "http://localhost:8502",
    "http://127.0.0.1:8501",
    "http://127.0.0.1:8502",
]

allowed_origins = list(
    dict.fromkeys([*configured_origins, *local_origins])
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    # Starlette does not interpret "https://*.streamlit.app" as a wildcard
    # in allow_origins. A regular expression is required for Streamlit apps.
    allow_origin_regex=r"https://.*\.streamlit\.app",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["system"])
async def root() -> dict[str, str]:
    """Basic endpoint used to confirm that the deployed API is online."""
    return {
        "status": "ok",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    """Lightweight health check that does not call external providers."""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
    }


app.include_router(base_router)
app.include_router(ingestion_router)
app.include_router(export_router)
app.include_router(hybrid_retrieval_router)
app.include_router(retrieval_pipeline_router)
app.include_router(rag_router)
