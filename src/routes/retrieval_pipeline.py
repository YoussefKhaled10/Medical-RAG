from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import get_db_session
from src.services.retrieval_pipeline_factory import (
    create_retrieval_pipeline_service,
)
import logging
logger = logging.getLogger(__name__)

retrieval_pipeline_router = APIRouter(
    prefix="/api/v1/retrieval",
    tags=["Retrieval"],
)


class RetrievalPipelineRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=20)
    project_id: int | None = Field(default=None, gt=0)
    asset_id: int | None = Field(default=None, gt=0)
    use_deduplication: bool = True
    use_reranking: bool = True
    use_cross_language_keyword: bool | None = None


class PipelineResult(BaseModel):
    rank: int
    asset_id: int
    project_id: int | None
    chunk_id: str
    document_name: str
    section_title: str
    page_number: int
    text: str
    semantic_rank: int | None
    semantic_score: float | None
    keyword_rank: int | None
    keyword_score: float | None
    rrf_score: float
    pre_rerank_rank: int | None
    rerank_score: float | None


class RetrievalPipelineResponse(BaseModel):
    query: str
    pipeline: list[str]
    use_deduplication: bool
    use_reranking: bool
    cross_language_keyword_used: bool
    effective_keyword_query: str
    candidate_limit: int
    pre_dedup_count: int
    post_dedup_count: int
    removed_duplicates: list[dict[str, Any]]
    total_results: int
    filters: dict[str, Any]
    results: list[PipelineResult]


@retrieval_pipeline_router.post(
    "/pipeline-search",
    response_model=RetrievalPipelineResponse,
    summary="Hybrid RRF search with deduplication and reranking",
)
async def pipeline_search(
    request: RetrievalPipelineRequest,
    session: AsyncSession = Depends(get_db_session),
) -> RetrievalPipelineResponse:
    service = create_retrieval_pipeline_service()
    try:
        output = await service.search(
            session=session,
            query=request.query,
            limit=request.limit,
            project_id=request.project_id,
            asset_id=request.asset_id,
            use_deduplication=request.use_deduplication,
            use_reranking=request.use_reranking,
            use_cross_language_keyword=(
                request.use_cross_language_keyword
            ),
        )
    except Exception as exc:
        logger.exception(
            "Retrieval pipeline failed for query=%r",
            request.query,
        )

        raise HTTPException(
            status_code=500,
            detail={
                "message": "Retrieval pipeline failed.",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        ) from exc
    finally:
        await service.close()

    filters: dict[str, Any] = {}
    if request.project_id is not None:
        filters["project_id"] = request.project_id
    if request.asset_id is not None:
        filters["asset_id"] = request.asset_id

    translation_step = (
        "arabic_to_english_keyword_translation"
        if output["cross_language_keyword_used"]
        else "original_keyword_query"
    )
    return RetrievalPipelineResponse(
        query=request.query,
        pipeline=[
            "semantic_top_20",
            translation_step,
            "keyword_top_20",
            "rrf_fusion",
            (
                "deduplication"
                if request.use_deduplication
                else "deduplication_disabled"
            ),
            (
                "cohere_rerank"
                if request.use_reranking
                else "reranking_disabled"
            ),
            f"final_top_{request.limit}",
        ],
        use_deduplication=request.use_deduplication,
        use_reranking=request.use_reranking,
        cross_language_keyword_used=output[
            "cross_language_keyword_used"
        ],
        effective_keyword_query=output["effective_keyword_query"],
        candidate_limit=20,
        pre_dedup_count=output["pre_dedup_count"],
        post_dedup_count=output["post_dedup_count"],
        removed_duplicates=output["removed_duplicates"],
        total_results=len(output["results"]),
        filters=filters,
        results=[PipelineResult(**item) for item in output["results"]],
    )
