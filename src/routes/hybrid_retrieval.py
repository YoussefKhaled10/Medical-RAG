from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import get_db_session
from src.services.HybridRetrievalService import SearchType
from src.services.hybrid_retrieval_factory import (
    create_hybrid_retrieval_service,
)


hybrid_retrieval_router = APIRouter(
    prefix="/api/v1/retrieval",
    tags=["Retrieval"],
)


class HybridRetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    search_type: Literal["semantic", "keyword", "hybrid"] = "hybrid"
    limit: int = Field(default=5, ge=1, le=20)
    project_id: int | None = Field(default=None, gt=0)
    asset_id: int | None = Field(default=None, gt=0)
    use_query_rewriting: bool = False
    use_cross_language_keyword: bool = False


class RetrievalResult(BaseModel):
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


class HybridRetrievalResponse(BaseModel):
    query: str
    rewritten_semantic_query: str | None
    rewritten_keyword_query: str | None
    query_expansions: list[str]
    use_query_rewriting: bool
    use_cross_language_keyword: bool
    effective_keyword_query: str
    search_type: str
    candidate_limits: dict[str, int]
    total_results: int
    filters: dict[str, Any]
    results: list[RetrievalResult]


@hybrid_retrieval_router.post(
    "/search",
    response_model=HybridRetrievalResponse,
)
async def search_chunks(
    request: HybridRetrievalRequest,
    session: AsyncSession = Depends(get_db_session),
) -> HybridRetrievalResponse:
    service = create_hybrid_retrieval_service()
    try:
        results, rewritten, effective_keyword_query = await service.search(
            session=session,
            query=request.query,
            search_type=SearchType(request.search_type),
            limit=request.limit,
            project_id=request.project_id,
            asset_id=request.asset_id,
            use_query_rewriting=request.use_query_rewriting,
            use_cross_language_keyword=request.use_cross_language_keyword,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"message": "Retrieval failed", "error": str(exc)},
        ) from exc
    finally:
        await service.close()

    filters: dict[str, Any] = {}
    if request.project_id is not None:
        filters["project_id"] = request.project_id
    if request.asset_id is not None:
        filters["asset_id"] = request.asset_id

    return HybridRetrievalResponse(
        query=request.query,
        rewritten_semantic_query=(
            rewritten.semantic_query
            if request.use_query_rewriting else None
        ),
        rewritten_keyword_query=(
            rewritten.keyword_query
            if request.use_query_rewriting else None
        ),
        query_expansions=(
            list(rewritten.expansions)
            if request.use_query_rewriting else []
        ),
        use_query_rewriting=request.use_query_rewriting,
        use_cross_language_keyword=request.use_cross_language_keyword,
        effective_keyword_query=effective_keyword_query,
        search_type=request.search_type,
        candidate_limits={"semantic": 20, "keyword": 20},
        total_results=len(results),
        filters=filters,
        results=[RetrievalResult(**item) for item in results],
    )
