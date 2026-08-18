from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import get_db_session
from src.services.rag_factory import create_rag_service
from src.stores.llm.GenerationExceptions import GenerationProviderError


rag_router = APIRouter(prefix="/api/v1/rag", tags=["RAG"])


class RAGRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4000)
    project_id: int | None = Field(default=None, gt=0)
    asset_id: int | None = Field(default=None, gt=0)
    retrieval_limit: int = Field(default=5, ge=1, le=10)
    generation_provider: Literal["gemini", "groq", "manus"] | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1200, ge=64, le=8192)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = " ".join(value.split()).strip()
        if not normalized:
            raise ValueError("question must not be empty")
        return normalized


class RAGSource(BaseModel):
    source_id: str
    document_name: str | None
    section_title: str | None
    page_number: int | None
    chunk_id: str | None
    rerank_score: float | None


class RAGEvidence(RAGSource):
    excerpt: str
    citation: str


class RAGRelevance(BaseModel):
    passed: bool
    reason: str
    top_score: float | None
    second_score: float | None
    score_margin: float | None
    threshold: float
    qualified_chunk_count: int
    minimum_qualified_chunks: int


class RAGRefusal(BaseModel):
    reason: str
    stage: Literal["pre_generation", "generation"]
    generation_skipped: bool


class RAGResponse(BaseModel):
    question: str
    answer: str
    recommendation: str
    answer_language: Literal["ar", "en"]
    grounded: bool
    refused: bool
    refusal: RAGRefusal | None
    relevance: RAGRelevance
    provider: str | None
    model: str | None
    request_id: str | None
    sources: list[RAGSource]
    evidence: list[RAGEvidence]
    context_characters: int
    generation_config: dict[str, Any] | None
    timings_ms: dict[str, float]
    retrieval_summary: dict[str, Any]


@rag_router.post(
    "/ask",
    response_model=RAGResponse,
    status_code=status.HTTP_200_OK,
)
async def ask_rag(
    request: RAGRequest,
    session: AsyncSession = Depends(get_db_session),
) -> RAGResponse:
    try:
        service = create_rag_service(request.generation_provider)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        output = await service.ask(
            session=session,
            question=request.question,
            project_id=request.project_id,
            asset_id=request.asset_id,
            retrieval_limit=request.retrieval_limit,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
        )
    except GenerationProviderError as exc:
        detail: dict[str, Any] = {
            "message": str(exc),
            "provider": exc.provider,
            "retryable": exc.retryable,
        }
        if exc.retry_after_seconds is not None:
            detail["retry_after_seconds"] = exc.retry_after_seconds
        raise HTTPException(status_code=exc.status_code, detail=detail) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "RAG answer generation failed.",
                "error": str(exc),
            },
        ) from exc
    finally:
        await service.close()

    retrieval = output["retrieval"]
    return RAGResponse(
        question=output["question"],
        answer=output["answer"],
        recommendation=output["recommendation"],
        answer_language=output["answer_language"],
        grounded=output["grounded"],
        refused=output["refused"],
        refusal=(
            RAGRefusal(**output["refusal"])
            if output["refusal"] is not None
            else None
        ),
        relevance=RAGRelevance(**output["relevance"]),
        provider=output["provider"],
        model=output["model"],
        request_id=output["request_id"],
        sources=[RAGSource(**source) for source in output["sources"]],
        evidence=[RAGEvidence(**item) for item in output["evidence"]],
        context_characters=output["context_characters"],
        generation_config=output["generation_config"],
        timings_ms=output["timings_ms"],
        retrieval_summary={
            "pre_dedup_count": retrieval["pre_dedup_count"],
            "post_dedup_count": retrieval["post_dedup_count"],
            "final_result_count": len(retrieval["results"]),
            "removed_duplicate_count": len(retrieval["removed_duplicates"]),
            "cross_language_keyword_used": retrieval.get(
                "cross_language_keyword_used", False
            ),
            "effective_keyword_query": retrieval.get(
                "effective_keyword_query"
            ),
            "chunk_ids": [
                item["chunk_id"] for item in retrieval["results"]
            ],
        },
    )
