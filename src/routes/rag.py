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
    generation_provider: Literal[
        "gemini",
        "groq",
        "manus",
        "glm",
    ] | None = None
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


class RAGEvidenceStrength(BaseModel):
    level: Literal["insufficient", "moderate", "strong"]
    top_score: float | None
    relevance_threshold: float
    strong_threshold: float
    language_policy: str
    answer_allowed: bool
    rationale: str


class RAGRefusalGuidance(BaseModel):
    category: str
    requires_professional: bool
    urgent: bool


class RAGRefusal(BaseModel):
    reason: str
    stage: Literal[
        "pre_generation",
        "generation",
        "post_generation",
    ]
    generation_skipped: bool


class RAGClaim(BaseModel):
    claim_id: str
    text: str
    cited_source_ids: tuple[str, ...]
    sentence_index: int


class RAGClaimResult(BaseModel):
    claim_id: str
    claim: str
    cited_source_ids: tuple[str, ...]
    evaluated_source_ids: tuple[str, ...]
    supported: bool
    support_score: float
    reason: str


class RAGCitationRepair(BaseModel):
    attempted: bool
    repaired: bool
    initial_passed: bool
    final_passed: bool
    reason: str


class RAGCitationEvaluationItem(BaseModel):
    claim_id: str
    source_id: str
    source_exists: bool
    evidence_exists: bool
    document_name_matches: bool
    section_title_matches: bool
    page_number_matches: bool
    chunk_id_matches: bool
    claim_support_passed: bool
    metadata_correct: bool
    correct: bool
    reason: str


class RAGCitationEvaluation(BaseModel):
    total_claims: int
    cited_claims: int
    uncited_claims: int
    citation_completeness: float
    total_citation_links: int
    correct_citation_links: int
    incorrect_citation_links: int
    citation_accuracy: float | None
    unique_source_count: int
    invalid_source_ids: tuple[str, ...]
    metadata_accuracy: float | None
    claim_support_accuracy: float | None
    passed: bool
    minimum_citation_accuracy: float
    minimum_citation_completeness: float
    items: tuple[RAGCitationEvaluationItem, ...]
    reason: str | None = None


class RAGClaimValidation(BaseModel):
    passed: bool
    reason: str
    total_claims: int
    supported_claims: int
    unsupported_claims: int
    faithfulness: float
    minimum_faithfulness: float
    unsupported_claim_ids: tuple[str, ...]


class RAGResponse(BaseModel):
    question: str
    answer: str
    recommendation: str
    answer_language: str
    grounded: bool
    refused: bool
    safety_flagged: bool
    refusal: RAGRefusal | None
    refusal_guidance: RAGRefusalGuidance | None = None
    relevance: RAGRelevance
    evidence_strength: RAGEvidenceStrength
    provider: str | None
    model: str | None
    request_id: str | None
    sources: list[RAGSource]
    evidence: list[RAGEvidence]
    claims: list[RAGClaim]
    claim_results: list[RAGClaimResult]
    citation_repair: RAGCitationRepair
    citation_evaluation: RAGCitationEvaluation
    claim_validation: RAGClaimValidation
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
        raise HTTPException(
            status_code=exc.status_code,
            detail=detail,
        ) from exc
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
        safety_flagged=output["safety_flagged"],
        refusal=(
            RAGRefusal(**output["refusal"])
            if output["refusal"] is not None
            else None
        ),
        refusal_guidance=(
            RAGRefusalGuidance(**output["refusal_guidance"])
            if output.get("refusal_guidance")
            else None
        ),
        relevance=RAGRelevance(**output["relevance"]),
        evidence_strength=RAGEvidenceStrength(
            **output["evidence_strength"]
        ),
        provider=output["provider"],
        model=output["model"],
        request_id=output["request_id"],
        sources=[RAGSource(**item) for item in output["sources"]],
        evidence=[RAGEvidence(**item) for item in output["evidence"]],
        claims=[RAGClaim(**item) for item in output["claims"]],
        claim_results=[
            RAGClaimResult(**item)
            for item in output["claim_results"]
        ],
        citation_repair=RAGCitationRepair(
            **output["citation_repair"]
        ),
        citation_evaluation=RAGCitationEvaluation(
            **output["citation_evaluation"]
        ),
        claim_validation=RAGClaimValidation(
            **output["claim_validation"]
        ),
        context_characters=output["context_characters"],
        generation_config=output["generation_config"],
        timings_ms=output["timings_ms"],
        retrieval_summary={
            "pre_dedup_count": retrieval["pre_dedup_count"],
            "post_dedup_count": retrieval["post_dedup_count"],
            "final_result_count": len(retrieval["results"]),
            "removed_duplicate_count": len(
                retrieval["removed_duplicates"]
            ),
            "cross_language_keyword_used": retrieval.get(
                "cross_language_keyword_used",
                False,
            ),
            "effective_keyword_query": retrieval.get(
                "effective_keyword_query"
            ),
            "chunk_ids": [
                item["chunk_id"]
                for item in retrieval["results"]
            ],
        },
    )
