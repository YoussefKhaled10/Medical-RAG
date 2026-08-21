from src.helpers.config import settings
from src.services.ClaimExtractor import ClaimExtractor
from src.services.ClaimSupportEvaluator import ClaimSupportEvaluator
from src.services.CitationComplianceValidator import CitationComplianceValidator
from src.services.CitationAccuracyEvaluator import CitationAccuracyEvaluator
from src.services.CitationRepairService import CitationRepairService
from src.services.ContextBuilder import ContextBuilder
from src.services.EvidenceBuilder import EvidenceBuilder
from src.services.EvidenceStrengthClassifier import EvidenceStrengthClassifier
from src.services.PostGenerationSafetyGate import PostGenerationSafetyGate
from src.services.RAGPromptBuilder import RAGPromptBuilder
from src.services.RAGService import RAGService
from src.services.RelevanceGate import RelevanceGate
from src.services.retrieval_pipeline_factory import (
    create_retrieval_pipeline_service,
)
from src.stores.llm import GenerationFactory


def _generation_credentials(
    provider_name: str,
    *,
    model_override: str | None = None,
) -> tuple[str, str]:
    provider_name = provider_name.lower()
    if provider_name == "groq":
        return (
            settings.GROQ_API_KEY,
            model_override
            or getattr(
                settings,
                "GROQ_GENERATION_MODEL",
                "openai/gpt-oss-120b",
            ),
        )
    if provider_name == "gemini":
        return (
            settings.GEMINI_API_KEY,
            model_override
            or getattr(
                settings,
                "GEMINI_GENERATION_MODEL",
                "gemini-2.5-flash",
            ),
        )
    if provider_name == "manus":
        return (
            settings.MANUS_API_KEY,
            model_override
            or getattr(settings, "MANUS_AGENT_PROFILE", "manus-1.6"),
        )
    if provider_name == "glm":
        return (
            getattr(settings, "ZAI_API_KEY", ""),
            model_override
            or getattr(
                settings,
                "GLM_GENERATION_MODEL",
                "glm-4.7-flash",
            ),
        )
    raise ValueError(f"Unsupported generation provider: {provider_name}")


def create_rag_service(
    generation_provider_name: str | None = None,
) -> RAGService:
    answer_provider_name = (
        generation_provider_name
        or getattr(settings, "GENERATION_PROVIDER", "groq")
    ).lower()
    answer_api_key, answer_model = _generation_credentials(
        answer_provider_name
    )
    generation_provider = GenerationFactory.create(
        provider=answer_provider_name,
        api_key=answer_api_key,
        model_name=answer_model,
        base_url=(
            getattr(
                settings,
                "GLM_BASE_URL",
                "https://api.z.ai/api/paas/v4/",
            )
            if answer_provider_name == "glm"
            else None
        ),
        timeout_seconds=(
            getattr(settings, "GLM_TIMEOUT_SECONDS", 120.0)
            if answer_provider_name == "glm"
            else None
        ),
    )

    judge_provider_name = getattr(
        settings,
        "CLAIM_JUDGE_PROVIDER",
        "groq",
    ).lower()
    judge_model_override = getattr(
        settings,
        "CLAIM_JUDGE_MODEL",
        "openai/gpt-oss-20b",
    )
    judge_api_key, judge_model = _generation_credentials(
        judge_provider_name,
        model_override=judge_model_override,
    )
    claim_judge_provider = GenerationFactory.create(
        provider=judge_provider_name,
        api_key=judge_api_key,
        model_name=judge_model,
    )

    claim_extractor = ClaimExtractor()
    citation_validator = CitationComplianceValidator(claim_extractor)
    citation_repair_service = CitationRepairService(
        claim_judge_provider,
        citation_validator,
        max_output_tokens=getattr(
            settings, "CITATION_REPAIR_MAX_OUTPUT_TOKENS", 700
        ),
    )

    claim_support_evaluator = ClaimSupportEvaluator(
        claim_judge_provider,
        support_threshold=getattr(
            settings,
            "CLAIM_SUPPORT_THRESHOLD",
            0.80,
        ),
        max_output_tokens=getattr(
            settings,
            "CLAIM_JUDGE_MAX_OUTPUT_TOKENS",
            300,
        ),
        malformed_response_retries=getattr(
            settings,
            "CLAIM_JUDGE_MALFORMED_RESPONSE_RETRIES",
            1,
        ),
    )

    return RAGService(
        retrieval_pipeline=create_retrieval_pipeline_service(),
        context_builder=ContextBuilder(
            max_context_characters=getattr(
                settings,
                "RAG_MAX_CONTEXT_CHARACTERS",
                24000,
            )
        ),
        prompt_builder=RAGPromptBuilder(),
        generation_provider=generation_provider,
        evidence_strength_classifier=EvidenceStrengthClassifier(
            relevance_threshold=getattr(
                settings, "RAG_RELEVANCE_THRESHOLD", 0.320982
            ),
            strong_threshold=getattr(
                settings, "RAG_STRONG_EVIDENCE_THRESHOLD", 0.533
            ),
        ),
        relevance_gate=RelevanceGate(
            threshold=getattr(
                settings,
                "RAG_RELEVANCE_THRESHOLD",
                0.320982,
            ),
            minimum_qualified_chunks=getattr(
                settings,
                "RAG_MIN_RELEVANT_CHUNKS",
                1,
            ),
        ),
        claim_extractor=claim_extractor,
        citation_repair_service=citation_repair_service,
        citation_accuracy_evaluator=CitationAccuracyEvaluator(
            minimum_citation_accuracy=getattr(
                settings, "RAG_MIN_CITATION_ACCURACY", 0.95
            ),
            minimum_citation_completeness=getattr(
                settings, "RAG_MIN_CITATION_COMPLETENESS", 1.0
            ),
        ),
        claim_support_evaluator=claim_support_evaluator,
        post_generation_safety_gate=PostGenerationSafetyGate(
            minimum_faithfulness=getattr(
                settings,
                "RAG_MIN_FAITHFULNESS",
                0.90,
            ),
            block_on_any_unsupported_claim=getattr(
                settings,
                "RAG_BLOCK_ON_ANY_UNSUPPORTED_CLAIM",
                True,
            ),
        ),
        claim_judge_provider=claim_judge_provider,
        evidence_builder=EvidenceBuilder(),
    )
