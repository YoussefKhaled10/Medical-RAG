from src.helpers.config import settings
from src.services.ContextBuilder import ContextBuilder
from src.services.EvidenceBuilder import EvidenceBuilder
from src.services.RAGPromptBuilder import RAGPromptBuilder
from src.services.RAGService import RAGService
from src.services.RelevanceGate import RelevanceGate
from src.services.retrieval_pipeline_factory import (
    create_retrieval_pipeline_service,
)
from src.stores.llm import GenerationFactory


def create_rag_service(
    generation_provider_name: str | None = None,
) -> RAGService:
    provider_name = (
        generation_provider_name
        or getattr(settings, "GENERATION_PROVIDER", "groq")
    ).lower()

    if provider_name == "gemini":
        api_key = getattr(settings, "GEMINI_API_KEY", "")
        model_name = getattr(
            settings,
            "GEMINI_GENERATION_MODEL",
            "gemini-2.5-flash",
        )
    elif provider_name == "groq":
        api_key = getattr(settings, "GROQ_API_KEY", "")
        model_name = getattr(
            settings,
            "GROQ_GENERATION_MODEL",
            "openai/gpt-oss-120b",
        )
    elif provider_name == "manus":
        api_key = getattr(settings, "MANUS_API_KEY", "")
        model_name = getattr(
            settings,
            "MANUS_AGENT_PROFILE",
            "manus-1.6",
        )
    else:
        raise ValueError(
            f"Unsupported generation provider: {provider_name}"
        )

    generation_provider = GenerationFactory.create(
        provider=provider_name,
        api_key=api_key,
        model_name=model_name,
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
        evidence_builder=EvidenceBuilder(),
    )
