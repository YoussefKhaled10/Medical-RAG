from src.helpers.config import settings
from src.services.CrossLanguageKeywordTranslator import (
    CrossLanguageKeywordTranslator,
)
from src.services.HybridRetrievalService import HybridRetrievalService
from src.stores.llm import GenerationFactory
from src.stores.llm.LLMFactory import LLMFactory
from src.stores.vectordb.VectorDBFactory import VectorDBFactory


def _get_keyword_translation_credentials() -> tuple[str, str, str]:
    """Return provider name, API key, and model for keyword translation."""

    provider_name = getattr(
        settings,
        "KEYWORD_TRANSLATION_PROVIDER",
        "groq",
    ).strip().lower()

    configured_model = getattr(
        settings,
        "KEYWORD_TRANSLATION_MODEL",
        "",
    ).strip()

    if provider_name == "groq":
        api_key = getattr(settings, "GROQ_API_KEY", "").strip()
        model_name = configured_model or getattr(
            settings,
            "GROQ_GENERATION_MODEL",
            "openai/gpt-oss-20b",
        ).strip()
    elif provider_name == "gemini":
        api_key = getattr(settings, "GEMINI_API_KEY", "").strip()
        model_name = configured_model or getattr(
            settings,
            "GEMINI_GENERATION_MODEL",
            "gemini-2.5-flash",
        ).strip()
    else:
        raise ValueError(
            "Unsupported keyword translation provider: "
            f"{provider_name}. Supported providers: groq, gemini"
        )

    if not api_key:
        raise ValueError(
            f"API key is missing for keyword translation provider: "
            f"{provider_name}"
        )
    if not model_name:
        raise ValueError(
            f"Model name is missing for keyword translation provider: "
            f"{provider_name}"
        )

    return provider_name, api_key, model_name


def create_hybrid_retrieval_service() -> HybridRetrievalService:
    embedding_provider = LLMFactory.create(
        provider=settings.EMBEDDING_BACKEND,
        api_key=settings.COHERE_API_KEY,
        model_name=settings.COHERE_EMBEDDING_MODEL,
        embedding_dimension=settings.EMBEDDING_MODEL_SIZE,
        batch_size=settings.COHERE_EMBEDDING_BATCH_SIZE,
        truncate=settings.COHERE_TRUNCATE,
    )

    vector_db = VectorDBFactory.create(
        provider=settings.VECTOR_DB_PROVIDER,
        database_url=settings.POSTGRES_URL,
        embedding_dimension=settings.EMBEDDING_MODEL_SIZE,
        table_name=settings.VECTOR_DB_TABLE,
        min_pool_size=settings.VECTOR_DB_MIN_POOL_SIZE,
        max_pool_size=settings.VECTOR_DB_MAX_POOL_SIZE,
    )

    (
        translation_provider_name,
        translation_api_key,
        translation_model_name,
    ) = _get_keyword_translation_credentials()

    translator_provider = GenerationFactory.create(
        provider=translation_provider_name,
        api_key=translation_api_key,
        model_name=translation_model_name,
    )

    keyword_translator = CrossLanguageKeywordTranslator(
        translator_provider
    )

    return HybridRetrievalService(
        embedding_provider=embedding_provider,
        vector_db=vector_db,
        semantic_fetch_limit=20,
        keyword_fetch_limit=20,
        rrf_k=60,
        keyword_translator=keyword_translator,
    )
