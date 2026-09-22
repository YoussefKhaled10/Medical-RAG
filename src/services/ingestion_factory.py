from src.chunkers import HybridIntelligentChunker, SemanticChunker
from src.helpers.config import settings
from src.parsers import PyMuPDFParser, SectionBuilder, TextParser
from src.services.IngestionService import IngestionService
from src.stores.llm import GenerationFactory
from src.stores.llm.LLMFactory import LLMFactory
from src.stores.vectordb.VectorDBFactory import VectorDBFactory



def create_ingestion_service() -> IngestionService:
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
    parser = PyMuPDFParser(
        title_size_ratio=1.25,
        repeated_text_ratio=0.30,
    )
    text_parser = TextParser(
        lines_per_page=50,
    )
    fallback_chunker = SemanticChunker(
        embedding_provider=embedding_provider,
        similarity_threshold=settings.CHUNK_SIMILARITY_THRESHOLD,
        minimum_tokens=settings.CHUNK_MIN_TOKENS,
        target_tokens=settings.CHUNK_TARGET_TOKENS,
        maximum_tokens=settings.CHUNK_MAX_TOKENS,
    )
    if settings.CHUNKING_STRATEGY.lower() == "hybrid_llm":
        planner_provider = GenerationFactory.create(
            provider=settings.CHUNKING_PROVIDER,
            api_key=settings.GEMINI_API_KEY,
            model_name=settings.CHUNKING_MODEL,
        )
        chunker = HybridIntelligentChunker(
            planner_provider=planner_provider,
            fallback_chunker=fallback_chunker,
            minimum_tokens=settings.CHUNK_MIN_TOKENS,
            target_tokens=settings.CHUNK_TARGET_TOKENS,
            maximum_tokens=settings.CHUNK_MAX_TOKENS,
            planner_window_tokens=settings.CHUNK_PLANNER_WINDOW_TOKENS,
            planner_max_blocks=settings.CHUNK_PLANNER_MAX_BLOCKS,
            planner_overlap_blocks=settings.CHUNK_PLANNER_OVERLAP_BLOCKS,
            planner_max_output_tokens=settings.CHUNKING_MAX_OUTPUT_TOKENS,
            planner_concurrency=settings.CHUNK_PLANNER_CONCURRENCY,
            planner_timeout_seconds=settings.CHUNK_PLANNER_TIMEOUT_SECONDS,
            use_llm_only_for_complex_sections=settings.CHUNK_LLM_COMPLEX_ONLY,
            remove_duplicates=settings.CHUNK_REMOVE_DUPLICATES,
            near_duplicate_threshold=settings.CHUNK_NEAR_DUPLICATE_THRESHOLD,
            boundary_overlap_words=settings.CHUNK_BOUNDARY_OVERLAP_WORDS,
        )
    else:
        chunker = fallback_chunker
    return IngestionService(
        parser=parser,
        section_builder=SectionBuilder(),
        semantic_chunker=chunker,
        embedding_provider=embedding_provider,
        vector_db=vector_db,
        text_parser=text_parser,
    )

