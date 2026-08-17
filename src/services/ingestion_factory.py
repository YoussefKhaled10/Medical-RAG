from src.chunkers import SemanticChunker
from src.helpers.config import settings
from src.parsers import PyMuPDFParser, SectionBuilder
from src.services import IngestionService
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
    chunker = SemanticChunker(
        embedding_provider=embedding_provider,
        similarity_threshold=0.50,
        minimum_tokens=180,
        target_tokens=350,
        maximum_tokens=550,
    )
    return IngestionService(
        parser=parser,
        section_builder=SectionBuilder(),
        semantic_chunker=chunker,
        embedding_provider=embedding_provider,
        vector_db=vector_db,
    )
