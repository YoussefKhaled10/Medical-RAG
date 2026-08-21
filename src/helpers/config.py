from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ENV_FILE_PATH = Path(__file__).resolve().parents[1] / ".env"


class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str

    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int
    POSTGRES_URL: str
    POSTGRES_SYNC_URL: str

    VECTOR_DB_PROVIDER: str = "PGVECTOR"
    VECTOR_DB_TABLE: str = "vector_documents"
    VECTOR_DB_MIN_POOL_SIZE: int = 1
    VECTOR_DB_MAX_POOL_SIZE: int = 10

    EMBEDDING_DIMENSION: int = 384
    VECTOR_DISTANCE_METHOD: str = "cosine"
    VECTOR_INDEX_TYPE: str = "hnsw"
    VECTOR_SEARCH_LIMIT: int = 5
    
    EMBEDDING_BACKEND : str
    COHERE_API_KEY : str
    COHERE_EMBEDDING_MODEL : str
    COHERE_EMBEDDING_BATCH_SIZE : int
    COHERE_TRUNCATE : str
    EMBEDDING_MODEL_SIZE : int = 384
    COHERE_RERANK_MODEL : str = "rerank-v3.5"
    
    GENERATION_PROVIDER: str = "gemini"
    
    ZENMUX_API_KEY: str = ""
    GLM_BASE_URL: str = (
        "https://api.z.ai/api/paas/v4/"
    )
    GLM_GENERATION_MODEL: str = (
        "glm-4.7-flash"
    )

    GLM_TIMEOUT_SECONDS: float = 120.0
    
    GEMINI_API_KEY: str = ""
    GROQ_API_KEY : str = ""
    GROQ_GENERATION_MODEL : str = "openai/gpt-oss-120b"
    GEMINI_GENERATION_MODEL: str = "gemini-2.5-flash"
    MANUS_API_KEY: str = ""
    MANUS_AGENT_PROFILE: str = "manus-1.6"
    RAG_MAX_CONTEXT_CHARACTERS: int = 24000
    
    RAG_RELEVANCE_THRESHOLD : float = 0.320982
    RAG_MIN_RELEVANT_CHUNKS : int = 1
    
    KEYWORD_TRANSLATION_PROVIDER : str = "groq"
    KEYWORD_TRANSLATION_MODEL : str = "openai/gpt-oss-120b"
    
    CLAIM_JUDGE_PROVIDER: str = "groq"
    CLAIM_JUDGE_MODEL: str = "openai/gpt-oss-20b"

    CLAIM_SUPPORT_THRESHOLD: float = 0.80
    CLAIM_JUDGE_MAX_OUTPUT_TOKENS: int = 300

    RAG_MIN_FAITHFULNESS: float = 0.90
    RAG_BLOCK_ON_ANY_UNSUPPORTED_CLAIM: bool = True
    
    CITATION_REPAIR_MAX_OUTPUT_TOKENS: int = 700
    CLAIM_JUDGE_MALFORMED_RESPONSE_RETRIES: int = 1
    
    RAG_MIN_CITATION_ACCURACY: float = 0.95
    RAG_MIN_CITATION_COMPLETENESS: float = 1.0
    
    RAG_MIN_CITATION_ACCURACY: float = 0.95
    RAG_MIN_CITATION_COMPLETENESS: float = 1.0
    CLAIM_JUDGE_MALFORMED_RESPONSE_RETRIES: int = 1
    
    
    RAG_STRONG_EVIDENCE_THRESHOLD: float = 0.533
    RAG_CANDIDATE_ABSOLUTE_THRESHOLD: float = 0.0
    RAG_CANDIDATE_RELATIVE_TO_TOP_RATIO: float = 0.0

    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH,
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


def get_settings() -> Settings:
    return Settings()


settings = get_settings()
