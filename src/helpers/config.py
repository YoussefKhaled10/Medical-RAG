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
    
    INTENT_PROVIDER: str
    INTENT_MODEL: str
    INTENT_MODEL_MAX_OUTPUT_TOKENS: int
    GENERATION_PROVIDER: str = "gemini"
    GEMINI_GENERATION_MODEL : str
    
    ZENMUX_API_KEY: str = ""
    GLM_BASE_URL: str = (
        "https://api.z.ai/api/paas/v4/"
    )
    GLM_GENERATION_MODEL: str = (
        "glm-4.7-flash"
    )

    GLM_TIMEOUT_SECONDS: float = 120.0
    
    GEMINI_API_KEY: str = ""
    # Hybrid intelligent chunking
    CHUNKING_STRATEGY: str
    CHUNKING_PROVIDER: str
    CHUNKING_MODEL: str
    CHUNKING_MAX_OUTPUT_TOKENS: int
    CHUNK_MIN_TOKENS: int
    CHUNK_TARGET_TOKENS: int
    CHUNK_MAX_TOKENS: int
    CHUNK_SIMILARITY_THRESHOLD: float
    CHUNK_PLANNER_WINDOW_TOKENS: int
    CHUNK_PLANNER_MAX_BLOCKS: int
    CHUNK_PLANNER_OVERLAP_BLOCKS: int
    CHUNK_PLANNER_CONCURRENCY: int
    CHUNK_PLANNER_TIMEOUT_SECONDS: float
    CHUNK_LLM_COMPLEX_ONLY: bool
    CHUNK_REMOVE_DUPLICATES: bool
    CHUNK_PYTHON_FALLBACK: bool
    CHUNK_NEAR_DUPLICATE_THRESHOLD: float
    CHUNK_BOUNDARY_OVERLAP_WORDS: int

    GROQ_API_KEY : str = ""
    GROQ_GENERATION_MODEL : str
    GEMINI_GENERATION_MODEL: str
    MANUS_API_KEY: str = ""
    MANUS_AGENT_PROFILE: str = "manus-1.6"
    RAG_MAX_CONTEXT_CHARACTERS: int = 24000
    
    RAG_RELEVANCE_THRESHOLD : float = 0.320982
    RAG_MIN_RELEVANT_CHUNKS : int = 1
    
    KEYWORD_TRANSLATION_PROVIDER : str 
    KEYWORD_TRANSLATION_MODEL : str 
    
    CLAIM_JUDGE_PROVIDER: str
    CLAIM_JUDGE_MODEL: str

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
    
    SHOW_DEVELOPER_MODE : bool = False

    # Response Variation and Diversity Settings
    RAG_ENABLE_RESPONSE_VARIATION: bool = True
    RAG_DEFAULT_TEMPERATURE: float = 0.30
    RAG_MIN_TEMPERATURE: float = 0.20
    RAG_MAX_TEMPERATURE: float = 0.38
    RAG_TOP_P: float = 0.90
    RAG_VARIATION_PROFILE_COUNT: int = 4
    RAG_MAX_ANSWER_SENTENCES: int = 6

    # Calibrated Refusal and Retrieval Gating Settings
    RAG_ENABLE_CALIBRATED_RELEVANCE: bool = True
    RAG_ENABLE_RETRIEVAL_RETRY: bool = True
    RAG_ENABLE_EXPANDED_INTENTS: bool = True
    RAG_RELEVANCE_WEAK_RECOVERABLE_THRESHOLD: float = 0.26

    # Dynamic Turn-by-Turn Language Switching Settings
    RAG_ENABLE_DYNAMIC_LANGUAGE: bool = True


    # Authentication and JWT Settings
    JWT_SECRET_KEY: str = "recoverypath_secret_jwt_key_2026_super_secure_salt_9988"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30
    GLOBAL_PROJECT_ID: int = 2

    # Gmail SMTP Settings
    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT: int = 465
    SMTP_USERNAME: str = ""
    SMTP_PASSWORD: str = ""
    EMAIL_FROM: str = "RecoveryPath AI <no-reply@recoverypath.ai>"


    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH,
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


def get_settings() -> Settings:
    return Settings()


settings = get_settings()
