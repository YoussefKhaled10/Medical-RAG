from enum import Enum


class LLMEnums(str, Enum):
    COHERE = "COHERE"


class CohereEmbeddingModelEnums(str, Enum):
    EMBED_V4 = "embed-v4.0"
    EMBED_MULTILINGUAL_V3 = "embed-multilingual-v3.0"
    EMBED_ENGLISH_V3 = "embed-english-v3.0"


class CohereInputTypeEnums(str, Enum):
    SEARCH_DOCUMENT = "search_document"
    SEARCH_QUERY = "search_query"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"


class CohereEmbeddingTypeEnums(str, Enum):
    FLOAT = "float"
    INT8 = "int8"
    UINT8 = "uint8"
    BINARY = "binary"
    UBINARY = "ubinary"


class CohereTruncateEnums(str, Enum):
    NONE = "NONE"
    START = "START"
    END = "END"
