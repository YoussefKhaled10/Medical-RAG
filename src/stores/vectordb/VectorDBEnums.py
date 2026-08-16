from enum import Enum


class VectorDBEnums(str, Enum):
    PGVECTOR = "PGVECTOR"


class DistanceMethodEnums(str, Enum):
    COSINE = "cosine"
    DOT = "dot"


class PgVectorTableSchemeEnums(str, Enum):
    ID = "id"
    TEXT = "text"
    VECTOR = "vector"
    CHUNK_ID = "chunk_id"
    METADATA = "metadata"
    PREFIX = "collection_"


class PgVectorDistanceMethodEnums(str, Enum):
    COSINE = "vector_cosine_ops"
    DOT = "vector_ip_ops"


class PgVectorDistanceOperatorEnums(str, Enum):
    COSINE = "<=>"
    DOT = "<#>"


class PgVectorIndexTypeEnums(str, Enum):
    HNSW = "hnsw"
    IVFFLAT = "ivfflat"
