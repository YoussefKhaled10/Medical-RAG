import asyncio
from collections.abc import Sequence

import cohere

from ..LLMEnums import (
    CohereEmbeddingTypeEnums,
    CohereInputTypeEnums,
    CohereTruncateEnums,
)
from ..LLMInterface import LLMInterface


class CohereProvider(LLMInterface):
    """Cohere text-embedding provider."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "embed-v4.0",
        embedding_dimension: int = 1536,
        batch_size: int = 96,
        truncate: str = CohereTruncateEnums.END.value,
    ) -> None:
        if not api_key:
            raise ValueError("Cohere API key must not be empty")
        if embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be greater than zero")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        self._client = cohere.ClientV2(api_key=api_key)
        self._model_name = model_name
        self._embedding_dimension = embedding_dimension
        self._batch_size = batch_size
        self._truncate = truncate

    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dimension

    @staticmethod
    def _validate_texts(texts: Sequence[str]) -> list[str]:
        normalized = [text.strip() for text in texts]
        if not normalized:
            raise ValueError("At least one text is required")
        if any(not text for text in normalized):
            raise ValueError("Embedding texts must not be empty")
        return normalized

    def _embed_sync(
        self,
        texts: list[str],
        input_type: str,
    ) -> list[list[float]]:
        response = self._client.embed(
            texts=texts,
            model=self._model_name,
            input_type=input_type,
            embedding_types=[CohereEmbeddingTypeEnums.FLOAT.value],
            truncate=self._truncate,
        )

        embeddings = response.embeddings.float
        if embeddings is None:
            raise RuntimeError("Cohere returned no float embeddings")

        result = [list(map(float, embedding)) for embedding in embeddings]
        if len(result) != len(texts):
            raise RuntimeError("Cohere returned an unexpected embedding count")
        if any(len(vector) != self._embedding_dimension for vector in result):
            raise RuntimeError("Cohere returned an unexpected embedding dimension")
        return result

    async def _embed(
        self,
        texts: Sequence[str],
        input_type: str,
    ) -> list[list[float]]:
        normalized = self._validate_texts(texts)
        embeddings: list[list[float]] = []

        for start in range(0, len(normalized), self._batch_size):
            batch = normalized[start : start + self._batch_size]
            batch_embeddings = await asyncio.to_thread(
                self._embed_sync,
                batch,
                input_type,
            )
            embeddings.extend(batch_embeddings)

        return embeddings

    async def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        return await self._embed(
            texts=texts,
            input_type=CohereInputTypeEnums.SEARCH_DOCUMENT.value,
        )

    async def embed_query(self, text: str) -> list[float]:
        embeddings = await self._embed(
            texts=[text],
            input_type=CohereInputTypeEnums.SEARCH_QUERY.value,
        )
        return embeddings[0]

    async def health_check(self) -> bool:
        try:
            embedding = await self.embed_query("health check")
            return len(embedding) == self._embedding_dimension
        except Exception:
            return False

    async def close(self) -> None:
        # Cohere ClientV2 does not require explicit connection cleanup.
        return None
