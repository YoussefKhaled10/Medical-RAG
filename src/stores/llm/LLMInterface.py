from abc import ABC, abstractmethod
from collections.abc import Sequence


class LLMInterface(ABC):
    """Contract for embedding providers used by the RAG pipeline."""

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    async def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """Generate embeddings for documents that will be indexed."""
        raise NotImplementedError

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a search query."""
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError
