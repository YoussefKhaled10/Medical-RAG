from abc import ABC, abstractmethod

from src.schemas.ingestion import DocumentSection, SemanticChunk


class ChunkerInterface(ABC):
    """Contract implemented by document chunking strategies."""

    @abstractmethod
    async def chunk(
        self,
        sections: list[DocumentSection],
    ) -> list[SemanticChunk]:
        """Convert document sections into ordered semantic chunks."""
        raise NotImplementedError
