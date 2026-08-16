from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(slots=True, frozen=True)
class VectorDocument:
    id: str
    text: str
    embedding: Sequence[float]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class VectorSearchResult:
    id: str
    text: str
    score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


class VectorDBInterface(ABC):
    @abstractmethod
    async def initialize(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def upsert(self, documents: Sequence[VectorDocument]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def similarity_search(
        self,
        query_embedding: Sequence[float],
        limit: int = 5,
        filters: Mapping[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        raise NotImplementedError

    @abstractmethod
    async def delete_by_ids(self, document_ids: Sequence[str]) -> int:
        raise NotImplementedError

    @abstractmethod
    async def delete_by_asset_id(self, asset_id: int) -> int:
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError
