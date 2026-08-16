import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SemanticChunk(BaseModel):
    """The final public chunk contract produced by semantic chunking."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    chunk_id: str = Field(pattern=r"^chunk_\d{4,}$")
    document_name: str = Field(min_length=1, max_length=255)
    section_title: str = Field(min_length=1, max_length=500)
    page_number: int = Field(ge=1)
    text: str = Field(min_length=1)

    @field_validator("chunk_id")
    @classmethod
    def normalize_chunk_id(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not re.fullmatch(r"chunk_\d{4,}", normalized):
            raise ValueError(
                "chunk_id must follow the format chunk_0001"
            )
        return normalized

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Semantic chunk text must not be empty")
        return normalized

    def to_database_dict(
        self,
        asset_id: int,
        chunk_index: int,
        token_count: int,
        page_end: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add internal persistence fields without changing the public model."""
        if asset_id <= 0:
            raise ValueError("asset_id must be greater than zero")
        if chunk_index <= 0:
            raise ValueError("chunk_index must be greater than zero")
        if token_count < 0:
            raise ValueError("token_count must not be negative")
        if page_end is not None and page_end < self.page_number:
            raise ValueError("page_end cannot be before page_number")

        return {
            **self.model_dump(),
            "asset_id": asset_id,
            "chunk_index": chunk_index,
            "token_count": token_count,
            "page_end": page_end,
            "metadata": metadata or {},
        }
