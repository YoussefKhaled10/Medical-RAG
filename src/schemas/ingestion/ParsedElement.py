from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ParsedElement(BaseModel):
    """A normalized element extracted from a source document."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    element_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    category: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    is_title: bool = False
    is_table: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Parsed element text must not be empty")
        return normalized
