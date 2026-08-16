from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from .ParsedElement import ParsedElement

class DocumentSection(BaseModel):
    """A group of consecutive parsed elements under one section title."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    document_name: str = Field(min_length=1, max_length=255)
    section_title: str = Field(
        default="Introduction",
        min_length=1,
        max_length=500,
    )
    elements: list[ParsedElement] = Field(min_length=1)

    @field_validator("elements")
    @classmethod
    def validate_element_order(
        cls,
        elements: list[ParsedElement],
    ) -> list[ParsedElement]:
        indexes = [element.element_index for element in elements]
        if indexes != sorted(indexes):
            raise ValueError("Section elements must be ordered by element_index")
        return elements

    @computed_field
    @property
    def page_number(self) -> int:
        return min(element.page_number for element in self.elements)

    @computed_field
    @property
    def page_end(self) -> int:
        return max(element.page_number for element in self.elements)

    @computed_field
    @property
    def text(self) -> str:
        return "\n\n".join(
            element.text
            for element in self.elements
            if not element.is_title
        ).strip()
