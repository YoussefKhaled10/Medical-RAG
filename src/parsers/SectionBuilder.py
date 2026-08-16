import re

from src.schemas.ingestion import DocumentSection, ParsedElement


class SectionBuilder:
    """Group consecutive parsed elements under meaningful section titles."""

    def __init__(
        self,
        default_section_title: str = "Introduction",
        detect_title_patterns: bool = True,
    ) -> None:
        normalized_title = default_section_title.strip()
        if not normalized_title:
            raise ValueError("default_section_title must not be empty")

        self._default_section_title = normalized_title
        self._detect_title_patterns = detect_title_patterns

    @staticmethod
    def _looks_like_title(element: ParsedElement) -> bool:
        if element.is_title:
            return True

        text = element.text.strip()
        if not text or len(text) > 500 or "\n" in text:
            return False

        words = text.split()
        numbered_title = bool(
            re.match(
                r"^(?:\d+(?:\.\d+)*|[A-Z]|[IVXLC]+)[.)]?\s+\S+",
                text,
            )
        )
        known_heading = bool(
            re.match(
                r"^(?:contents|quality statement|rationale|quality measures|"
                r"source guidance|definitions? of terms|equality and diversity|"
                r"recommendations?|introduction|overview|references?)\b",
                text,
                flags=re.IGNORECASE,
            )
        )
        uppercase_title = text.isupper() and 1 <= len(words) <= 15

        return numbered_title or known_heading or uppercase_title

    def build(
        self,
        document_name: str,
        elements: list[ParsedElement],
    ) -> list[DocumentSection]:
        normalized_document_name = document_name.strip()
        if not normalized_document_name:
            raise ValueError("document_name must not be empty")
        if not elements:
            return []

        ordered_elements = sorted(elements, key=lambda item: item.element_index)
        sections: list[DocumentSection] = []
        current_title = self._default_section_title
        current_elements: list[ParsedElement] = []

        def save_current_section() -> None:
            if not current_elements:
                return

            sections.append(
                DocumentSection(
                    document_name=normalized_document_name,
                    section_title=current_title,
                    elements=list(current_elements),
                )
            )

        for element in ordered_elements:
            is_title = element.is_title or (
                self._detect_title_patterns
                and self._looks_like_title(element)
            )

            if is_title:
                save_current_section()
                current_title = element.text[:500]
                current_elements = []
                continue

            current_elements.append(element)

        save_current_section()
        return sections
