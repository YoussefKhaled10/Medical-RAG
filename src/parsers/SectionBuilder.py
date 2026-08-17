import re

from src.schemas.ingestion import DocumentSection, ParsedElement


class SectionBuilder:
    """Build conservative, hierarchical sections from parsed PDF elements."""

    SUBSECTION_NAMES = {
        "quality statement",
        "rationale",
        "quality measures",
        "process",
        "structure",
        "outcome",
        "source guidance",
        "definitions of terms used in this quality statement",
        "equality and diversity considerations",
    }

    def __init__(self, default_section_title: str = "Document information") -> None:
        self._default_section_title = default_section_title

    @staticmethod
    def _normalize_title(text: str) -> str:
        value = " ".join(text.split()).strip()
        value = re.sub(r"\s*\.{4,}\s*\d+\s*$", "", value)
        return value[:500].strip()

    @staticmethod
    def _is_real_title(element: ParsedElement) -> bool:
        if not element.is_title:
            return False
        text = " ".join(element.text.split()).strip()
        if not text or len(text) > 180:
            return False
        if text.endswith(('.', ',', ';', ':')):
            return False
        if text[0].islower():
            return False
        if re.fullmatch(r"(?:\d+|\d+\s+weeks?|[a-z]+\))", text, re.I):
            return False
        if len(text.split()) > 20:
            return False
        return True

    @staticmethod
    def _is_parent_title(title: str) -> bool:
        return bool(re.match(r"^Quality statement\s+\d+:", title, flags=re.I))

    def build(
        self,
        document_name: str,
        elements: list[ParsedElement],
    ) -> list[DocumentSection]:
        if not document_name.strip():
            raise ValueError("document_name must not be empty")
        if not elements:
            return []

        sections: list[DocumentSection] = []
        current_title = self._default_section_title
        parent_title: str | None = None
        current_elements: list[ParsedElement] = []
        in_contents = False

        def save() -> None:
            if current_elements:
                sections.append(DocumentSection(
                    document_name=document_name.strip(),
                    section_title=current_title,
                    elements=list(current_elements),
                ))

        for element in sorted(elements, key=lambda item: item.element_index):
            if not self._is_real_title(element):
                current_elements.append(element)
                continue

            title = self._normalize_title(element.text)
            lower_title = title.lower()

            # Keep the complete table of contents under one section.
            if lower_title == "contents":
                save()
                current_elements = []
                current_title = "Contents"
                parent_title = None
                in_contents = True
                continue

            if in_contents:
                if lower_title == "quality statements" or element.page_number >= 4:
                    save()
                    current_elements = []
                    in_contents = False
                else:
                    current_elements.append(element.model_copy(update={"is_title": False}))
                    continue

            save()
            current_elements = []

            if self._is_parent_title(title):
                parent_title = title
                current_title = title
            elif lower_title in self.SUBSECTION_NAMES and parent_title:
                current_title = f"{parent_title} | {title}"
            else:
                if lower_title in {"update information", "about this quality standard", "diversity, equality and language", "endorsing organisation", "supporting organisations"}:
                    parent_title = None
                current_title = title

        save()
        return sections
