import re
from pathlib import Path
from typing import Any

from unstructured.documents.elements import Table, Title
from unstructured.partition.pdf import partition_pdf

from src.schemas.ingestion import ParsedElement

from .PDFParserInterface import PDFParserInterface


class UnstructuredPDFParser(PDFParserInterface):
    """Parse text PDFs with Unstructured and normalize their elements."""

    SKIPPED_CATEGORIES = {
        "pagebreak",
        "image",
        "figurecaption",
    }

    def __init__(
        self,
        strategy: str = "fast",
        infer_table_structure: bool = True,
        include_page_breaks: bool = False,
        skip_headers_and_footers: bool = False,
    ) -> None:
        allowed_strategies = {"auto", "fast", "hi_res", "ocr_only"}
        if strategy not in allowed_strategies:
            raise ValueError(
                f"Unsupported parsing strategy: {strategy}. "
                f"Allowed values: {sorted(allowed_strategies)}"
            )

        self._strategy = strategy
        self._infer_table_structure = infer_table_structure
        self._include_page_breaks = include_page_breaks
        self._skip_headers_and_footers = skip_headers_and_footers

    @staticmethod
    def _clean_text(text: str | None) -> str:
        if not text:
            return ""

        normalized = text.replace("\u00a0", " ").replace("\u200b", "")
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r" *\n *", "\n", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)

        # Join words split by PDF line wrapping, while preserving real hyphens.
        normalized = re.sub(
            r"(?<=[A-Za-z])-[ \t]*\n(?=[a-z])",
            "",
            normalized,
        )
        return normalized.strip()

    @staticmethod
    def _get_category(element: Any) -> str:
        category = getattr(element, "category", None)
        if category:
            return str(category)
        return element.__class__.__name__

    @staticmethod
    def _extract_metadata(element: Any) -> dict[str, Any]:
        metadata = getattr(element, "metadata", None)
        if metadata is None:
            return {}

        if hasattr(metadata, "to_dict"):
            raw_metadata = metadata.to_dict()
        else:
            raw_metadata = dict(getattr(metadata, "__dict__", {}))

        allowed_keys = {
            "filename",
            "filetype",
            "languages",
            "parent_id",
            "category_depth",
            "text_as_html",
            "coordinates",
        }
        return {
            key: value
            for key, value in raw_metadata.items()
            if key in allowed_keys and value is not None
        }

    def parse(self, pdf_path: str | Path) -> list[ParsedElement]:
        path = Path(pdf_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PDF file was not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {path.suffix}")

        raw_elements = partition_pdf(
            filename=str(path),
            strategy=self._strategy,
            infer_table_structure=self._infer_table_structure,
            include_page_breaks=self._include_page_breaks,
        )

        parsed_elements: list[ParsedElement] = []
        current_page = 1

        for raw_index, element in enumerate(raw_elements):
            category = self._get_category(element)
            category_key = category.lower()

            if category_key in self.SKIPPED_CATEGORIES:
                continue
            if self._skip_headers_and_footers and category_key in {
                "header",
                "footer",
            }:
                continue

            metadata_object = getattr(element, "metadata", None)
            page_number = getattr(metadata_object, "page_number", None)
            if page_number is not None:
                current_page = max(1, int(page_number))

            text = self._clean_text(str(element))
            if not text:
                continue

            parsed_elements.append(
                ParsedElement(
                    element_index=len(parsed_elements),
                    text=text,
                    category=category,
                    page_number=current_page,
                    is_title=(
                        isinstance(element, Title)
                        or category_key == "title"
                    ),
                    is_table=(
                        isinstance(element, Table)
                        or category_key == "table"
                    ),
                    metadata={
                        "source_element_index": raw_index,
                        **self._extract_metadata(element),
                    },
                )
            )

        if not parsed_elements:
            raise ValueError(
                "The PDF parser returned no usable text elements. "
                "Try strategy='hi_res' or 'ocr_only' for scanned PDFs."
            )

        return parsed_elements
