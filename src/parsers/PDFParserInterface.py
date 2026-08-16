from abc import ABC, abstractmethod
from pathlib import Path

from src.schemas.ingestion import ParsedElement


class PDFParserInterface(ABC):
    """Contract implemented by PDF parsing providers."""

    @abstractmethod
    def parse(self, pdf_path: str | Path) -> list[ParsedElement]:
        """Parse a PDF into normalized, page-aware elements."""
        raise NotImplementedError
