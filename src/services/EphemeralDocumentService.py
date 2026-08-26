import re
import tempfile
from pathlib import Path
from typing import Any

from src.parsers.PyMuPDFParser import PyMuPDFParser
from src.parsers.SectionBuilder import SectionBuilder
from src.parsers.TextParser import TextParser


class EphemeralDocumentService:
    """Parse and extract relevant in-memory chunks from user-uploaded documents with zero DB persistence."""

    def __init__(
        self,
        pdf_parser: PyMuPDFParser | None = None,
        text_parser: TextParser | None = None,
        section_builder: SectionBuilder | None = None,
    ) -> None:
        self._pdf_parser = pdf_parser or PyMuPDFParser()
        self._text_parser = text_parser or TextParser()
        self._section_builder = section_builder or SectionBuilder()

    def parse_document_bytes(
        self,
        file_bytes: bytes,
        file_name: str,
    ) -> list[dict[str, Any]]:
        """Parse raw file bytes in memory into a list of chunk dicts."""
        suffix = Path(file_name).suffix.lower()
        if suffix not in {".pdf", ".txt"}:
            raise ValueError(f"Unsupported file format: {suffix}. Only PDF and TXT are supported.")

        document_name = Path(file_name).stem

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)

        try:
            if suffix == ".pdf":
                elements = self._pdf_parser.parse(tmp_path)
            else:
                elements = self._text_parser.parse(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        if not elements:
            raise ValueError("No readable text could be extracted from the document.")

        sections = self._section_builder.build(
            document_name=document_name,
            elements=elements,
        )

        chunks: list[dict[str, Any]] = []
        chunk_idx = 1
        for section in sections:
            section_text = section.text
            if not section_text or not section_text.strip():
                continue

            # Split large sections into smaller chunks of ~800-1200 characters if needed
            paragraphs = section_text.split("\n\n")
            current_buffer = []
            current_len = 0
            page_num = section.elements[0].page_number if section.elements else 1

            for p in paragraphs:
                p_clean = p.strip()
                if not p_clean:
                    continue
                if current_len + len(p_clean) > 1200 and current_buffer:
                    chunk_content = "\n\n".join(current_buffer)
                    chunks.append({
                        "chunk_id": f"ephemeral_chunk_{chunk_idx}",
                        "document_name": document_name,
                        "section_title": section.section_title,
                        "page_number": page_num,
                        "text": chunk_content,
                        "content": chunk_content,
                        "asset_id": None,
                        "project_id": None,
                    })
                    chunk_idx += 1
                    current_buffer = [p_clean]
                    current_len = len(p_clean)
                else:
                    current_buffer.append(p_clean)
                    current_len += len(p_clean)

            if current_buffer:
                chunk_content = "\n\n".join(current_buffer)
                chunks.append({
                    "chunk_id": f"ephemeral_chunk_{chunk_idx}",
                    "document_name": document_name,
                    "section_title": section.section_title,
                    "page_number": page_num,
                    "text": chunk_content,
                    "content": chunk_content,
                    "asset_id": None,
                    "project_id": None,
                })
                chunk_idx += 1

        return chunks

    @staticmethod
    def _query_tokens(query: str) -> set[str]:
        return {
            token
            for token in re.findall(r"\w+", str(query).casefold())
            if len(token) > 1
        }

    @classmethod
    def _score_chunk(cls, query_tokens: set[str], chunk_text: str) -> float:
        text = str(chunk_text or "").casefold()
        if not text or not query_tokens:
            return 0.0

        words = set(re.findall(r"\w+", text))
        exact_matches = sum(1 for token in query_tokens if token in words)
        partial_matches = sum(
            1 for token in query_tokens
            if token not in words and token in text
        )
        coverage = (exact_matches + (0.5 * partial_matches)) / len(query_tokens)
        return round(min(max(coverage, 0.0), 1.0), 6)

    def rank_chunks_for_query(
        self,
        chunks: list[dict[str, Any]],
        query: str,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        """Rank only the uploaded document using a transparent local score."""
        if limit < 1:
            raise ValueError("limit must be greater than zero")
        if not chunks:
            return []

        query_tokens = self._query_tokens(query)
        scored: list[tuple[float, int, dict[str, Any]]] = []
        for original_index, source in enumerate(chunks):
            item = dict(source)
            score = self._score_chunk(query_tokens, item.get("text", ""))
            item["rerank_score"] = score
            item["ephemeral_score_type"] = "normalized_lexical_coverage"
            item["pre_rerank_rank"] = original_index + 1
            scored.append((score, original_index, item))

        scored.sort(key=lambda row: (-row[0], row[1]))
        selected = [item for _, _, item in scored[:limit]]
        for rank, item in enumerate(selected, start=1):
            item["rank"] = rank
        return selected

