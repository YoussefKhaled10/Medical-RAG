from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    source_id: str
    excerpt: str
    citation: str
    document_name: str | None
    section_title: str | None
    page_number: int | None
    chunk_id: str | None
    rerank_score: float | None


class EvidenceBuilder:
    """Build exact evidence excerpts and verifiable citations from metadata."""

    @staticmethod
    def _format_citation(
        *,
        document_name: str | None,
        section_title: str | None,
        page_number: int | None,
    ) -> str:
        document = document_name or "Unknown document"
        section = section_title or "Unknown section"
        page = str(page_number) if page_number is not None else "Unknown"
        return f"[{document}, {section}, Page {page}]"

    @staticmethod
    def _result_key(result: dict[str, Any]) -> tuple[Any, Any]:
        return result.get("asset_id"), result.get("chunk_id")

    def build(
        self,
        *,
        used_sources: list[dict[str, Any]],
        retrieval_results: list[dict[str, Any]],
    ) -> list[EvidenceItem]:
        result_by_key = {
            self._result_key(result): result
            for result in retrieval_results
        }
        result_by_chunk_id = {
            result.get("chunk_id"): result
            for result in retrieval_results
            if result.get("chunk_id") is not None
        }

        evidence: list[EvidenceItem] = []
        for source in used_sources:
            result = result_by_key.get(
                (source.get("asset_id"), source.get("chunk_id"))
            )
            if result is None:
                result = result_by_chunk_id.get(source.get("chunk_id"))
            if result is None:
                continue

            excerpt = str(result.get("text") or "").strip()
            if not excerpt:
                continue

            document_name = source.get("document_name")
            section_title = source.get("section_title")
            page_number = source.get("page_number")

            evidence.append(
                EvidenceItem(
                    source_id=str(source["source_id"]),
                    excerpt=excerpt,
                    citation=self._format_citation(
                        document_name=document_name,
                        section_title=section_title,
                        page_number=page_number,
                    ),
                    document_name=document_name,
                    section_title=section_title,
                    page_number=page_number,
                    chunk_id=source.get("chunk_id"),
                    rerank_score=source.get("rerank_score"),
                )
            )

        return evidence
