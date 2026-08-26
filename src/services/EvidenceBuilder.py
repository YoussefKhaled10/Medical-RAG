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
    """Build evidence from the exact source content shown to generation."""

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
    def _result_key(item: dict[str, Any]) -> tuple[str, str]:
        return (
            str(item.get("asset_id") or ""),
            str(item.get("chunk_id") or ""),
        )

    def build(
        self,
        *,
        used_sources: list[dict[str, Any]],
        retrieval_results: list[dict[str, Any]],
    ) -> list[EvidenceItem]:
        # Composite keys are mandatory because chunk IDs repeat across assets.
        result_by_key = {
            self._result_key(result): result
            for result in retrieval_results
            if result.get("asset_id") is not None
            and result.get("chunk_id") is not None
        }

        evidence: list[EvidenceItem] = []
        for source in used_sources:
            key = self._result_key(source)
            result = result_by_key.get(key)

            # Prefer the exact source content that was placed in the prompt.
            excerpt = str(
                source.get("content")
                or source.get("text")
                or source.get("excerpt")
                or ""
            ).strip()

            if not excerpt and result is not None:
                excerpt = str(result.get("text") or "").strip()
            if not excerpt:
                continue

            # If both exist, they must represent the same asset/chunk pair.
            document_name = source.get("document_name")
            section_title = source.get("section_title")
            page_number = source.get("page_number")

            evidence.append(
                EvidenceItem(
                    source_id=str(source["source_id"]).upper(),
                    excerpt=excerpt,
                    citation=self._format_citation(
                        document_name=document_name,
                        section_title=section_title,
                        page_number=page_number,
                    ),
                    document_name=document_name,
                    section_title=section_title,
                    page_number=page_number,
                    chunk_id=(
                        str(source.get("chunk_id"))
                        if source.get("chunk_id") is not None
                        else None
                    ),
                    rerank_score=(
                        float(source.get("rerank_score"))
                        if source.get("rerank_score") is not None
                        else None
                    ),
                )
            )
        return evidence
