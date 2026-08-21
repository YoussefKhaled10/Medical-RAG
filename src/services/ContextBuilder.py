from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True, slots=True)
class BuiltContext:
    text: str
    sources: tuple[dict[str, Any], ...]
    total_characters: int


class ContextBuilder:
    """Build citation-ready context from final reranked chunks."""

    def __init__(self, max_context_characters: int = 24000) -> None:
        if max_context_characters < 1000:
            raise ValueError(
                "max_context_characters must be at least 1000"
            )
        self._max_context_characters = max_context_characters

    def build(self, results: list[dict[str, Any]]) -> BuiltContext:
        blocks: list[str] = []
        sources: list[dict[str, Any]] = []
        used_characters = 0

        for index, result in enumerate(results, start=1):
            text = str(result.get("text") or "").strip()
            if not text:
                continue

            source_id = f"S{index}"
            source = {
                "source_id": source_id,
                "document_name": result.get("document_name"),
                "section_title": result.get("section_title"),
                "page_number": result.get("page_number"),
                "chunk_id": result.get("chunk_id"),
                "rerank_score": result.get("rerank_score"),
            }
            block = (
                f"[{source_id}]\n"
                f"Document: {source['document_name']}\n"
                f"Section: {source['section_title']}\n"
                f"Page: {source['page_number']}\n"
                f"Chunk ID: {source['chunk_id']}\n"
                f"Content:\n{text}"
            )

            separator_length = 2 if blocks else 0
            remaining = (
                self._max_context_characters
                - used_characters
                - separator_length
            )
            if remaining < 300:
                break
            if len(block) > remaining:
                block = block[:remaining].rstrip()

            blocks.append(block)
            sources.append(source)
            used_characters += len(block) + separator_length

        return BuiltContext(
            text="\n\n".join(blocks),
            sources=tuple(sources),
            total_characters=used_characters,
        )
