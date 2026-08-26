from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BuiltContext:
    text: str
    sources: tuple[dict[str, Any], ...]
    total_characters: int


class ContextBuilder:
    """Build one authoritative source catalog from final reranked chunks."""

    def __init__(self, *, max_context_characters: int = 24000) -> None:
        if max_context_characters < 1000:
            raise ValueError("max_context_characters must be at least 1000")
        self._max_context_characters = max_context_characters

    @staticmethod
    def _source_block(source: dict[str, Any]) -> str:
        return "\n".join(
            (
                f"[{source['source_id']}]",
                f"Document: {source['document_name'] or 'Unknown document'}",
                f"Section: {source['section_title'] or 'Unknown section'}",
                f"Page: {source['page_number'] if source['page_number'] is not None else 'Unknown'}",
                f"Asset ID: {source['asset_id'] if source['asset_id'] is not None else 'Unknown'}",
                f"Chunk ID: {source['chunk_id'] or 'Unknown'}",
                "Content:",
                source["content"],
            )
        )

    def build(self, results: list[dict[str, Any]]) -> BuiltContext:
        sources: list[dict[str, Any]] = []
        blocks: list[str] = []
        consumed = 0

        for result in results:
            content = str(result.get("text") or "").strip()
            if not content:
                continue

            source_id = f"S{len(sources) + 1}"
            source = {
                "source_id": source_id,
                "asset_id": result.get("asset_id"),
                "project_id": result.get("project_id"),
                "chunk_id": result.get("chunk_id"),
                "document_name": result.get("document_name"),
                "section_title": result.get("section_title"),
                "page_number": result.get("page_number"),
                "rerank_score": result.get("rerank_score"),
                # This exact content is authoritative for generation and judging.
                "content": content,
                "text": content,
                "excerpt": content,
            }
            block = self._source_block(source)
            separator_size = 2 if blocks else 0
            remaining = self._max_context_characters - consumed - separator_size
            if remaining <= 0:
                break

            if len(block) > remaining:
                fixed_prefix = block[: max(0, block.rfind("Content:\n") + 9)]
                content_budget = max(0, remaining - len(fixed_prefix))
                if content_budget < 100:
                    break
                source["content"] = content[:content_budget].rstrip()
                source["text"] = source["content"]
                source["excerpt"] = source["content"]
                block = self._source_block(source)

            sources.append(source)
            blocks.append(block)
            consumed += len(block) + separator_size

        context_text = "\n\n".join(blocks)
        return BuiltContext(
            text=context_text,
            sources=tuple(sources),
            total_characters=len(context_text),
        )
