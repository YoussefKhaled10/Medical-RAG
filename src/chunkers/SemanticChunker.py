import math
import re
from dataclasses import dataclass

from src.schemas.ingestion import (
    DocumentSection,
    ParsedElement,
    SemanticChunk,
)
from src.stores.llm.LLMInterface import LLMInterface

from .ChunkerInterface import ChunkerInterface


@dataclass(slots=True)
class _ChunkDraft:
    document_name: str
    section_title: str
    page_number: int
    page_end: int
    elements: list[ParsedElement]

    @property
    def text(self) -> str:
        return "\n\n".join(
            element.text.strip()
            for element in self.elements
            if element.text.strip()
        ).strip()


class SemanticChunker(ChunkerInterface):
    """Create chunks using section boundaries and adjacent semantic similarity."""

    def __init__(
        self,
        embedding_provider: LLMInterface,
        similarity_threshold: float = 0.55,
        minimum_tokens: int = 120,
        target_tokens: int = 350,
        maximum_tokens: int = 500,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if minimum_tokens <= 0:
            raise ValueError("minimum_tokens must be greater than zero")
        if target_tokens < minimum_tokens:
            raise ValueError("target_tokens must be >= minimum_tokens")
        if maximum_tokens < target_tokens:
            raise ValueError("maximum_tokens must be >= target_tokens")

        self._embedding_provider = embedding_provider
        self._similarity_threshold = similarity_threshold
        self._minimum_tokens = minimum_tokens
        self._target_tokens = target_tokens
        self._maximum_tokens = maximum_tokens

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate tokens without coupling the chunker to one tokenizer."""
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))

    @staticmethod
    def _cosine_similarity(
        vector_a: list[float],
        vector_b: list[float],
    ) -> float:
        if len(vector_a) != len(vector_b):
            raise ValueError("Embedding vectors must have the same dimension")

        dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
        norm_a = math.sqrt(sum(value * value for value in vector_a))
        norm_b = math.sqrt(sum(value * value for value in vector_b))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def _split_oversized_element(
        self,
        element: ParsedElement,
    ) -> list[ParsedElement]:
        """Split a single oversized element while retaining its source metadata."""
        if self._estimate_tokens(element.text) <= self._maximum_tokens:
            return [element]

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+|\n{2,}", element.text)
            if sentence.strip()
        ]
        if not sentences:
            sentences = [element.text]

        pieces: list[str] = []
        current_sentences: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)

            if sentence_tokens > self._maximum_tokens:
                words = sentence.split()
                for start in range(0, len(words), self._maximum_tokens):
                    word_piece = " ".join(
                        words[start : start + self._maximum_tokens]
                    ).strip()
                    if word_piece:
                        if current_sentences:
                            pieces.append(" ".join(current_sentences))
                            current_sentences = []
                            current_tokens = 0
                        pieces.append(word_piece)
                continue

            if (
                current_sentences
                and current_tokens + sentence_tokens > self._maximum_tokens
            ):
                pieces.append(" ".join(current_sentences))
                current_sentences = []
                current_tokens = 0

            current_sentences.append(sentence)
            current_tokens += sentence_tokens

        if current_sentences:
            pieces.append(" ".join(current_sentences))

        return [
            element.model_copy(
                update={
                    "element_index": (element.element_index * 1000) + piece_index,
                    "text": piece,
                    "metadata": {
                        **element.metadata,
                        "split_from_element_index": element.element_index,
                        "split_piece_index": piece_index,
                    },
                }
            )
            for piece_index, piece in enumerate(pieces, start=1)
        ]

    def _prepare_elements(
        self,
        section: DocumentSection,
    ) -> list[ParsedElement]:
        prepared: list[ParsedElement] = []

        for element in section.elements:
            if element.is_title or not element.text.strip():
                continue
            prepared.extend(self._split_oversized_element(element))

        return prepared

    async def _chunk_section(
        self,
        section: DocumentSection,
    ) -> list[_ChunkDraft]:
        elements = self._prepare_elements(section)
        if not elements:
            return []

        embeddings = await self._embedding_provider.embed_documents(
            [element.text for element in elements]
        )
        if len(embeddings) != len(elements):
            raise RuntimeError(
                "Embedding provider returned an unexpected number of vectors"
            )

        drafts: list[_ChunkDraft] = []
        current_elements: list[ParsedElement] = [elements[0]]
        current_tokens = self._estimate_tokens(elements[0].text)

        for index in range(1, len(elements)):
            next_element = elements[index]
            next_tokens = self._estimate_tokens(next_element.text)
            similarity = self._cosine_similarity(
                embeddings[index - 1],
                embeddings[index],
            )

            exceeds_maximum = (
                current_tokens + next_tokens > self._maximum_tokens
            )
            reached_target = current_tokens >= self._target_tokens
            semantic_break = similarity < self._similarity_threshold
            enough_to_close = current_tokens >= self._minimum_tokens

            should_close = exceeds_maximum or (
                semantic_break and enough_to_close
            ) or (
                reached_target
                and similarity < self._similarity_threshold + 0.10
            )

            if should_close:
                drafts.append(
                    self._build_draft(section, current_elements)
                )
                current_elements = [next_element]
                current_tokens = next_tokens
            else:
                current_elements.append(next_element)
                current_tokens += next_tokens

        if current_elements:
            drafts.append(self._build_draft(section, current_elements))

        # Avoid leaving an extremely small final chunk when it can be merged safely.
        if len(drafts) >= 2:
            last_tokens = self._estimate_tokens(drafts[-1].text)
            merged_tokens = self._estimate_tokens(
                f"{drafts[-2].text}\n\n{drafts[-1].text}"
            )
            if (
                last_tokens < self._minimum_tokens
                and merged_tokens <= self._maximum_tokens
            ):
                previous = drafts[-2]
                final = drafts[-1]
                drafts[-2] = self._build_draft(
                    section,
                    previous.elements + final.elements,
                )
                drafts.pop()

        return drafts

    @staticmethod
    def _build_draft(
        section: DocumentSection,
        elements: list[ParsedElement],
    ) -> _ChunkDraft:
        return _ChunkDraft(
            document_name=section.document_name,
            section_title=section.section_title,
            page_number=min(element.page_number for element in elements),
            page_end=max(element.page_number for element in elements),
            elements=list(elements),
        )

    async def chunk(
        self,
        sections: list[DocumentSection],
    ) -> list[SemanticChunk]:
        if not sections:
            return []

        drafts: list[_ChunkDraft] = []
        for section in sections:
            drafts.extend(await self._chunk_section(section))

        chunks = [
            SemanticChunk(
                chunk_id=f"chunk_{chunk_index:04d}",
                document_name=draft.document_name,
                section_title=draft.section_title,
                page_number=draft.page_number,
                text=draft.text,
            )
            for chunk_index, draft in enumerate(drafts, start=1)
            if draft.text
        ]

        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise RuntimeError("Duplicate chunk IDs were generated")

        return chunks
