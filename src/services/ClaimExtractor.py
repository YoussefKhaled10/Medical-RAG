import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ExtractedClaim:
    claim_id: str
    text: str
    cited_source_ids: tuple[str, ...]
    sentence_index: int


class ClaimExtractor:
    """Extract complete factual sentences while ignoring presentation-only text."""

    _CITATION_PATTERN = re.compile(r"\[(S\d+)\]")
    _BULLET_PATTERN = re.compile(r"^\s*(?:[-*•▪◦]|\d+[.)])\s+")
    _MARKDOWN_HEADING_PATTERN = re.compile(
        r"^\s*(?:#{1,6}\s+|\*\*[^\n]+\*\*\s*:?)\s*$"
    )
    _SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?؟؛])\s+|\n+")
    _WHITESPACE_PATTERN = re.compile(r"\s+")

    def __init__(self, *, minimum_claim_characters: int = 3) -> None:
        if minimum_claim_characters < 1:
            raise ValueError("minimum_claim_characters must be at least one")
        self._minimum_claim_characters = minimum_claim_characters

    @classmethod
    def _normalize(cls, text: str) -> str:
        text = cls._WHITESPACE_PATTERN.sub(" ", text).strip()
        return re.sub(r"\s+([.!?؟؛,:])", r"\1", text)

    @classmethod
    def _source_ids(cls, text: str) -> tuple[str, ...]:
        return tuple(dict.fromkeys(cls._CITATION_PATTERN.findall(text)))

    @classmethod
    def _remove_citations(cls, text: str) -> str:
        return cls._CITATION_PATTERN.sub("", text)

    @staticmethod
    def _is_refusal(answer: str, refusal_sentences: tuple[str, ...]) -> bool:
        normalized = " ".join(answer.split()).strip()
        return normalized in {
            " ".join(sentence.split()).strip()
            for sentence in refusal_sentences
        }

    @classmethod
    def _is_non_claim_heading(cls, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return True
        if cls._MARKDOWN_HEADING_PATTERN.match(stripped):
            return True
        # A colon-ended uncited line is a label or introduction, not a claim.
        if stripped.endswith(":") and not cls._source_ids(stripped):
            return True
        return False

    @classmethod
    def _merge_list_blocks(cls, answer: str) -> list[str]:
        """Merge a colon introduction and following bullets into one cited claim."""
        lines = [line.strip() for line in answer.splitlines() if line.strip()]
        if len(lines) <= 1:
            return cls._SENTENCE_BOUNDARY_PATTERN.split(answer)

        merged: list[str] = []
        index = 0
        while index < len(lines):
            current = lines[index]
            if current.endswith(":"):
                items: list[str] = []
                citations: list[str] = list(cls._source_ids(current))
                next_index = index + 1
                while next_index < len(lines) and cls._BULLET_PATTERN.match(lines[next_index]):
                    item = cls._BULLET_PATTERN.sub("", lines[next_index]).strip()
                    citations.extend(cls._source_ids(item))
                    clean = cls._normalize(cls._remove_citations(item)).rstrip(".!؟؛")
                    if clean:
                        items.append(clean)
                    next_index += 1
                if items:
                    intro = cls._normalize(cls._remove_citations(current)).rstrip(":")
                    suffix = "".join(f"[{sid}]" for sid in dict.fromkeys(citations))
                    merged.append(f"{intro}: {', '.join(items)} {suffix}.".strip())
                    index = next_index
                    continue
            merged.extend(cls._SENTENCE_BOUNDARY_PATTERN.split(current))
            index += 1
        return merged

    def extract(
        self,
        answer: str,
        *,
        refusal_sentences: tuple[str, ...] = (),
    ) -> list[ExtractedClaim]:
        normalized_answer = answer.strip()
        if not normalized_answer or self._is_refusal(normalized_answer, refusal_sentences):
            return []

        claims: list[ExtractedClaim] = []
        for sentence_index, raw_segment in enumerate(self._merge_list_blocks(normalized_answer)):
            segment = self._normalize(self._BULLET_PATTERN.sub("", raw_segment))
            if self._is_non_claim_heading(segment):
                continue
            source_ids = self._source_ids(segment)
            claim_text = self._normalize(self._remove_citations(segment)).strip(" -–—•")
            if len(claim_text) < self._minimum_claim_characters:
                continue
            claims.append(
                ExtractedClaim(
                    claim_id=f"C{len(claims) + 1}",
                    text=claim_text,
                    cited_source_ids=source_ids,
                    sentence_index=sentence_index,
                )
            )
        return claims

    def extract_as_dicts(self, answer: str, **kwargs: Any) -> list[dict[str, Any]]:
        return [asdict(claim) for claim in self.extract(answer, **kwargs)]
