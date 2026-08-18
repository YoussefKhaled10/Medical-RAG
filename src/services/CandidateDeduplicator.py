import re
from typing import Any


class CandidateDeduplicator:
    """Remove exact and near-duplicate retrieval candidates."""

    def __init__(
        self,
        token_jaccard_threshold: float = 0.90,
        minimum_tokens_for_similarity: int = 20,
    ) -> None:
        if not 0.0 < token_jaccard_threshold <= 1.0:
            raise ValueError("token_jaccard_threshold must be in (0, 1]")
        self._threshold = token_jaccard_threshold
        self._minimum_tokens = minimum_tokens_for_similarity

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return " ".join(text.split())

    @classmethod
    def _tokens(cls, candidate: dict[str, Any]) -> set[str]:
        combined = (
            f"{candidate.get('section_title', '')} "
            f"{candidate.get('text', '')}"
        )
        return set(cls._normalize(combined).split())

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        union = left | right
        return len(left & right) / len(union) if union else 1.0

    def deduplicate(
        self,
        candidates: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        kept: list[dict[str, Any]] = []
        kept_tokens: list[set[str]] = []
        removed: list[dict[str, Any]] = []
        seen_keys: set[tuple[int | None, str | None]] = set()
        seen_normalized: set[str] = set()

        for candidate in candidates:
            key = (candidate.get("asset_id"), candidate.get("chunk_id"))
            normalized = self._normalize(str(candidate.get("text") or ""))

            if key in seen_keys or normalized in seen_normalized:
                removed.append({
                    "chunk_id": candidate.get("chunk_id"),
                    "reason": "exact_duplicate",
                })
                continue

            tokens = self._tokens(candidate)
            duplicate_of = None
            similarity = 0.0
            if len(tokens) >= self._minimum_tokens:
                for index, existing_tokens in enumerate(kept_tokens):
                    if len(existing_tokens) < self._minimum_tokens:
                        continue
                    score = self._jaccard(tokens, existing_tokens)
                    if score >= self._threshold:
                        duplicate_of = kept[index].get("chunk_id")
                        similarity = score
                        break

            if duplicate_of is not None:
                removed.append({
                    "chunk_id": candidate.get("chunk_id"),
                    "reason": "near_duplicate",
                    "duplicate_of": duplicate_of,
                    "similarity": round(similarity, 6),
                })
                continue

            seen_keys.add(key)
            seen_normalized.add(normalized)
            kept.append(candidate)
            kept_tokens.append(tokens)

        for rank, candidate in enumerate(kept, start=1):
            candidate["rank"] = rank
        return kept, removed
