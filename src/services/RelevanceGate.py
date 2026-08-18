from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RelevanceDecision:
    passed: bool
    reason: str
    top_score: float | None
    second_score: float | None
    score_margin: float | None
    threshold: float
    qualified_chunk_count: int
    minimum_qualified_chunks: int


class RelevanceGate:
    """Decide whether retrieval evidence is sufficient for generation."""

    def __init__(
        self,
        threshold: float,
        minimum_qualified_chunks: int = 1,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        if minimum_qualified_chunks < 1:
            raise ValueError(
                "minimum_qualified_chunks must be at least one"
            )
        self._threshold = threshold
        self._minimum_qualified_chunks = minimum_qualified_chunks

    @staticmethod
    def _scores(results: list[dict[str, Any]]) -> list[float]:
        return sorted(
            [
                float(result["rerank_score"])
                for result in results
                if result.get("rerank_score") is not None
            ],
            reverse=True,
        )

    def evaluate(
        self,
        results: list[dict[str, Any]],
    ) -> RelevanceDecision:
        scores = self._scores(results)
        if not scores:
            return RelevanceDecision(
                passed=False,
                reason="no_rerank_scores",
                top_score=None,
                second_score=None,
                score_margin=None,
                threshold=self._threshold,
                qualified_chunk_count=0,
                minimum_qualified_chunks=self._minimum_qualified_chunks,
            )

        top_score = scores[0]
        second_score = scores[1] if len(scores) > 1 else None
        qualified_count = sum(
            score >= self._threshold for score in scores
        )
        passed = qualified_count >= self._minimum_qualified_chunks

        return RelevanceDecision(
            passed=passed,
            reason=(
                "sufficient_retrieval_evidence"
                if passed
                else "low_retrieval_relevance"
            ),
            top_score=round(top_score, 6),
            second_score=(
                round(second_score, 6)
                if second_score is not None
                else None
            ),
            score_margin=(
                round(top_score - second_score, 6)
                if second_score is not None
                else None
            ),
            threshold=self._threshold,
            qualified_chunk_count=qualified_count,
            minimum_qualified_chunks=self._minimum_qualified_chunks,
        )

    def evaluate_as_dict(
        self,
        results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return asdict(self.evaluate(results))
