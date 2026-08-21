from dataclasses import asdict, dataclass
from typing import Any, Literal


EvidenceLevel = Literal["insufficient", "moderate", "strong"]


@dataclass(frozen=True, slots=True)
class EvidenceStrengthDecision:
    level: EvidenceLevel
    top_score: float | None
    relevance_threshold: float
    strong_threshold: float
    language_policy: str
    answer_allowed: bool
    rationale: str


class EvidenceStrengthClassifier:
    """Convert retrieval relevance into a safe answer-language policy."""

    def __init__(
        self,
        *,
        relevance_threshold: float = 0.320982,
        strong_threshold: float = 0.533,
    ) -> None:
        if not 0.0 <= relevance_threshold <= 1.0:
            raise ValueError("relevance_threshold must be between 0 and 1")
        if not relevance_threshold < strong_threshold <= 1.0:
            raise ValueError(
                "strong_threshold must be greater than relevance_threshold"
            )
        self._relevance_threshold = relevance_threshold
        self._strong_threshold = strong_threshold

    def classify(
        self,
        top_score: float | None,
    ) -> EvidenceStrengthDecision:
        if top_score is None or top_score < self._relevance_threshold:
            return EvidenceStrengthDecision(
                level="insufficient",
                top_score=top_score,
                relevance_threshold=self._relevance_threshold,
                strong_threshold=self._strong_threshold,
                language_policy="refuse_without_guessing",
                answer_allowed=False,
                rationale=(
                    "Retrieval evidence is below the calibrated relevance gate."
                ),
            )

        if top_score < self._strong_threshold:
            return EvidenceStrengthDecision(
                level="moderate",
                top_score=top_score,
                relevance_threshold=self._relevance_threshold,
                strong_threshold=self._strong_threshold,
                language_policy="qualified_evidence_language",
                answer_allowed=True,
                rationale=(
                    "Evidence clears the answer gate but is below the "
                    "provisional strong-evidence baseline."
                ),
            )

        return EvidenceStrengthDecision(
            level="strong",
            top_score=top_score,
            relevance_threshold=self._relevance_threshold,
            strong_threshold=self._strong_threshold,
            language_policy="direct_grounded_language",
            answer_allowed=True,
            rationale=(
                "Evidence meets the provisional strong-evidence baseline."
            ),
        )

    def classify_as_dict(
        self,
        top_score: float | None,
    ) -> dict[str, Any]:
        return asdict(self.classify(top_score))

    @staticmethod
    def prompt_instruction(
        level: EvidenceLevel,
        language: Literal["ar", "en"],
    ) -> str:
        if level == "strong":
            if language == "ar":
                return (
                    "استخدم لغة مباشرة مرتبطة بالمصدر مثل: تشير الإرشادات إلى. "
                    "تجنب اليقين المطلق، ولا تضف أي تفصيل غير موجود في الأدلة."
                )
            return (
                "Use direct source-bound wording such as: The guideline states. "
                "Avoid absolute certainty and add no detail absent from evidence."
            )

        if level == "moderate":
            if language == "ar":
                return (
                    "استخدم لغة مقيدة مثل: تشير الأدلة المسترجعة إلى. وضّح أن "
                    "المصدر قد لا يجيب مباشرة عن جميع تفاصيل السؤال. لا تخمن."
                )
            return (
                "Use qualified wording such as: The retrieved evidence suggests. "
                "State that the source may not directly address every detail. "
                "Do not guess."
            )

        if language == "ar":
            return "ارفض الإجابة بوضوح ولا تقدم أي تخمين."
        return "Refuse clearly and do not provide any guess."
