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
    def __init__(self, *, relevance_threshold: float = 0.320982, strong_threshold: float = 0.533) -> None:
        self._relevance_threshold = relevance_threshold
        self._strong_threshold = strong_threshold

    def classify(self, top_score: float | None) -> EvidenceStrengthDecision:
        if top_score is None or top_score < self._relevance_threshold:
            return EvidenceStrengthDecision("insufficient", top_score, self._relevance_threshold, self._strong_threshold, "refuse_without_guessing", False, "Retrieval evidence is below the calibrated relevance gate.")
        if top_score < self._strong_threshold:
            return EvidenceStrengthDecision("moderate", top_score, self._relevance_threshold, self._strong_threshold, "qualified_evidence_language", True, "Evidence clears the answer gate but remains below the strong-evidence baseline.")
        return EvidenceStrengthDecision("strong", top_score, self._relevance_threshold, self._strong_threshold, "direct_grounded_language", True, "Evidence meets the strong-evidence baseline.")

    def classify_as_dict(self, top_score: float | None) -> dict[str, Any]:
        return asdict(self.classify(top_score))

    @staticmethod
    def prompt_instruction(level: EvidenceLevel, language: str) -> str:
        strong = {
            "ar": "استخدم لغة مباشرة مرتبطة بالمصدر، وتجنب اليقين المطلق ولا تضف أي تفصيل غير موجود في الأدلة.",
            "fr": "Utilisez une formulation directe liée à la source, évitez toute certitude absolue et n’ajoutez aucun détail absent des preuves.",
            "en": "Use direct source-bound wording, avoid absolute certainty, and add no detail absent from the evidence.",
        }
        moderate = {
            "ar": "استخدم لغة مقيدة ووضح أن المعلومات قد لا تغطي كل تفاصيل السؤال. لا تخمن.",
            "fr": "Utilisez une formulation nuancée et précisez que les informations peuvent ne pas couvrir tous les détails. Ne faites aucune supposition.",
            "en": "Use qualified wording and state that the information may not cover every detail. Do not guess.",
        }
        insufficient = {
            "ar": "ارفض الإجابة بوضوح ولا تقدم أي تخمين.",
            "fr": "Refusez clairement de répondre et ne fournissez aucune supposition.",
            "en": "Refuse clearly and do not provide any guess.",
        }
        catalog = strong if level == "strong" else moderate if level == "moderate" else insufficient
        return catalog.get(language, catalog["en"])
