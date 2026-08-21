import re
from dataclasses import dataclass
from typing import Literal

RefusalReason = Literal[
    "insufficient_evidence",
    "professional_care",
    "personalized_treatment",
    "urgent_help",
    "out_of_scope",
]


@dataclass(frozen=True, slots=True)
class RefusalDecision:
    reason: RefusalReason
    message: str
    requires_professional: bool
    urgent: bool


class RefusalPolicy:
    """Create plain-language, situation-specific safety guidance."""

    _ARABIC_PATTERN = re.compile(r"[\u0600-\u06FF]")
    _URGENT_TERMS = (
        "طوارئ", "فاقد الوعي", "لا يتنفس", "نزيف شديد", "تشنجات",
        "emergency", "unconscious", "not breathing", "severe bleeding", "seizure",
    )
    _DOSAGE_TERMS = (
        "جرعة", "الجرعة", "كام قرص", "كم قرص", "dose", "dosage", "how many tablets",
    )
    _PERSONAL_TERMS = (
        "ليا", "ليّ", "لحالتي", "مناسب لي", "أفضل دواء لي", "شخصيا",
        "for me", "my condition", "best medicine for me", "personally",
    )
    _PROFESSIONAL_TERMS = (
        "علاج", "تشخيص", "وصفة", "دواء", "كسر", "حالة صحية",
        "treatment", "diagnosis", "prescription", "medicine", "fracture", "medical condition",
    )

    @classmethod
    def detect_language(cls, text: str) -> str:
        return "ar" if cls._ARABIC_PATTERN.search(text) else "en"

    @staticmethod
    def marker(reason: RefusalReason) -> str:
        return f"[REFUSAL:{reason.upper()}]"

    @classmethod
    def classify(cls, question: str) -> RefusalReason:
        normalized = " ".join(question.casefold().split())
        if any(term in normalized for term in cls._URGENT_TERMS):
            return "urgent_help"
        if any(term in normalized for term in cls._DOSAGE_TERMS):
            return "professional_care"
        if any(term in normalized for term in cls._PERSONAL_TERMS):
            return "personalized_treatment"
        if any(term in normalized for term in cls._PROFESSIONAL_TERMS):
            return "professional_care"
        return "out_of_scope"

    @classmethod
    def decision(cls, question: str, *, reason: RefusalReason | None = None) -> RefusalDecision:
        selected = reason or cls.classify(question)
        language = cls.detect_language(question)
        messages = {
            "ar": {
                "insufficient_evidence": (
                    "المعلومات المتاحة لا تكفي لإجابة دقيقة. جرّب توضيح السؤال أكثر، "
                    "وإذا كان متعلقًا بصحتك أو علاجك فاستشر مختصًا مؤهلًا."
                ),
                "professional_care": (
                    "لا أستطيع تقديم جرعة أو تشخيص أو قرار علاجي آمن لهذه الحالة. "
                    "استشر طبيبًا أو صيدليًا مؤهلًا للحصول على تقييم مناسب."
                ),
                "personalized_treatment": (
                    "لا أستطيع اختيار دواء أو علاج مناسب لحالة فردية. يحتاج الاختيار "
                    "إلى طبيب يراجع التاريخ الصحي والأدوية الحالية وموانع الاستعمال."
                ),
                "urgent_help": (
                    "لا يمكن تقييم حالة عاجلة بأمان هنا. إذا كان هناك خطر فوري أو "
                    "أعراض شديدة، تواصل الآن مع خدمات الطوارئ المحلية أو توجّه إلى "
                    "أقرب قسم طوارئ."
                ),
                "out_of_scope": (
                    "لا أملك معلومات موثوقة كافية عن هذا الموضوع ضمن نطاق المساعدة "
                    "الحالي. يمكنك السؤال عن التعافي من الكحول، وإذا كان السؤال صحيًا "
                    "فاستشر المختص المناسب."
                ),
            },
            "en": {
                "insufficient_evidence": (
                    "The available information is not enough for an accurate answer. "
                    "Try clarifying the question, and consult a qualified professional "
                    "if it concerns your health or treatment."
                ),
                "professional_care": (
                    "I cannot safely provide a dosage, diagnosis, or treatment decision "
                    "for this situation. Please consult a qualified doctor or pharmacist."
                ),
                "personalized_treatment": (
                    "I cannot choose a medicine or treatment for an individual condition. "
                    "A doctor needs to review the health history, current medicines, and contraindications."
                ),
                "urgent_help": (
                    "An urgent situation cannot be assessed safely here. If there is "
                    "immediate danger or severe symptoms, contact local emergency services "
                    "now or go to the nearest emergency department."
                ),
                "out_of_scope": (
                    "I do not have enough reliable information on that topic within the "
                    "current service. You can ask about alcohol recovery, or consult the "
                    "appropriate professional if the question concerns health."
                ),
            },
        }
        return RefusalDecision(
            reason=selected,
            message=messages[language][selected],
            requires_professional=selected in {
                "professional_care", "personalized_treatment", "urgent_help"
            },
            urgent=selected == "urgent_help",
        )

    @classmethod
    def marked_message(cls, question: str, *, reason: RefusalReason) -> str:
        result = cls.decision(question, reason=reason)
        return f"{cls.marker(reason)}\n{result.message}"

    @staticmethod
    def parse_marked_answer(answer: str) -> tuple[str | None, str]:
        match = re.match(
            r"^\[REFUSAL:([A-Z_]+)\]\s*(.*)$",
            answer.strip(),
            flags=re.DOTALL,
        )
        if not match:
            return None, answer.strip()
        return match.group(1).casefold(), match.group(2).strip()
