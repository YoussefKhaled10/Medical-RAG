import re
from dataclasses import dataclass
from typing import Literal

from src.services.LanguageDetector import LanguageDetector


RefusalReason = Literal[
    "insufficient_evidence",
    "out_of_scope",
    "professional_care",
    "personalized_treatment",
    "urgent_help",
    "prompt_injection",
]


@dataclass(frozen=True, slots=True)
class RefusalDecision:
    reason: RefusalReason
    message: str
    requires_professional: bool
    urgent: bool


class RefusalPolicy:
    """Classify refusal needs and return plain-language guidance."""

    _URGENT_TERMS = (
        "طوارئ", "فاقد الوعي", "فقد الوعي", "لا يتنفس", "نزيف شديد",
        "تشنجات", "emergency", "unconscious", "not breathing",
        "severe bleeding", "seizure", "urgence", "inconscient",
        "ne respire pas", "convulsions",
    )
    _DOSAGE_TERMS = (
        "جرعة", "الجرعة", "كام قرص", "كم قرص", "تشخيص", "وصفة",
        "dose", "dosage", "diagnosis", "prescription", "how many tablets",
        "posologie", "diagnostic", "ordonnance",
    )
    _PERSONAL_TERMS = (
        "ليا", "ليّ", "لحالتي", "مناسب لي", "أفضل دواء لي", "شخصيا",
        "for me", "my condition", "best medicine for me", "personally",
        "pour moi", "mon cas", "meilleur médicament pour moi",
    )
    _INJECTION_TERMS = (
        "تجاهل التعليمات", "تجاهل قواعد", "بدون مصادر", "اخترع مصدر",
        "ignore previous instructions", "ignore the evidence", "without sources",
        "invent citations", "reveal hidden instructions", "bypass safety",
        "ignore les instructions", "sans sources", "invente des citations",
    )
    _HEALTH_TERMS = (
        "علاج", "تشخيص", "دواء", "كسر", "حالة صحية", "جرعة",
        "treatment", "diagnosis", "medicine", "fracture", "medical",
        "traitement", "diagnostic", "médicament", "fracture", "santé",
    )

    _MESSAGES = {
        "ar": {
            "insufficient_evidence": "المعلومات المتاحة لا تكفي لتقديم إجابة دقيقة. جرّب توضيح السؤال أو إعادة صياغته، وإذا كان متعلقًا بصحتك أو علاجك فاستشر مختصًا مؤهلًا.",
            "out_of_scope": "هذا الموضوع خارج نطاق المساعدة الحالية. يركز RecoveryPath AI على معلومات التعافي من الكحول. يمكنك طرح سؤال متعلق بالتعافي، وإذا كان السؤال صحيًا فاستشر المختص المناسب.",
            "professional_care": "لا أستطيع تقديم جرعة أو تشخيص أو قرار علاجي آمن لحالة فردية. استشر طبيبًا أو صيدليًا مؤهلًا للحصول على تقييم مناسب.",
            "personalized_treatment": "لا أستطيع اختيار دواء أو علاج مناسب لحالة فردية. يتطلب ذلك تقييمًا من طبيب مؤهل يراجع التاريخ الصحي والأدوية الحالية وموانع الاستعمال.",
            "urgent_help": "لا يمكن تقييم حالة عاجلة بأمان هنا. إذا كان هناك خطر فوري أو أعراض شديدة، تواصل الآن مع خدمات الطوارئ المحلية أو توجّه إلى أقرب قسم طوارئ.",
            "prompt_injection": "لا يمكنني تنفيذ طلب يتجاوز قواعد الأدلة والأمان. يجب أن تظل الإجابة مرتبطة بالمعلومات المتاحة وقابلة للتحقق.",
        },
        "en": {
            "insufficient_evidence": "The available information is not sufficient for an accurate answer. Try clarifying or rephrasing the question, and consult a qualified professional if it concerns your health or treatment.",
            "out_of_scope": "This topic is outside the current area of assistance. RecoveryPath AI focuses on alcohol-recovery information. You can ask a recovery-related question, or consult an appropriate professional if the question concerns health.",
            "professional_care": "I cannot safely provide an individualized dosage, diagnosis, or treatment decision. Please consult a qualified doctor or pharmacist for an appropriate assessment.",
            "personalized_treatment": "I cannot select a medicine or treatment for an individual condition. A qualified doctor needs to review the medical history, current medicines, and contraindications.",
            "urgent_help": "This system cannot safely assess an emergency. If there is immediate danger or severe symptoms, contact local emergency services now or go to the nearest emergency department.",
            "prompt_injection": "I cannot follow a request that bypasses the evidence and safety rules. Answers must remain grounded in verifiable information.",
        },
        "fr": {
            "insufficient_evidence": "Les informations disponibles ne permettent pas de fournir une réponse suffisamment précise. Vous pouvez reformuler la question ou consulter un professionnel qualifié si elle concerne votre santé ou votre traitement.",
            "out_of_scope": "Ce sujet ne relève pas du domaine d’assistance actuel. RecoveryPath AI se concentre sur les informations relatives au rétablissement après une consommation d’alcool. Vous pouvez poser une question liée au rétablissement ou consulter un professionnel approprié si elle concerne la santé.",
            "professional_care": "Je ne peux pas fournir en toute sécurité une posologie, un diagnostic ou une décision thérapeutique personnalisée. Veuillez consulter un médecin ou un pharmacien qualifié.",
            "personalized_treatment": "Je ne peux pas choisir un médicament ou un traitement pour une situation individuelle. Un médecin qualifié doit examiner les antécédents médicaux, les médicaments actuels et les contre-indications.",
            "urgent_help": "Ce système ne peut pas évaluer une situation urgente en toute sécurité. En cas de danger immédiat ou de symptômes graves, contactez les services d’urgence locaux ou rendez-vous au service d’urgence le plus proche.",
            "prompt_injection": "Je ne peux pas suivre une demande qui contourne les règles de sécurité et de vérification. Les réponses doivent rester fondées sur des informations vérifiables.",
        },
    }

    @classmethod
    def classify(cls, question: str, *, low_relevance: bool = False) -> RefusalReason:
        text = " ".join(str(question).casefold().split())
        if any(term in text for term in cls._URGENT_TERMS):
            return "urgent_help"
        if any(term in text for term in cls._DOSAGE_TERMS):
            return "professional_care"
        if any(term in text for term in cls._PERSONAL_TERMS):
            return "personalized_treatment"
        if any(term in text for term in cls._INJECTION_TERMS):
            return "prompt_injection"
        if low_relevance:
            return "out_of_scope"
        return "insufficient_evidence"

    @classmethod
    def decision(cls, question: str, *, reason: RefusalReason | None = None, low_relevance: bool = False) -> RefusalDecision:
        selected = reason or cls.classify(question, low_relevance=low_relevance)
        language = LanguageDetector.detect(question).code
        messages = cls._MESSAGES.get(language, cls._MESSAGES["en"])
        return RefusalDecision(
            reason=selected,
            message=messages[selected],
            requires_professional=selected in {"professional_care", "personalized_treatment", "urgent_help"},
            urgent=selected == "urgent_help",
        )

    @staticmethod
    def marker(reason: RefusalReason) -> str:
        return f"[REFUSAL:{reason.upper()}]"

    @classmethod
    def marked_message(cls, question: str, *, reason: RefusalReason) -> str:
        return f"{cls.marker(reason)}\n{cls.decision(question, reason=reason).message}"

    @staticmethod
    def parse_marked_answer(answer: str) -> tuple[str | None, str]:
        match = re.match(r"^\[REFUSAL:([A-Z_]+)\]\s*(.*)$", str(answer).strip(), flags=re.DOTALL)
        return (match.group(1).casefold(), match.group(2).strip()) if match else (None, str(answer).strip())
