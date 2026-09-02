import re
from dataclasses import dataclass
from typing import Literal

from src.services.LanguageDetector import LanguageDetector
from src.services.QueryUnderstandingService import QueryUnderstandingService

RefusalReason = Literal["insufficient_evidence", "out_of_scope", "professional_care", "personalized_treatment", "urgent_help", "prompt_injection"]

@dataclass(frozen=True, slots=True)
class RefusalDecision:
    reason: RefusalReason
    message: str
    requires_professional: bool
    urgent: bool

class RefusalPolicy:
    """Return contextual guidance without treating low relevance as out of scope."""

    _MESSAGES = {
        "ar": {
            "insufficient_evidence": "لم أجد في المصادر المتاحة معلومات كافية للإجابة بدقة. جرّب توضيح المقصود أو اسأل عن آثار الكحول، أعراض الانسحاب، العلاج، الانتكاس، أو الدعم المتاح.",
            "out_of_scope": "هذا السؤال خارج نطاق معلومات التعافي من الكحول المتاحة حاليًا. يمكنك سؤالي عن آثار الكحول، الانسحاب، العلاج، الوقاية من الانتكاس، أو الدعم أثناء التعافي.",
            "professional_care": "لا أستطيع تحديد جرعة أو تشخيص أو قرار علاجي لحالة فردية. استشر طبيبًا أو صيدليًا مؤهلًا للحصول على تقييم مناسب.",
            "personalized_treatment": "لا أستطيع اختيار الدواء أو العلاج الأنسب لحالة فردية. يحتاج ذلك إلى مختص يراجع التاريخ الصحي والأدوية الحالية وموانع الاستعمال.",
            "urgent_help": "قد تكون هذه حالة عاجلة. تواصل الآن مع خدمات الطوارئ المحلية أو توجّه إلى أقرب قسم طوارئ، ولا تعتمد على هذه المحادثة لتقييم الحالة.",
            "prompt_injection": "لا يمكنني تجاوز قواعد الأدلة والأمان أو اختراع مصادر. يمكنني الإجابة فقط من المعلومات المتاحة والقابلة للتحقق.",
        },
        "en": {
            "insufficient_evidence": "I could not find enough information in the available sources to answer accurately. Clarify whether you mean alcohol effects, withdrawal, treatment, relapse, or recovery support.",
            "out_of_scope": "This question is outside the currently available alcohol-recovery information. You can ask about alcohol effects, withdrawal, treatment, relapse prevention, or recovery support.",
            "professional_care": "I cannot determine an individualized dosage, diagnosis, or treatment decision. Please consult a qualified doctor or pharmacist.",
            "personalized_treatment": "I cannot choose the best medicine or treatment for an individual. A qualified professional must review medical history, current medicines, and contraindications.",
            "urgent_help": "This may be an emergency. Contact local emergency services now or go to the nearest emergency department, and do not rely on this chat to assess the situation.",
            "prompt_injection": "I cannot bypass the evidence and safety rules or invent sources. I can answer only from available verifiable information.",
        },
        "fr": {
            "insufficient_evidence": "Je n'ai pas trouvé suffisamment d'informations dans les sources disponibles pour répondre avec précision. Précisez s'il s'agit des effets de l'alcool, du sevrage, du traitement, de la rechute ou du soutien au rétablissement.",
            "out_of_scope": "Cette question dépasse les informations actuellement disponibles sur le rétablissement lié à l'alcool.",
            "professional_care": "Je ne peux pas déterminer une posologie, un diagnostic ou une décision thérapeutique personnalisée. Consultez un médecin ou un pharmacien qualifié.",
            "personalized_treatment": "Je ne peux pas choisir le meilleur médicament ou traitement pour une personne. Une évaluation professionnelle est nécessaire.",
            "urgent_help": "Il peut s'agir d'une urgence. Contactez immédiatement les services d'urgence locaux ou rendez-vous au service d'urgence le plus proche.",
            "prompt_injection": "Je ne peux pas contourner les règles de sécurité ou inventer des sources.",
        },
    }

    @classmethod
    def classify(cls, question: str, *, low_relevance: bool = False) -> RefusalReason:
        understood = QueryUnderstandingService().understand(question)
        if understood.safety_reason:
            return understood.safety_reason  # type: ignore[return-value]
        if low_relevance:
            return "insufficient_evidence" if understood.domain_related else "out_of_scope"
        return "insufficient_evidence"

    @classmethod
    def decision(
        cls,
        question: str,
        *,
        reason: RefusalReason | None = None,
        low_relevance: bool = False,
        response_language: str | None = None,
    ) -> RefusalDecision:
        selected = reason or cls.classify(question, low_relevance=low_relevance)
        language = response_language or LanguageDetector.detect(question).code
        messages = cls._MESSAGES.get(language, cls._MESSAGES["en"])
        return RefusalDecision(
            selected,
            messages.get(selected, messages["insufficient_evidence"]),
            selected in {"professional_care", "personalized_treatment", "urgent_help"},
            selected == "urgent_help",
        )

    @staticmethod
    def marker(reason: RefusalReason) -> str:
        return f"[REFUSAL:{reason.upper()}]"

    @classmethod
    def marked_message(
        cls,
        question: str,
        *,
        reason: RefusalReason,
        response_language: str | None = None,
    ) -> str:
        return f"{cls.marker(reason)}\n{cls.decision(question, reason=reason, response_language=response_language).message}"


    @staticmethod
    def parse_marked_answer(answer: str) -> tuple[str | None, str]:
        match = re.match(r"^\[REFUSAL:([A-Z_]+)\]\s*(.*)$", str(answer).strip(), flags=re.DOTALL)
        return (match.group(1).casefold(), match.group(2).strip()) if match else (None, str(answer).strip())
