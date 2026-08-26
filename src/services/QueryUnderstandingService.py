import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Literal


QueryIntent = Literal[
    "health_effects",
    "withdrawal",
    "relapse",
    "treatment",
    "medication_information",
    "support_groups",
    "screening",
    "prevention",
    "definition",
    "ambiguous_alcohol_symptoms",
    "urgent_help",
    "professional_care",
    "personalized_treatment",
    "prompt_injection",
    "out_of_scope",
    "general_alcohol_information",
]


@dataclass(frozen=True, slots=True)
class QueryUnderstanding:
    original_question: str
    normalized_question: str
    semantic_query: str
    keyword_hints: tuple[str, ...]
    intent: QueryIntent
    domain_related: bool
    ambiguous: bool
    clarification_message: str | None
    safety_reason: str | None
    detected_style: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class QueryUnderstandingService:
    """Understand common formal, colloquial, misspelled, and Arabizi questions."""

    _ARABIC = re.compile(r"[\u0600-\u06ff]")
    _ARABIZI = re.compile(
        r"\b(?:5amra|khamra|khmra|shorb|sharab|yeshrab|byeshrab|"
        r"yebatal|ybtl|ens7ab|se7b|edman|idman|sobriety|sober)\b",
        re.I,
    )

    _REPLACEMENTS = (
        (
            r"(?:انا\s+)?(?:عاوز|عايز|نفسي|محتاج)\s+"
            r"(?:ا?بطل|اوقف|اتوقف|اتعالج|أتعالج)(?:\s+عن|\s+من)?\s*"
            r"(?:ادمان|الادمان|الكحول|الخمرة|الخمرا|الشرب)?",
            "أريد المساعدة في علاج اضطراب استخدام الكحول والتوقف عنه",
        ),
        (
            r"(?:ساعدني|ساعدونى|ساعدوني|محتاج\s+مساعدة)\s*"
            r"(?:عشان|علشان|كي)?\s*(?:ا?بطل|اوقف|اتعالج|أتعالج)?",
            "أريد المساعدة في علاج اضطراب استخدام الكحول والتوقف عنه",
        ),
        (
            r"(?:ازاي|إزاي|كيف)\s+(?:ا?بطل|اوقف|اتوقف|اتعالج|أتعالج)"
            r"(?:\s+عن|\s+من)?\s*"
            r"(?:ادمان|الادمان|الكحول|الخمرة|الخمرا|الشرب)?",
            "ما خيارات علاج اضطراب استخدام الكحول والتوقف عنه",
        ),
        (
            r"(?:مش\s+عاوز|مش\s+عايز)\s+ا?شرب\s+(?:تاني|تانى)",
            "أريد التوقف عن استخدام الكحول وبدء التعافي",
        ),
        (
            r"(?:ابدأ|ابدا|أبدأ)\s+(?:تعافي|التعافي|رحلة تعافي)",
            "بدء التعافي من اضطراب استخدام الكحول",
        ),
        (
            r"(?:اتخلص من|الخلاص من|اعالج|اتعالج من)\s+"
            r"(?:ادمان|الادمان|ادمان الكحول|الكحول|الخمرة|الشرب)",
            "علاج اضطراب استخدام الكحول ودعم التعافي",
        ),
        (r"(?:الخمرة|الخمرا|خمرة|خمرا)", "الكحول"),
        (r"(?:بيشرب|بيشربوا|يشرب|بيسكر|يسكر)", "يستخدم الكحول"),
        (
            r"(?:شرب كتير|بيشرب كتير|افراط في الشرب|إفراط في الشرب)",
            "الإفراط في استخدام الكحول",
        ),
        (
            r"(?:بطل شرب|يبطل شرب|بعد ما يبطل|وقف شرب|يوقف شرب)",
            "التوقف عن استخدام الكحول",
        ),
        (
            r"(?:يرجع يشرب|رجع يشرب|بيشرب تاني|يشرب تاني)",
            "الانتكاس والعودة إلى استخدام الكحول",
        ),
        (
            r"(?:نفسه يشرب|عايز يشرب|محتاج يشرب)",
            "الرغبة الشديدة في استخدام الكحول",
        ),
        (r"(?:بيترعش|بيرتعش|رعشة|رعشه)", "الرعشة"),
        (
            r"(?:مش واعي|مش في وعيه|مغمى عليه|اغمى عليه)",
            "فقدان الوعي",
        ),
        (
            r"(?:اضرار الشرب|أضرار الشرب|اضرار الكحول|أضرار الكحول)",
            "الآثار الصحية لاستخدام الكحول",
        ),
        (
            r"(?:دواء يبطل|دوا يبطل|حاجة تبطله|حاجه تبطله)",
            "خيارات علاج اضطراب استخدام الكحول",
        ),
        (r"(?:ايه|إيه|اية)", "ما"),
        (r"(?:ازاي|إزاي)", "كيف"),
        (r"(?:ليه)", "لماذا"),
        (r"(?:عاوز|عايز)", "أريد"),
        (r"(?:قولي|قولى|قول لي)", "اشرح"),
    )

    _ARABIZI_REPLACEMENTS = {
        "5amra": "alcohol",
        "khamra": "alcohol",
        "khmra": "alcohol",
        "shorb": "drinking",
        "sharab": "drinking",
        "yeshrab": "alcohol use",
        "byeshrab": "alcohol use",
        "yebatal": "stop alcohol use",
        "ybtl": "stop alcohol use",
        "ens7ab": "alcohol withdrawal",
        "se7b": "alcohol withdrawal",
        "edman": "addiction",
        "idman": "addiction",
    }

    _DOMAIN = (
        "كحول", "الشرب", "شرب", "انسحاب", "تعافي", "انتكاس",
        "ادمان", "الإدمان", "سكر", "ثمل", "يبطل", "ابطل", "أبطل",
        "اتعالج", "التوقف", "alcohol", "drinking", "withdrawal",
        "recovery", "relapse", "addiction", "alcoholic", "sober",
        "sobriety", "مجموعات الدعم", "مجموعة دعم", "aa", "smart recovery",
        "audit", "audit-c", "fast", "alcool", "sevrage", "rechute",
    )
    _URGENT = (
        "فقدان الوعي", "مش واعي", "مغمى عليه", "لا يتنفس",
        "مش بيتنفس", "صعوبة التنفس", "تشنجات", "نزيف شديد",
        "هذيان شديد", "ارتباك شديد", "emergency", "unconscious",
        "not breathing", "difficulty breathing", "seizure", "convulsions",
    )
    _DOSAGE = (
        "جرعة", "كام قرص", "كم قرص", "قد ايه من الدواء", "مقدار الدواء",
        "ازود الجرعة", "أزود الجرعة", "ابدأ الدوا", "اوقف الدوا",
        "dose", "dosage", "how many tablets", "increase the dose",
        "start the medicine", "stop the medicine", "posologie",
    )
    _PERSONAL = (
        "انهي دواء ليا", "أنهي دواء ليا", "انهي دوا ليا", "أنهي دوا ليا",
        "انهي دوا احسن ليا", "أنهي دوا أحسن ليا", "انهي دواء احسن ليا",
        "أفضل دواء ليا", "افضل دوا ليا", "لحالتي", "مناسب ليا",
        "اختارلي علاج", "اختار لي علاج", "best medicine for me",
        "for my condition", "choose a treatment for me", "pour moi",
    )
    _INJECTION = (
        "تجاهل التعليمات", "بدون مصادر", "من غير مصادر", "اخترع مصادر",
        "جاوب من دماغك", "ignore instructions", "ignore the evidence",
        "without citations", "invent citations", "reveal system prompt",
        "bypass safety",
    )

    @staticmethod
    def _strip_diacritics(text: str) -> str:
        return "".join(
            character
            for character in unicodedata.normalize("NFKD", text)
            if not unicodedata.combining(character)
        )

    @classmethod
    def normalize(cls, question: str) -> tuple[str, str]:
        text = " ".join(str(question).strip().split())
        text = cls._strip_diacritics(text)
        style = "formal"

        if cls._ARABIZI.search(text):
            style = "arabizi"
            words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
            text = " ".join(
                cls._ARABIZI_REPLACEMENTS.get(word.casefold(), word)
                for word in words
            )
        elif cls._ARABIC.search(text):
            style = "arabic_natural"
            for pattern, replacement in cls._REPLACEMENTS:
                updated = re.sub(pattern, replacement, text, flags=re.I)
                if updated != text:
                    style = "arabic_colloquial"
                    text = updated

        return " ".join(text.split()).strip(), style

    @staticmethod
    def _contains(text: str, terms: tuple[str, ...]) -> bool:
        lowered = text.casefold()
        return any(term.casefold() in lowered for term in terms)

    @classmethod
    def understand(cls, question: str) -> QueryUnderstanding:
        original = " ".join(str(question).split()).strip()
        if not original:
            raise ValueError("question must not be empty")

        normalized, style = cls.normalize(original)
        combined = f"{original} {normalized}".casefold()
        domain_related = cls._contains(combined, cls._DOMAIN)

        if cls._contains(combined, cls._URGENT):
            intent: QueryIntent = "urgent_help"
            safety_reason = "urgent_help"
        elif cls._contains(combined, cls._DOSAGE):
            intent = "professional_care"
            safety_reason = "professional_care"
        elif cls._contains(combined, cls._PERSONAL):
            intent = "personalized_treatment"
            safety_reason = "personalized_treatment"
        elif cls._contains(combined, cls._INJECTION):
            intent = "prompt_injection"
            safety_reason = "prompt_injection"
        else:
            safety_reason = None
            if not domain_related:
                intent = "out_of_scope"
            elif cls._contains(
                combined,
                (
                    "بعد التوقف", "بعد ما", "بيتعب", "اعراض الانسحاب",
                    "أعراض الانسحاب", "انسحاب", "withdrawal symptoms",
                    "after stopping", "sevrage",
                ),
            ):
                intent = "withdrawal"
            elif cls._contains(
                combined,
                (
                    "انتكاس", "يرجع", "الرغبة الشديدة", "relapse",
                    "craving", "rechute",
                ),
            ):
                intent = "relapse"
            elif cls._contains(
                combined,
                (
                    "اضرار", "أضرار", "اثار", "آثار", "الجسم", "الكبد",
                    "القلب", "سرطان", "يحصله", "يحصل له", "bt3ml eh",
                    "health effects", "harm",
                ),
            ):
                intent = "health_effects"
            elif cls._contains(
                combined,
                (
                    "علاج", "يبطل", "ابطل", "أبطل", "اوقف", "اتوقف",
                    "اتعالج", "أتعالج", "اعالج", "بدء التعافي", "مساعدة في علاج",
                    "therapy", "treatment", "stop drinking", "quit alcohol",
                    "help me stop", "get sober", "recovery help",
                ),
            ):
                intent = "treatment"
            elif cls._contains(
                combined,
                ("دواء", "دوا", "ادوية", "أدوية", "medication", "medicine"),
            ):
                intent = "medication_information"
            elif cls._contains(
                combined,
                ("مجموعة دعم", "مجموعات الدعم", "aa", "smart recovery", "support group"),
            ):
                intent = "support_groups"
            elif cls._contains(
                combined,
                ("اختبار", "استبيان", "audit", "screening"),
            ):
                intent = "screening"
            elif cls._contains(
                combined,
                ("وقاية", "منع", "prevention", "underage"),
            ):
                intent = "prevention"
            elif cls._contains(
                combined,
                ("ما هو", "يعني ايه", "تعريف", "what is", "define"),
            ):
                intent = "definition"
            elif cls._contains(
                combined,
                ("اعراض الشرب", "أعراض الشرب", "اعراض الكحول", "أعراض الكحول", "symptoms of drinking"),
            ):
                intent = "ambiguous_alcohol_symptoms"
            else:
                intent = "general_alcohol_information"

        ambiguous = intent == "ambiguous_alcohol_symptoms"
        clarification_message = None
        if ambiguous:
            clarification_message = (
                "هل تقصد أعراض انسحاب الكحول بعد التوقف، أم الآثار الصحية "
                "لشرب الكحول، أم علامات التسمم بالكحول؟"
            )

        templates: dict[str, tuple[str, tuple[str, ...]]] = {
            "health_effects": (
                "ما الآثار الصحية قصيرة وطويلة المدى لاستخدام الكحول على الجسم؟",
                ("alcohol health effects", "short-term effects", "long-term effects", "excessive alcohol use"),
            ),
            "withdrawal": (
                "ما أعراض انسحاب الكحول بعد التوقف عن استخدامه وما المضاعفات الخطيرة؟",
                ("alcohol withdrawal symptoms", "acute alcohol withdrawal", "withdrawal complications"),
            ),
            "relapse": (
                "ما أسباب الانتكاس والعودة إلى استخدام الكحول وما وسائل الوقاية منها؟",
                ("alcohol relapse", "relapse prevention", "craving", "recovery support"),
            ),
            "treatment": (
                "ما خيارات علاج اضطراب استخدام الكحول، والتدخلات النفسية والاجتماعية، والدعم المتاح للتوقف والتعافي؟",
                (
                    "alcohol use disorder treatment",
                    "stopping alcohol use",
                    "psychosocial interventions",
                    "behavioral treatment",
                    "recovery support",
                    "relapse prevention",
                ),
            ),
            "medication_information": (
                "ما المعلومات العامة التي تذكرها المصادر عن أدوية علاج اضطراب استخدام الكحول؟",
                ("alcohol use disorder medication", "pharmacological interventions"),
            ),
            "support_groups": (
                "ما دور مجموعات الدعم المتبادل في التعافي من اضطراب استخدام الكحول؟",
                ("mutual help groups", "Alcoholics Anonymous", "SMART Recovery", "social support"),
            ),
            "screening": (
                "ما أدوات الفحص والتقييم المستخدمة لاضطراب استخدام الكحول؟",
                ("alcohol screening", "AUDIT", "AUDIT-C", "FAST"),
            ),
            "prevention": (
                "ما استراتيجيات الوقاية من أضرار استخدام الكحول؟",
                ("alcohol prevention", "community strategies", "harm reduction"),
            ),
            "definition": (
                normalized,
                ("alcohol use disorder definition",),
            ),
            "general_alcohol_information": (
                normalized,
                ("alcohol use", "alcohol recovery"),
            ),
            "ambiguous_alcohol_symptoms": (
                normalized,
                ("alcohol symptoms",),
            ),
        }
        semantic_query, keyword_hints = templates.get(
            intent,
            (normalized, tuple()),
        )

        return QueryUnderstanding(
            original_question=original,
            normalized_question=normalized,
            semantic_query=semantic_query,
            keyword_hints=tuple(keyword_hints),
            intent=intent,
            domain_related=domain_related,
            ambiguous=ambiguous,
            clarification_message=clarification_message,
            safety_reason=safety_reason,
            detected_style=style,
        )
