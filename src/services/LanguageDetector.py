import re
from dataclasses import dataclass

from langdetect import DetectorFactory, LangDetectException, detect


DetectorFactory.seed = 0


@dataclass(frozen=True, slots=True)
class DetectedLanguage:
    code: str
    name: str
    direction: str


class LanguageDetector:
    """Detect the language of a natural user question consistently."""

    _ARABIC_PATTERN = re.compile(r"[\u0600-\u06FF]")
    _FRENCH_MARKERS = {
        "alcool", "alcoolique", "sevrage", "symptômes", "symptomes",
        "quels", "quelles", "quel", "quelle", "traitement",
        "médicament", "medicament", "médicaments", "medicaments",
        "après", "apres", "personne", "évaluation", "evaluation",
        "dose", "posologie",
    }
    _NAMES = {
        "ar": "Arabic", "en": "English", "fr": "French",
        "es": "Spanish", "de": "German", "it": "Italian",
        "pt": "Portuguese", "nl": "Dutch", "tr": "Turkish",
    }

    @classmethod
    def detect(cls, text: str) -> DetectedLanguage:
        normalized = " ".join(str(text).casefold().split())
        if not normalized:
            return DetectedLanguage("en", "English", "ltr")

        if cls._ARABIC_PATTERN.search(normalized):
            return DetectedLanguage("ar", "Arabic", "rtl")

        words = set(re.findall(r"[a-zàâçéèêëîïôûùüÿœæ]+", normalized))
        if words.intersection(cls._FRENCH_MARKERS):
            return DetectedLanguage("fr", "French", "ltr")

        try:
            code = detect(normalized)
        except LangDetectException:
            code = "en"

        if code not in cls._NAMES:
            code = "en"
        return DetectedLanguage(code, cls._NAMES[code], "rtl" if code == "ar" else "ltr")
