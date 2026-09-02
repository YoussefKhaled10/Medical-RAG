import re
from dataclasses import dataclass

try:
    from langdetect import DetectorFactory, LangDetectException, detect
    DetectorFactory.seed = 0
except ImportError:  # pragma: no cover
    LangDetectException = Exception
    detect = None


@dataclass(frozen=True, slots=True)
class DetectedLanguage:
    code: str
    name: str
    direction: str


class LanguageDetector:
    _ARABIC = re.compile(r"[\u0600-\u06FF]")
    _SHORT_NEUTRAL = re.compile(r"^(?:ok|yes|no|thanks|thank you|continue|next|more|sure|yep|yup|تمام|شكرا|شكراً|ايوه|نعم|كمل|موافق|اوكي|أوكي|👍|🙏|❤️)$", re.IGNORECASE)
    _COMMON_ENGLISH = re.compile(r"\b(?:the|is|are|was|were|what|how|why|when|where|which|who|can|you|your|i|me|my|we|our|more|in|on|at|to|for|from|with|about|after|before|stopping|recovery|alcohol|treatment|guidance|evidence|help|symptoms|relapse|support|health|dose|doctor)\b", re.IGNORECASE)
    _NAMES = {
        "ar": "Arabic",
        "en": "English",
        "fr": "French",
        "es": "Spanish",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "nl": "Dutch",
        "tr": "Turkish",
    }

    @classmethod
    def detect(
        cls,
        text: str,
        last_explicit_user_language: str | None = None,
    ) -> DetectedLanguage:
        value = " ".join(str(text).split()).strip()
        if not value:
            code = last_explicit_user_language or "en"
            return DetectedLanguage(code, cls._NAMES.get(code, "English"), "rtl" if code == "ar" else "ltr")

        # 1. Direct Arabic character detection
        if cls._ARABIC.search(value):
            return DetectedLanguage("ar", "Arabic", "rtl")

        # 2. Short neutral message check
        clean_text = value.strip().lower()
        if cls._SHORT_NEUTRAL.match(clean_text) and last_explicit_user_language in cls._NAMES:
            code = last_explicit_user_language
            return DetectedLanguage(code, cls._NAMES[code], "rtl" if code == "ar" else "ltr")

        # 3. Common English keywords detection (fast & accurate)
        if cls._COMMON_ENGLISH.search(value):
            return DetectedLanguage("en", "English", "ltr")

        # 4. Standard text language detection fallback
        code = "en"
        if detect is not None and value:
            try:
                detected_code = detect(value)
                if detected_code in cls._NAMES:
                    code = detected_code
            except LangDetectException:
                code = "en"

        if code not in cls._NAMES:
            code = "en"

        return DetectedLanguage(code, cls._NAMES[code], "rtl" if code == "ar" else "ltr")
