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
    _NAMES = {"ar":"Arabic","en":"English","fr":"French","es":"Spanish","de":"German","it":"Italian","pt":"Portuguese","nl":"Dutch","tr":"Turkish"}
    @classmethod
    def detect(cls, text: str) -> DetectedLanguage:
        value = " ".join(str(text).split()).strip()
        if cls._ARABIC.search(value):
            return DetectedLanguage("ar", "Arabic", "rtl")
        code = "en"
        if detect is not None and value:
            try:
                code = detect(value)
            except LangDetectException:
                code = "en"
        if code not in cls._NAMES:
            code = "en"
        return DetectedLanguage(code, cls._NAMES[code], "rtl" if code == "ar" else "ltr")
