import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RewrittenQuery:
    original_query: str
    semantic_query: str
    keyword_query: str
    expansions: tuple[str, ...]


class QueryRewriter:
    """Deterministic medical query expansion for retrieval.

    The original query is preserved. Semantic search receives a natural-language
    clarification, while keyword search receives an OR-oriented expansion query.
    """

    _PHRASE_EXPANSIONS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
        (
            re.compile(r"\b(?:medicine|medicines|medication|medications|drugs?)\b", re.I),
            ("pharmacological interventions", "medication"),
        ),
        (
            re.compile(r"\b(?:relapse|recurrence)\b", re.I),
            ("prevent relapse", "relapse prevention"),
        ),
        (
            re.compile(r"\b(?:mean|meaning|definition|define)\b", re.I),
            ("definition", "definitions of terms"),
        ),
        (
            re.compile(r"\b(?:questionnaire|questionnaires|screening tool|screening tools)\b", re.I),
            ("validated alcohol questionnaire", "AUDIT", "AUDIT-C", "FAST", "PAT"),
        ),
        (
            re.compile(r"\b(?:community groups?|peer groups?|self help|self-help)\b", re.I),
            ("community support networks", "self-help groups"),
        ),
        (
            re.compile(r"\b(?:triage)\b", re.I),
            ("brief triage assessment", "treatment needs", "associated risks"),
        ),
        (
            re.compile(r"\b(?:assessed|assessment|monitored|monitoring)\b", re.I),
            ("assessment and monitoring", "locally specified protocols"),
        ),
        (
            re.compile(r"\b(?:quality measures?)\b", re.I),
            ("quality measures", "process", "structure", "outcome"),
        ),
        (
            re.compile(r"\bciwa\s*[-–]?\s*ar\b", re.I),
            (
                "CIWA-Ar",
                "CIWA Ar",
                "Clinical Institute Withdrawal Assessment Alcohol revised",
            ),
        ),
        (
            re.compile(r"\b(?:psychological treatment|psychological treatments|therapy|therapies)\b", re.I),
            ("psychological interventions", "cognitive behavioural therapies"),
        ),
        (
            re.compile(r"\b(?:accessible|accessibility)\b", re.I),
            ("accessible information", "format that suits their needs"),
        ),
    )

    _STOPWORDS = {
        "a",
        "after",
        "an",
        "and",
        "are",
        "available",
        "be",
        "before",
        "can",
        "do",
        "does",
        "for",
        "help",
        "how",
        "in",
        "is",
        "of",
        "or",
        "should",
        "the",
        "to",
        "used",
        "what",
        "when",
        "which",
        "who",
        "with",
    }

    @classmethod
    def _extract_terms(cls, query: str) -> list[str]:
        tokens = re.findall(r"[A-Za-z0-9]+(?:[-–][A-Za-z0-9]+)*", query)
        return [
            token
            for token in tokens
            if token.lower() not in cls._STOPWORDS and len(token) > 1
        ]

    @classmethod
    def rewrite(cls, query: str) -> RewrittenQuery:
        original = " ".join(query.split()).strip()
        if not original:
            raise ValueError("query must not be empty")

        expansions: list[str] = []
        for pattern, values in cls._PHRASE_EXPANSIONS:
            if pattern.search(original):
                expansions.extend(values)

        unique_expansions = tuple(dict.fromkeys(expansions))
        semantic_query = original
        if unique_expansions:
            semantic_query = (
                f"{original} Relevant medical concepts: "
                + "; ".join(unique_expansions)
            )

        keyword_terms = cls._extract_terms(original)
        keyword_terms.extend(unique_expansions)
        keyword_terms = list(dict.fromkeys(keyword_terms))

        # websearch_to_tsquery interprets OR explicitly and supports quoted phrases.
        keyword_query = " OR ".join(
            f'"{term}"' if " " in term else term
            for term in keyword_terms
        ) or original

        return RewrittenQuery(
            original_query=original,
            semantic_query=semantic_query,
            keyword_query=keyword_query,
            expansions=unique_expansions,
        )
