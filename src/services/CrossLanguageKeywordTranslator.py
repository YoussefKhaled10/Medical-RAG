import re

from src.stores.llm.GenerationInterface import GenerationInterface


class CrossLanguageKeywordTranslator:
    """Translate Arabic queries into English keyword alternatives.

    Non-Arabic queries are returned unchanged and do not call
    the generation provider.
    """

    _ARABIC_PATTERN = re.compile(
        r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]"
    )

    def __init__(
        self,
        generation_provider: GenerationInterface,
    ) -> None:
        self._generation_provider = generation_provider

    @classmethod
    def contains_arabic(
        cls,
        query: str,
    ) -> bool:
        return bool(
            cls._ARABIC_PATTERN.search(query)
        )

    @staticmethod
    def _sanitize_keyword_query(
        query: str,
    ) -> str:
        cleaned = " ".join(query.split()).strip()

        cleaned = cleaned.replace("```text", "")
        cleaned = cleaned.replace("```", "")
        cleaned = cleaned.replace('"', "")
        cleaned = cleaned.replace("'", "")
        cleaned = cleaned.replace("`", "")

        parts = re.split(
            r"\s+OR\s+",
            cleaned,
            flags=re.IGNORECASE,
        )

        normalized_parts: list[str] = []

        for part in parts:
            normalized = re.sub(
                r"[^A-Za-z0-9\-\s]",
                " ",
                part,
            )

            normalized = " ".join(
                normalized.split()
            ).strip()

            if not normalized:
                continue

            if normalized.lower() == "or":
                continue

            if normalized.lower() not in {
                item.lower()
                for item in normalized_parts
            }:
                normalized_parts.append(
                    normalized
                )

        return " OR ".join(normalized_parts)

    async def translate_for_keyword_search(
        self,
        query: str,
    ) -> str:
        original_query = query.strip()

        if not original_query:
            raise ValueError(
                "query must not be empty"
            )

        if not self.contains_arabic(
            original_query
        ):
            return query

        result = await self._generation_provider.generate(
            system_prompt=(
                "Translate the Arabic search query into concise English "
                "keyword alternatives for PostgreSQL full-text search. "
                "Return exactly one line. "
                "Separate alternatives using uppercase OR. "
                "Do not use quotation marks. "
                "Translate only concepts explicitly present in the Arabic query. "
                "You may use direct English synonyms for those concepts. "
                "Do not add examples, answers, treatment names, medicine names, "
                "questionnaire names, or medical entities unless they are "
                "explicitly written in the Arabic query. "
                "Preserve abbreviations, medicine names, guideline identifiers, "
                "and numerical values only when they appear in the input. "
                "Do not answer the question. "
                "Do not explain anything."
            ),
            user_prompt=original_query,
            temperature=0.0,
            max_output_tokens=400,
        )

        translated = self._sanitize_keyword_query(
            result.text
        )

        if not translated:
            return original_query

        return translated

    async def close(self) -> None:
        await self._generation_provider.close()