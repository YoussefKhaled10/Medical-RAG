from src.stores.llm.GenerationInterface import GenerationInterface


class SupportedAnswerRebuilder:
    """Rebuild supported claims into a standalone conversational answer in the target language and style."""

    def __init__(
        self,
        provider: GenerationInterface,
        *,
        max_output_tokens: int = 600,
    ) -> None:
        self._provider = provider
        self._max_output_tokens = max_output_tokens

    @staticmethod
    def _system_prompt() -> str:
        return """You rebuild a medical recovery answer after unsupported claims were removed.

The input contains only claims that already passed strict evidence validation. Rewrite those claims into a coherent, natural answer matching the exact language of the user's latest question and the requested presentation style.

STRICT RULES
- Use only the supported claim text supplied in the input.
- Do not add any new medical fact, recommendation, benefit, warning, example, explanation, treatment option, or source.
- Preserve every source citation exactly, including source IDs such as [S1] and [S2] on the same line before punctuation.
- Keep each citation attached to the sentence that it supports.
- Write 100% of the text in the specified response language.
- Make every sentence understandable on its own, even if earlier generated sentences were deleted.
- Remove discourse dependencies on deleted text. Do not begin with a connector that assumes missing earlier content.
- Do not strengthen or weaken the medical meaning.
- Return only the rebuilt user-facing answer. Do not return JSON, notes, or analysis.
"""

    async def rebuild(
        self,
        *,
        question: str,
        supported_answer: str,
        response_language: str,
        variation_profile: str | None = None,
    ) -> str:
        clean_answer = supported_answer.strip()
        if not clean_answer:
            raise ValueError("supported_answer must not be empty")

        profile_text = f"STYLE / PROFILE: {variation_profile}\n" if variation_profile else ""

        generation = await self._provider.generate(
            system_prompt=self._system_prompt(),
            user_prompt=(
                "LATEST USER QUESTION:\n"
                f"{question}\n\n"
                "RESPONSE LANGUAGE CODE:\n"
                f"{response_language}\n\n"
                f"{profile_text}"
                "SUPPORTED CLAIMS WITH CITATIONS:\n"
                f"{clean_answer}\n\n"
                "Rebuild these supported claims into a standalone conversational answer in the specified response language."
            ),
            temperature=0.0,
            max_output_tokens=self._max_output_tokens,
        )
        rebuilt = generation.text.strip()
        if not rebuilt:
            raise ValueError("answer rebuilder returned empty text")
        return rebuilt
