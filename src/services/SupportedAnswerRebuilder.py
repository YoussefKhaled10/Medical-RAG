from src.stores.llm.GenerationInterface import GenerationInterface


class SupportedAnswerRebuilder:
    """Rebuild supported claims into a standalone conversational answer."""

    def __init__(
        self,
        provider: GenerationInterface,
        *,
        max_output_tokens: int = 500,
    ) -> None:
        self._provider = provider
        self._max_output_tokens = max_output_tokens

    @staticmethod
    def _system_prompt() -> str:
        return """You rebuild an answer after unsupported claims were removed.

The input contains only claims that already passed evidence validation. Rewrite those claims into a short, standalone, natural answer in the same language and communication style as the user's latest question.

STRICT RULES
- Use only the supported claim text supplied in the input.
- Do not add any medical fact, recommendation, benefit, warning, example, explanation, treatment option, or source.
- Preserve every source citation exactly, including source IDs such as [S1] and [S2].
- Keep each citation attached to the sentence that it supports.
- Make every sentence understandable on its own, even if earlier generated sentences were deleted.
- Remove discourse dependencies on deleted text. Do not begin with a connector that assumes missing earlier content.
- Do not refer to a previous point, another option, or an earlier explanation unless that reference is fully present in the supported text.
- Do not strengthen or weaken the meaning.
- Do not merge claims with different citations into one sentence.
- Return only the rebuilt user-facing answer. Do not return JSON, notes, headings, bullets, or analysis.
"""

    async def rebuild(
        self,
        *,
        question: str,
        supported_answer: str,
        response_language: str,
    ) -> str:
        clean_answer = supported_answer.strip()
        if not clean_answer:
            raise ValueError("supported_answer must not be empty")

        generation = await self._provider.generate(
            system_prompt=self._system_prompt(),
            user_prompt=(
                "LATEST USER QUESTION:\n"
                f"{question}\n\n"
                "RESPONSE LANGUAGE CODE:\n"
                f"{response_language}\n\n"
                "SUPPORTED CLAIMS WITH CITATIONS:\n"
                f"{clean_answer}\n\n"
                "Rebuild these supported claims as one short standalone answer."
            ),
            temperature=0.0,
            max_output_tokens=self._max_output_tokens,
        )
        rebuilt = generation.text.strip()
        if not rebuilt:
            raise ValueError("answer rebuilder returned empty text")
        return rebuilt
