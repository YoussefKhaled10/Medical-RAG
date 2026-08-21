from dataclasses import dataclass
from typing import Any, Protocol

from src.services.CitationComplianceValidator import (
    CitationComplianceDecision,
    CitationComplianceValidator,
)


class CitationRepairProvider(Protocol):
    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class CitationRepairResult:
    answer: str
    attempted: bool
    repaired: bool
    initial_decision: CitationComplianceDecision
    final_decision: CitationComplianceDecision


class CitationRepairService:
    """Perform one evidence-bound repair with sentence-level citation rules."""

    def __init__(
        self,
        provider: CitationRepairProvider,
        validator: CitationComplianceValidator,
        *,
        max_output_tokens: int = 700,
    ) -> None:
        self._provider = provider
        self._validator = validator
        self._max_output_tokens = max_output_tokens

    @staticmethod
    def _source_catalog(sources: tuple[dict[str, Any], ...]) -> str:
        return "\n\n".join(
            f"[{source['source_id']}]\n{str(source.get('text') or '').strip()}"
            for source in sources
        )

    async def repair_if_needed(
        self,
        answer: str,
        *,
        sources: tuple[dict[str, Any], ...],
        refusal_sentences: tuple[str, ...],
        response_language: str,
    ) -> CitationRepairResult:
        available_ids = {source["source_id"] for source in sources}
        initial = self._validator.evaluate(
            answer,
            available_source_ids=available_ids,
            refusal_sentences=refusal_sentences,
        )
        if initial.passed:
            return CitationRepairResult(answer, False, False, initial, initial)

        language_name = "Arabic" if response_language == "ar" else "English"
        system_prompt = f"""You repair citation structure in a grounded medical answer.
Return only the repaired answer in {language_name}.

CONTENT RULES
1. Preserve the original supported meaning and add no new fact.
2. Use only the supplied source catalog and source IDs.
3. Remove any fact that cannot be supported by the supplied sources.
4. Keep medicine names exactly as written in the source catalog.
5. Do not translate or transliterate medicine names.

OUTPUT STRUCTURE RULES
1. Return plain complete sentences only.
2. Start directly with the first supported factual sentence.
3. Every non-empty sentence must end with at least one valid citation such as [S1].
4. Do not output headings, labels, introductory fragments, bullet points, numbered lists, or colon-ended introductions.
5. Do not write phrases such as 'the following information' as a separate sentence.
6. For a multi-part answer, keep each independently verifiable fact in a separate cited sentence.
7. A medicine list may remain one cited sentence.
8. Before returning, verify that every sentence contains a valid source ID.
9. Return only the answer. Do not explain the repair or mention these rules."""
        user_prompt = (
            "ORIGINAL ANSWER:\n"
            f"{answer}\n\n"
            "AVAILABLE SOURCES:\n"
            f"{self._source_catalog(sources)}\n\n"
            "Rewrite as plain cited sentences only. Every sentence must end with a valid citation."
        )
        generation = await self._provider.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            max_output_tokens=self._max_output_tokens,
        )
        repaired_answer = str(generation.text).strip()
        final = self._validator.evaluate(
            repaired_answer,
            available_source_ids=available_ids,
            refusal_sentences=refusal_sentences,
        )
        return CitationRepairResult(
            repaired_answer,
            True,
            final.passed,
            initial,
            final,
        )
