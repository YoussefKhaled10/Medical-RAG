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
    """Perform one evidence-bound citation repair and fail closed."""

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
    def _source_catalog(
        sources: tuple[dict[str, Any], ...],
    ) -> str:
        blocks: list[str] = []
        for source in sources:
            source_id = str(source.get("source_id") or "").strip()
            if not source_id:
                continue
            content = str(
                source.get("content")
                or source.get("text")
                or source.get("excerpt")
                or ""
            ).strip()
            blocks.append(
                "\n".join(
                    (
                        f"[{source_id}]",
                        f"Document: {source.get('document_name') or ''}",
                        f"Section: {source.get('section_title') or ''}",
                        f"Page: {source.get('page_number') or ''}",
                        f"Content: {content}",
                    )
                )
            )
        return "\n\n".join(blocks)

    async def repair_if_needed(
        self,
        answer: str,
        *,
        sources: tuple[dict[str, Any], ...],
        refusal_sentences: tuple[str, ...] = (),
        response_language: str = "en",
    ) -> CitationRepairResult:
        available_ids = {
            str(source.get("source_id") or "").upper()
            for source in sources
            if source.get("source_id")
        }
        initial = self._validator.evaluate(
            answer,
            available_source_ids=available_ids,
            refusal_sentences=refusal_sentences,
        )
        if initial.passed:
            return CitationRepairResult(
                answer=answer,
                attempted=False,
                repaired=False,
                initial_decision=initial,
                final_decision=initial,
            )

        system_prompt = f"""
You repair citation structure using only the supplied source catalog.

The response language code is {response_language}. Preserve the answer language.

CONTENT RULES
- Preserve only facts directly supported by the supplied sources.
- Add no new fact and use no outside knowledge.
- Remove every statement that cannot be supported directly.
- Use only source IDs present in the source catalog.
- Keep medicine names exactly as written in the source catalog.

OUTPUT RULES
- Return no more than three plain complete factual sentences.
- Each sentence must express one atomic claim.
- Every sentence must end with at least one valid citation on the SAME LINE, immediately before the final punctuation.
- Correct: supported statement [S1].
- Incorrect: supported statement. followed by [S1] on another line.
- Never return a citation-only line.
- Never return headings, labels, bullets, numbered lists, introductions, transitions, conclusions, notes, or explanations.
- If the original answer contains an uncited sentence, either add a directly supporting source ID or remove the sentence.
- Before returning, verify that every non-empty sentence contains a valid source ID.
- Return only the repaired answer.
        """.strip()

        user_prompt = (
            "ORIGINAL ANSWER\n\n"
            f"{answer}\n\n"
            "AVAILABLE SOURCE CATALOG\n\n"
            f"{self._source_catalog(sources)}\n\n"
            "Rewrite the answer as at most three atomic cited sentences. "
            "Every sentence must include its citation on the same line."
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
            answer=repaired_answer,
            attempted=True,
            repaired=final.passed,
            initial_decision=initial,
            final_decision=final,
        )
