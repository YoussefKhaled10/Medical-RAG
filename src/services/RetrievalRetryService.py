import json
from dataclasses import dataclass
from typing import Any

from src.stores.llm.GenerationInterface import GenerationInterface


@dataclass(frozen=True, slots=True)
class RetrievalRetryQuery:
    semantic_query: str
    keyword_hints: tuple[str, ...]
    rationale: str


class RetrievalRetryService:
    """Improve a weak retrieval query using a model, without phrase rules."""

    def __init__(
        self,
        provider: GenerationInterface,
        *,
        max_output_tokens: int = 450,
    ) -> None:
        self._provider = provider
        self._max_output_tokens = max_output_tokens

    @staticmethod
    def _candidate_summaries(
        retrieval: dict[str, Any],
    ) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for item in retrieval.get("results", [])[:5]:
            text = " ".join(str(item.get("text") or "").split())
            summaries.append(
                {
                    "document_name": item.get("document_name"),
                    "section_title": item.get("section_title"),
                    "rerank_score": item.get("rerank_score"),
                    "preview": text[:700],
                }
            )
        return summaries

    @staticmethod
    def _system_prompt() -> str:
        return """You improve a weak retrieval query for an evidence-grounded alcohol-recovery system.

The rewritten query is for searching medical guidelines and treatment documents. It is not a conversational response to the user.

Identify the factual evidence topic behind the user's request. Convert conversational requests for help, guidance, direction, or starting recovery into the clinical information topic likely to appear in evidence documents.

Do not preserve conversational wording. Prefer medical-document terminology that matches the detected intent, such as treatment options, behavioral interventions, psychosocial treatment, recovery support, shared decision-making, screening, withdrawal, relapse prevention, or health effects.

Do not narrow a broad treatment request to only initial steps unless the user's meaning specifically requires that scope. Do not answer the question. Do not add a diagnosis, dosage, personalized recommendation, or new intent.

Return exactly one JSON object:
{
  "semantic_query": "concise standalone medical retrieval question",
  "keyword_hints": [
    "specific English medical search concept",
    "another distinct medical search concept"
  ],
  "rationale": "brief retrieval-focused explanation"
}

Return four to six distinct keyword hints when the intent is broad enough. The hints must be short evidence-search concepts, not conversational sentences."""

    @staticmethod
    def _parse_payload(text: str) -> dict[str, Any]:
        payload = json.loads(text.strip())
        if not isinstance(payload, dict):
            raise ValueError("retry model output must be a JSON object")
        return payload

    @staticmethod
    def _keyword_hints(value: Any) -> tuple[str, ...]:
        if not isinstance(value, list):
            return tuple()

        unique: list[str] = []
        seen: set[str] = set()
        for item in value:
            hint = " ".join(str(item).split()).strip()
            if not hint:
                continue
            key = hint.casefold()
            if key in seen:
                continue
            seen.add(key)
            unique.append(hint)
        return tuple(unique[:6])

    async def rewrite(
        self,
        *,
        user_question: str,
        intent: str,
        first_query: str,
        retrieval: dict[str, Any],
    ) -> RetrievalRetryQuery:
        user_prompt = json.dumps(
            {
                "user_question": user_question,
                "intent": intent,
                "first_semantic_query": first_query,
                "first_effective_keyword_query": retrieval.get(
                    "effective_keyword_query"
                ),
                "weak_candidates": self._candidate_summaries(retrieval),
            },
            ensure_ascii=False,
        )

        generation = await self._provider.generate(
            system_prompt=self._system_prompt(),
            user_prompt=user_prompt,
            temperature=0.0,
            max_output_tokens=self._max_output_tokens,
        )
        payload = self._parse_payload(generation.text)

        semantic_query = " ".join(
            str(payload.get("semantic_query") or "").split()
        ).strip()
        if not semantic_query:
            raise ValueError("retry semantic_query must not be empty")

        keyword_hints = self._keyword_hints(
            payload.get("keyword_hints")
        )
        if not keyword_hints:
            raise ValueError("retry keyword_hints must not be empty")

        return RetrievalRetryQuery(
            semantic_query=semantic_query,
            keyword_hints=keyword_hints,
            rationale=str(payload.get("rationale") or "").strip(),
        )
