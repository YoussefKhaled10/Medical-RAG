import re
from dataclasses import asdict
from time import perf_counter
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src.services.ContextBuilder import ContextBuilder
from src.services.EvidenceBuilder import EvidenceBuilder
from src.services.RAGPromptBuilder import RAGPromptBuilder
from src.services.RelevanceGate import RelevanceGate
from src.services.RetrievalPipelineService import RetrievalPipelineService
from src.stores.llm.GenerationInterface import GenerationInterface


class RAGService:
    """Run retrieval, relevance gating, generation, and evidence assembly."""

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipelineService,
        context_builder: ContextBuilder,
        prompt_builder: RAGPromptBuilder,
        generation_provider: GenerationInterface,
        relevance_gate: RelevanceGate,
        evidence_builder: EvidenceBuilder | None = None,
    ) -> None:
        self._retrieval_pipeline = retrieval_pipeline
        self._context_builder = context_builder
        self._prompt_builder = prompt_builder
        self._generation_provider = generation_provider
        self._relevance_gate = relevance_gate
        self._evidence_builder = evidence_builder or EvidenceBuilder()

    @staticmethod
    def _normalize_digits(value: str) -> str:
        return value.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))

    @staticmethod
    def _normalize_source_citations(answer: str) -> str:
        return re.sub(
            r"\[\s*[سsS]\s*([0-9٠-٩]+)\s*\]",
            lambda match: (
                f"[S{RAGService._normalize_digits(match.group(1))}]"
            ),
            answer,
        )

    @staticmethod
    def _used_source_ids(answer: str) -> set[str]:
        return set(re.findall(r"\[(S\d+)\]", answer))

    @classmethod
    def _select_sources(
        cls,
        answer: str,
        sources: tuple[dict[str, Any], ...],
    ) -> list[dict[str, Any]]:
        used = cls._used_source_ids(answer)
        return [source for source in sources if source["source_id"] in used]

    @staticmethod
    def _milliseconds(start: float, end: float) -> float:
        return round((end - start) * 1000, 2)

    async def ask(
        self,
        *,
        session: AsyncSession,
        question: str,
        project_id: int | None = None,
        asset_id: int | None = None,
        retrieval_limit: int = 5,
        temperature: float = 0.0,
        max_output_tokens: int = 1200,
    ) -> dict[str, Any]:
        total_started = perf_counter()
        normalized_question = " ".join(question.split()).strip()
        if not normalized_question:
            raise ValueError("question must not be empty")

        response_language = self._prompt_builder.detect_language(
            normalized_question
        )
        insufficient_answer = self._prompt_builder.insufficient_answer(
            response_language
        )

        retrieval_started = perf_counter()
        retrieval = await self._retrieval_pipeline.search(
            session=session,
            query=normalized_question,
            limit=retrieval_limit,
            project_id=project_id,
            asset_id=asset_id,
            use_deduplication=True,
            use_reranking=True,
        )
        retrieval_finished = perf_counter()
        results = retrieval["results"]

        relevance = self._relevance_gate.evaluate_as_dict(results)
        if not relevance["passed"]:
            total_finished = perf_counter()
            return self._refusal_output(
                question=normalized_question,
                answer=insufficient_answer,
                language=response_language,
                retrieval=retrieval,
                relevance=relevance,
                refusal_reason=relevance["reason"],
                timings_ms={
                    "retrieval": self._milliseconds(
                        retrieval_started, retrieval_finished
                    ),
                    "context_building": 0.0,
                    "generation": 0.0,
                    "evidence_building": 0.0,
                    "total": self._milliseconds(total_started, total_finished),
                },
            )

        context_started = perf_counter()
        context = self._context_builder.build(results)
        prompt = self._prompt_builder.build(
            question=normalized_question,
            context=context,
        )
        context_finished = perf_counter()

        generation_started = perf_counter()
        generation = await self._generation_provider.generate(
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        generation_finished = perf_counter()

        answer = self._normalize_source_citations(generation.text.strip())
        refused = answer == insufficient_answer
        used_sources = (
            [] if refused else self._select_sources(answer, context.sources)
        )

        evidence_started = perf_counter()
        evidence_items = (
            []
            if refused
            else self._evidence_builder.build(
                used_sources=used_sources,
                retrieval_results=results,
            )
        )
        evidence_finished = perf_counter()
        evidence = [asdict(item) for item in evidence_items]
        total_finished = perf_counter()

        return {
            "question": normalized_question,
            "answer": answer,
            "recommendation": answer,
            "answer_language": response_language,
            "grounded": bool(evidence) and not refused,
            "refused": refused,
            "refusal": (
                {
                    "reason": "generation_refusal",
                    "stage": "generation",
                    "generation_skipped": False,
                }
                if refused
                else None
            ),
            "relevance": relevance,
            "provider": generation.provider,
            "model": generation.model,
            "request_id": generation.request_id,
            "sources": used_sources,
            "evidence": evidence,
            "retrieval": retrieval,
            "context_characters": context.total_characters,
            "generation_config": {
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            },
            "timings_ms": {
                "retrieval": self._milliseconds(
                    retrieval_started, retrieval_finished
                ),
                "context_building": self._milliseconds(
                    context_started, context_finished
                ),
                "generation": self._milliseconds(
                    generation_started, generation_finished
                ),
                "evidence_building": self._milliseconds(
                    evidence_started, evidence_finished
                ),
                "total": self._milliseconds(total_started, total_finished),
            },
        }

    @staticmethod
    def _refusal_output(
        *,
        question: str,
        answer: str,
        language: str,
        retrieval: dict[str, Any],
        relevance: dict[str, Any],
        refusal_reason: str,
        timings_ms: dict[str, float],
    ) -> dict[str, Any]:
        return {
            "question": question,
            "answer": answer,
            "recommendation": answer,
            "answer_language": language,
            "grounded": False,
            "refused": True,
            "refusal": {
                "reason": refusal_reason,
                "stage": "pre_generation",
                "generation_skipped": True,
            },
            "relevance": relevance,
            "provider": None,
            "model": None,
            "request_id": None,
            "sources": [],
            "evidence": [],
            "retrieval": retrieval,
            "context_characters": 0,
            "generation_config": None,
            "timings_ms": timings_ms,
        }

    async def close(self) -> None:
        await self._retrieval_pipeline.close()
        await self._generation_provider.close()
