import re
from dataclasses import asdict
from time import perf_counter
from typing import Any
from src.helpers.config import settings
from sqlalchemy.ext.asyncio import AsyncSession

from src.services.CitationAccuracyEvaluator import CitationAccuracyEvaluator
from src.services.CitationRepairService import CitationRepairService
from src.services.ClaimExtractor import ClaimExtractor
from src.services.ClaimSupportEvaluator import ClaimSupportEvaluator
from src.services.ContextBuilder import ContextBuilder
from src.services.EvidenceBuilder import EvidenceBuilder
from src.services.EvidenceStrengthClassifier import EvidenceStrengthClassifier
from src.services.LanguageDetector import LanguageDetector
from src.services.PostGenerationSafetyGate import PostGenerationSafetyGate
from src.services.RAGPromptBuilder import RAGPromptBuilder
from src.services.RefusalPolicy import RefusalPolicy
from src.services.IntentUnderstandingService import IntentUnderstandingService
from src.services.QueryUnderstandingService import QueryUnderstandingService
from src.services.RelevanceGate import RelevanceGate
from src.services.RetrievalPipelineService import RetrievalPipelineService
from src.services.RetrievalRetryService import RetrievalRetryService
from src.services.SupportedAnswerRebuilder import SupportedAnswerRebuilder
from src.services.UnsupportedClaimPruner import UnsupportedClaimPruner
from src.stores.llm.GenerationInterface import GenerationInterface


class RAGService:
    """Run retrieval, generation, evidence assembly, and safety validation."""

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipelineService,
        context_builder: ContextBuilder,
        prompt_builder: RAGPromptBuilder,
        generation_provider: GenerationInterface,
        relevance_gate: RelevanceGate,
        evidence_strength_classifier: EvidenceStrengthClassifier,
        claim_extractor: ClaimExtractor,
        claim_support_evaluator: ClaimSupportEvaluator,
        citation_repair_service: CitationRepairService,
        citation_accuracy_evaluator: CitationAccuracyEvaluator,
        post_generation_safety_gate: PostGenerationSafetyGate,
        claim_judge_provider: GenerationInterface,
        evidence_builder: EvidenceBuilder | None = None,
        query_understanding_service: IntentUnderstandingService | None = None,
        retrieval_retry_service: RetrievalRetryService | None = None,
        unsupported_claim_pruner: UnsupportedClaimPruner | None = None,
        supported_answer_rebuilder: SupportedAnswerRebuilder | None = None,
    ) -> None:
        self._retrieval_pipeline = retrieval_pipeline
        self._context_builder = context_builder
        self._prompt_builder = prompt_builder
        self._generation_provider = generation_provider
        self._relevance_gate = relevance_gate
        self._evidence_strength_classifier = evidence_strength_classifier
        self._claim_extractor = claim_extractor
        self._claim_support_evaluator = claim_support_evaluator
        self._citation_repair_service = citation_repair_service
        self._citation_accuracy_evaluator = citation_accuracy_evaluator
        self._post_generation_safety_gate = post_generation_safety_gate
        self._claim_judge_provider = claim_judge_provider
        self._evidence_builder = evidence_builder or EvidenceBuilder()
        self._unsupported_claim_pruner = (
            unsupported_claim_pruner
            or UnsupportedClaimPruner()
        )
        self._supported_answer_rebuilder = supported_answer_rebuilder
        if query_understanding_service is None:
            raise ValueError("query_understanding_service is required")
        self._query_understanding = query_understanding_service
        self._retrieval_retry_service = retrieval_retry_service

    @staticmethod
    def _normalize_question(question: str) -> str:
        normalized = " ".join(
            str(question or "").split()
        ).strip()

        if not normalized:
            raise ValueError(
                "question must not be empty"
            )

        return normalized

    @staticmethod
    def _clean_conversation_history(
        conversation_history: list[dict[str, str]] | None,
    ) -> list[dict[str, str]]:
        cleaned: list[dict[str, str]] = []

        for item in (conversation_history or [])[-8:]:
            role = str(
                item.get("role") or ""
            ).strip().lower()

            content = " ".join(
                str(
                    item.get("content") or ""
                ).split()
            ).strip()

            if role not in {
                "user",
                "assistant",
            }:
                continue

            if not content:
                continue

            cleaned.append(
                {
                    "role": role,
                    "content": content[:1800],
                }
            )

        return cleaned

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
        return [
            source
            for source in sources
            if source["source_id"] in used
        ]

    @staticmethod
    def _milliseconds(start: float, end: float) -> float:
        return round((end - start) * 1000, 2)

    @staticmethod
    def _refusal_sentences() -> tuple[str, ...]:
        """Known refusal text excluded from factual claim extraction."""
        questions = {
            "ar": "ما الجرعة المناسبة لحالتي؟",
            "en": "What dose is right for my condition?",
            "fr": "Quelle dose convient à mon cas ?",
        }
        reasons = (
            "insufficient_evidence",
            "out_of_scope",
            "professional_care",
            "personalized_treatment",
            "urgent_help",
            "prompt_injection",
        )
        return tuple(
            RefusalPolicy.decision(question, reason=reason).message
            for question in questions.values()
            for reason in reasons
        )

    @staticmethod
    def _social_fallback(language: str) -> str:
        """Use only when the intent provider fails to return social text."""
        messages = {
            "ar": "تمام، أنا موجود لو احتجتني.",
            "fr": "D'accord, je reste disponible si vous en avez besoin.",
            "en": "All right, I'm here if you need me.",
        }
        return messages.get(language, messages["en"])

    @staticmethod
    def _social_output(
        *,
        question: str,
        answer: str,
        language: str,
        query_understanding: dict[str, Any],
        total_ms: float,
    ) -> dict[str, Any]:
        """Return the intent model's social response without RAG."""
        return {
            "question": question,
            "answer": answer,
            "recommendation": answer,
            "answer_language": language,
            "grounded": False,
            "refused": False,
            "safety_flagged": False,
            "refusal": None,
            "refusal_guidance": None,
            "relevance": {
                "passed": True,
                "reason": "social_conversation_no_retrieval",
                "top_score": None,
                "second_score": None,
                "score_margin": None,
                "threshold": 0.320982,
                "qualified_chunk_count": 0,
                "minimum_qualified_chunks": 0,
            },
            "evidence_strength": {
                "level": "insufficient",
                "top_score": None,
                "relevance_threshold": 0.320982,
                "strong_threshold": 0.533,
                "language_policy": "social_response_no_evidence_needed",
                "answer_allowed": True,
                "rationale": "Social conversation does not require retrieval.",
            },
            "provider": None,
            "model": None,
            "request_id": None,
            "sources": [],
            "evidence": [],
            "claims": [],
            "claim_results": [],
            "citation_repair": {
                "attempted": False,
                "repaired": False,
                "initial_passed": True,
                "final_passed": True,
                "reason": "social_conversation_no_generation",
            },
            "claim_validation": {
                "passed": True,
                "reason": "social_conversation_no_generation",
                "total_claims": 0,
                "supported_claims": 0,
                "unsupported_claims": 0,
                "faithfulness": 1.0,
                "minimum_faithfulness": 0.90,
                "unsupported_claim_ids": (),
            },
            "citation_evaluation": {
                "total_claims": 0,
                "cited_claims": 0,
                "uncited_claims": 0,
                "citation_completeness": 1.0,
                "total_citation_links": 0,
                "correct_citation_links": 0,
                "incorrect_citation_links": 0,
                "citation_accuracy": None,
                "unique_source_count": 0,
                "invalid_source_ids": (),
                "metadata_accuracy": None,
                "claim_support_accuracy": None,
                "passed": True,
                "minimum_citation_accuracy": 0.95,
                "minimum_citation_completeness": 1.0,
                "items": (),
                "reason": "social_conversation_no_generation",
            },
            "retrieval": {
                "results": [],
                "pre_dedup_count": 0,
                "post_dedup_count": 0,
                "removed_duplicates": [],
                "cross_language_keyword_used": False,
                "effective_keyword_query": None,
                "query_understanding": query_understanding,
                "social_conversation": True,
            },
            "context_characters": 0,
            "generation_config": None,
            "timings_ms": {
                "retrieval": 0.0,
                "context_building": 0.0,
                "generation": 0.0,
                "evidence_building": 0.0,
                "citation_repair": 0.0,
                "claim_validation": 0.0,
                "citation_evaluation": 0.0,
                "total": total_ms,
            },
        }

    @staticmethod
    def _remove_dosage_details(answer: str) -> str:
        """Remove exact medical quantities and schedules from a final answer."""
        text = str(answer or "").strip()
        if not text:
            return text
        unit = r"(?:mg|mcg|µg|g|kg|ml|mL|iu|IU|مغ|مجم|ميكروغرام|غرام|جرام|مل|لتر)"
        number = r"(?:\d+(?:[.,]\d+)?)"
        range_part = rf"{number}(?:\s*(?:-|–|—|to|إلى)\s*{number})?"
        dosage = rf"{range_part}\s*{unit}"
        text = re.sub(rf"\(\s*{dosage}(?:\s+[^)]{{0,55}})?\)", "", text, flags=re.IGNORECASE)
        text = re.sub(dosage, "", text, flags=re.IGNORECASE)
        patterns = (
            r"\b(?:once|twice|three|four)\s+(?:a|per)\s+(?:day|week|month)\b",
            r"\b\d+\s*(?:times?|x)\s*(?:a|per)\s*(?:day|week|month)\b",
            r"\bevery\s+\d+\s*(?:hours?|days?|weeks?|months?)\b",
            r"\bfor\s+\d+(?:\s*(?:-|–|—|to)\s*\d+)?\s*(?:days?|weeks?|months?|years?)\b",
            r"(?:مرة|مرتين|ثلاث\s+مرات|أربع\s+مرات)\s+(?:يوميًا|يومياً|في\s+اليوم|أسبوعيًا|أسبوعياً)",
            r"كل\s+\d+\s*(?:ساعات?|أيام?|أسابيع|أشهر)",
            r"لمدة\s+\d+(?:\s*(?:-|–|—|إلى)\s*\d+)?\s*(?:أيام?|أسابيع|أشهر|سنوات)",
            r"\b(?:daily|weekly|monthly)\b",
            r"(?:يوميًا|يومياً|أسبوعيًا|أسبوعياً|شهريًا|شهرياً)",
        )
        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        text = re.sub(r"\(\s*\)", "", text)
        text = re.sub(r"\s+([،,؛;:.])", r"\1", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _medical_boundary(language: str) -> str:
        boundaries = {
            "ar": "هذه معلومات عامة مستندة إلى المصادر المتاحة، وليست وصفة أو جرعة مناسبة لحالة فردية. يُفضّل استشارة طبيب أو صيدلي مؤهل قبل بدء أي دواء أو مكمل، أو إيقافه، أو تغييره.",
            "en": "This is general information from the available sources, not an individualized prescription or dosage. Consult a qualified doctor or pharmacist before starting, stopping, or changing any medicine or supplement.",
            "fr": "Ces informations générales proviennent des sources disponibles et ne constituent pas une prescription ni une posologie personnalisée. Consultez un médecin ou un pharmacien qualifié avant de commencer, d’arrêter ou de modifier un médicament ou un complément.",
        }
        return boundaries.get(language, boundaries["en"])

    @staticmethod
    def _requires_medical_boundary(answer: str) -> bool:
        value = str(answer or "").casefold()
        triggers = (
            "دواء", "دوائي", "علاج", "فيتامين", "مكمل", "ثيامين", "إلكتروليت",
            "مغنيسيوم", "بوتاسيوم", "فوسفات", "medicine", "medication", "treatment",
            "vitamin", "supplement", "thiamine", "electrolyte", "pharmacological",
            "naltrexone", "acamprosate", "disulfiram", "nalmefene", "نالتريكسون",
            "أكامبروسيت", "ديسلفيرام", "نالميفين",
        )
        return any(item in value for item in triggers)

    @classmethod
    def _prepare_final_medical_answer(cls, answer: str, language: str) -> str:
        clean = cls._remove_dosage_details(answer)
        if not clean or not cls._requires_medical_boundary(clean):
            return clean
        boundary = cls._medical_boundary(language)
        if boundary in clean:
            return clean
        return f"{clean}\n\n{boundary}"

    @staticmethod
    def _post_generation_refusal(language: str) -> str:
        messages = {
            "ar": (
                "تعذر التحقق من جميع المعلومات الواردة في الإجابة، "
                "لذلك لم يتم عرضها."
            ),
            "fr": (
                "Toutes les informations de la réponse n'ont pas pu être "
                "vérifiées, elle n'a donc pas été affichée."
            ),
            "en": (
                "Not all information in the generated answer could be "
                "verified, so the answer was not displayed."
            ),
        }
        return messages.get(language, messages["en"])

    async def _recover_all_unsupported_claims(
        self,
        *,
        question: str,
        context: Any,
        results: list[dict[str, Any]],
        query_understanding: Any,
        conversation_history: list[dict[str, str]],
        response_language: str,
        refusal_sentences: tuple[str, ...],
        max_output_tokens: int,
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[Any], list[Any], dict[str, Any]] | None:
        """Run one strict recovery pass and keep it only if at least one claim is supported."""
        prompt = self._prompt_builder.build(
            question=question,
            context=context,
            query_understanding=query_understanding,
            conversation_history=conversation_history,
            response_language=response_language,
            variation_profile="direct_answer",
        )
        recovery_rules = """
ALL-UNSUPPORTED RECOVERY PASS:
The previous answer failed because it combined supported and unsupported details.
Generate a new, shorter answer using only facts explicitly stated in AVAILABLE EVIDENCE.
Write one independently verifiable medical claim per sentence.
Do not create long symptom or treatment lists.
Do not infer specific symptoms from a broad clinical category.
Use exactly one supporting source per sentence whenever possible.
Omit every detail not explicitly stated in the cited source.
A short answer containing one supported fact is better than a complete refusal.
""".strip()
        generation = await self._generation_provider.generate(
            system_prompt=prompt.system_prompt + "\n\n" + recovery_rules,
            user_prompt=prompt.user_prompt,
            temperature=0.0,
            max_output_tokens=min(max_output_tokens, 700),
            top_p=None,
        )
        candidate = self._normalize_source_citations(generation.text.strip())
        refusal_category, candidate = RefusalPolicy.parse_marked_answer(candidate)
        if refusal_category or not candidate:
            return None
        repaired = await self._citation_repair_service.repair_if_needed(
            candidate,
            sources=context.sources,
            refusal_sentences=refusal_sentences,
            response_language=response_language,
        )
        if not repaired.final_decision.passed:
            return None
        candidate = self._normalize_source_citations(repaired.answer)
        used_sources = self._select_sources(candidate, context.sources)
        evidence_items = self._evidence_builder.build(used_sources=used_sources, retrieval_results=results)
        evidence = [asdict(item) for item in evidence_items]
        claims = self._claim_extractor.extract(candidate, refusal_sentences=refusal_sentences)
        support_results = await self._claim_support_evaluator.evaluate(claims, evidence=evidence)
        if not any(item.supported for item in support_results):
            return None
        repair_info = {
            "attempted": True,
            "repaired": repaired.repaired,
            "initial_passed": repaired.initial_decision.passed,
            "final_passed": repaired.final_decision.passed,
            "reason": "all_unsupported_recovery_pass",
        }
        return candidate, used_sources, evidence, claims, support_results, repair_info

    async def ask(
        self,
        *,
        session: AsyncSession,
        question: str,
        project_id: int | list[int] | tuple[int, ...] | None = None,
        asset_id: int | None = None,
        retrieval_limit: int = 5,
        temperature: float = 0.0,
        max_output_tokens: int = 1200,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:

        total_started = perf_counter()
        normalized_question = " ".join(question.split()).strip()
        if not normalized_question:
            raise ValueError("question must not be empty")

        cleaned_history = self._clean_conversation_history(
            conversation_history
        )

        # Extract last explicit user language for short neutral message fallback
        last_explicit_user_lang = None
        for h in reversed(cleaned_history):
            if h.get("role") == "user":
                content = str(h.get("content") or "").strip()
                det = LanguageDetector.detect(content)
                if det.code in {"ar", "en", "fr", "es", "de"}:
                    last_explicit_user_lang = det.code
                    break

        detected_language = LanguageDetector.detect(
            normalized_question,
            last_explicit_user_language=last_explicit_user_lang,
        )
        response_language = detected_language.code

        understood_query = await self._query_understanding.understand(
            normalized_question,
            conversation_history=cleaned_history,
        )
        refusal_sentences = self._refusal_sentences()

        # The intent model both classifies and writes social responses.
        # QueryUnderstanding currently carries that response in
        # clarification_message, so no retrieval or second generation call
        # is needed here.
        if understood_query.intent == "social":
            social_answer = (
                understood_query.clarification_message
                or self._social_fallback(response_language)
            )
            social_finished = perf_counter()
            return self._social_output(
                question=normalized_question,
                answer=social_answer,
                language=response_language,
                query_understanding=understood_query.as_dict(),
                total_ms=self._milliseconds(
                    total_started,
                    social_finished,
                ),
            )

        # Handle safety, ambiguity, and clear out-of-scope requests
        # before retrieval, generation, or evidence validation.
        pre_reason = understood_query.safety_reason
        if pre_reason is None and understood_query.intent == "out_of_scope":
            pre_reason = "out_of_scope"

        if pre_reason is not None or understood_query.ambiguous:
            if understood_query.ambiguous:
                answer = (
                    understood_query.clarification_message
                    or RefusalPolicy.decision(
                        normalized_question,
                        reason="insufficient_evidence",
                        response_language=response_language,
                    ).message
                )
                category = "insufficient_evidence"
                requires_professional = False
                urgent = False
            else:
                decision = RefusalPolicy.decision(
                    normalized_question,
                    reason=pre_reason,
                    response_language=response_language,
                )
                answer = decision.message
                category = decision.reason
                requires_professional = decision.requires_professional
                urgent = decision.urgent


            total_finished = perf_counter()
            empty_retrieval = {
                "results": [],
                "pre_dedup_count": 0,
                "post_dedup_count": 0,
                "removed_duplicates": [],
                "cross_language_keyword_used": False,
                "effective_keyword_query": None,
                "query_understanding": understood_query.as_dict(),
            }
            empty_relevance = {
                "passed": False,
                "reason": "pre_retrieval_policy_decision",
                "top_score": None,
                "second_score": None,
                "score_margin": None,
                "threshold": 0.320982,
                "qualified_chunk_count": 0,
                "minimum_qualified_chunks": 1,
            }
            empty_strength = {
                "level": "insufficient",
                "top_score": None,
                "relevance_threshold": 0.320982,
                "strong_threshold": 0.533,
                "language_policy": "generation_not_run",
                "answer_allowed": False,
                "rationale": "Handled before retrieval by policy.",
            }
            return self._refusal_output(
                question=normalized_question,
                answer=answer,
                language=response_language,
                retrieval=empty_retrieval,
                relevance=empty_relevance,
                refusal_guidance={
                    "category": category,
                    "requires_professional": requires_professional,
                    "urgent": urgent,
                },
                evidence_strength=empty_strength,
                refusal_reason=category,
                refusal_stage="pre_generation",
                generation_skipped=True,
                safety_flagged=urgent,
                citation_repair={
                    "attempted": False,
                    "repaired": False,
                    "initial_passed": True,
                    "final_passed": True,
                    "reason": "generation_not_run",
                },
                citation_evaluation=self._empty_citation_evaluation(
                    reason="generation_not_run"
                ),
                claim_validation=self._empty_claim_validation(
                    reason="generation_not_run"
                ),
                claims=[],
                timings_ms={
                    "retrieval": 0.0,
                    "context_building": 0.0,
                    "generation": 0.0,
                    "evidence_building": 0.0,
                    "citation_repair": 0.0,
                    "claim_validation": 0.0,
                    "citation_evaluation": 0.0,
                    "total": self._milliseconds(total_started, total_finished),
                },
            )

        retrieval_started = perf_counter()
        retrieval = await self._retrieval_pipeline.search(
            session=session,
            query=understood_query.semantic_query,
            limit=retrieval_limit,
            project_id=project_id,
            asset_id=asset_id,
            use_deduplication=True,
            use_reranking=True,
            semantic_query=understood_query.semantic_query,
            keyword_hints=understood_query.keyword_hints,
        )
        retrieval_finished = perf_counter()

        retrieval["query_understanding"] = understood_query.as_dict()
        results = retrieval["results"]
        relevance = self._relevance_gate.evaluate_as_dict(results)
        evidence_strength = (
            self._evidence_strength_classifier.classify_as_dict(
                relevance.get("top_score")
            )
        )

        # ── Retrieval retry ─────────────────────────────────────────────
        # If the first pass fails the relevance gate and a retry service is
        # wired, generate a richer retrieval query from the weak candidates
        # and make a second attempt before giving up.
        retrieval_retry_info: dict[str, Any] = {
            "attempted": False,
            "used": False,
            "first_top_score": relevance.get("top_score"),
            "retry_top_score": None,
            "retry_query": None,
            "rationale": None,
            "error": None,
        }

        if not relevance["passed"] and self._retrieval_retry_service is not None:
            try:
                retry_q = await self._retrieval_retry_service.rewrite(
                    user_question=normalized_question,
                    intent=understood_query.intent,
                    first_query=understood_query.semantic_query,
                    retrieval=retrieval,
                )
                retrieval_retry_info["attempted"] = True
                retrieval_retry_info["retry_query"] = retry_q.semantic_query
                retrieval_retry_info["rationale"] = retry_q.rationale

                retry_retrieval = await self._retrieval_pipeline.search(
                    session=session,
                    query=retry_q.semantic_query,
                    limit=retrieval_limit,
                    project_id=project_id,
                    asset_id=asset_id,
                    use_deduplication=True,
                    use_reranking=True,
                    semantic_query=retry_q.semantic_query,
                    keyword_hints=retry_q.keyword_hints,  # fresh start — no hints to avoid re-narrowing
                )
                retry_relevance = self._relevance_gate.evaluate_as_dict(
                    retry_retrieval["results"]
                )
                retrieval_retry_info["retry_top_score"] = retry_relevance.get(
                    "top_score"
                )

                if retry_relevance["passed"]:
                    # Replace everything with the better retrieval
                    retrieval_retry_info["used"] = True
                    retry_retrieval["query_understanding"] = (
                        understood_query.as_dict()
                    )
                    retry_retrieval["retrieval_retry"] = retrieval_retry_info
                    retrieval = retry_retrieval
                    results = retrieval["results"]
                    relevance = retry_relevance
                    evidence_strength = (
                        self._evidence_strength_classifier.classify_as_dict(
                            relevance.get("top_score")
                        )
                    )
                else:
                    retrieval["retrieval_retry"] = retrieval_retry_info
            except Exception as exc:  # noqa: BLE001
                retrieval_retry_info["error"] = str(exc)
                retrieval["retrieval_retry"] = retrieval_retry_info
        else:
            retrieval["retrieval_retry"] = retrieval_retry_info
        # ────────────────────────────────────────────────────────────────

        if not relevance["passed"]:
            refusal_decision = RefusalPolicy.decision(
                normalized_question,
                low_relevance=True,
                response_language=response_language,
            )
            total_finished = perf_counter()
            return self._refusal_output(
                question=normalized_question,
                answer=refusal_decision.message,
                language=response_language,
                retrieval=retrieval,
                relevance=relevance,
                refusal_guidance={
                    "category": refusal_decision.reason,
                    "requires_professional": (
                        refusal_decision.requires_professional
                    ),
                    "urgent": refusal_decision.urgent,
                },
                evidence_strength=evidence_strength,
                refusal_reason=refusal_decision.reason,
                refusal_stage="pre_generation",
                generation_skipped=True,
                safety_flagged=False,
                citation_repair={
                    "attempted": False,
                    "repaired": False,
                    "initial_passed": True,
                    "final_passed": True,
                    "reason": "generation_not_run",
                },
                citation_evaluation=self._empty_citation_evaluation(
                    reason="generation_not_run"
                ),
                claim_validation=self._empty_claim_validation(
                    reason="generation_not_run"
                ),
                claims=[],
                timings_ms={
                    "retrieval": self._milliseconds(
                        retrieval_started,
                        retrieval_finished,
                    ),
                    "context_building": 0.0,
                    "generation": 0.0,
                    "evidence_building": 0.0,
                    "citation_repair": 0.0,
                    "claim_validation": 0.0,
                    "citation_evaluation": 0.0,
                    "total": self._milliseconds(
                        total_started,
                        total_finished,
                    ),
                },
            )

        # ── Controlled Response Variation ────────────────────────────────
        variation_profiles = ["direct_answer", "key_point_first", "practical_structure", "educational_explanation"]
        profile_temperatures = {
            "direct_answer": 0.22,
            "key_point_first": 0.28,
            "practical_structure": 0.32,
            "educational_explanation": 0.36,
        }
        seed_hash = abs(hash(f"{normalized_question}_{total_started}")) % len(variation_profiles)
        variation_profile = variation_profiles[seed_hash]

        effective_temp = temperature
        effective_top_p = None
        if settings.RAG_ENABLE_RESPONSE_VARIATION and temperature == 0.0:
            effective_temp = profile_temperatures.get(variation_profile, settings.RAG_DEFAULT_TEMPERATURE)
            effective_top_p = settings.RAG_TOP_P

        context_started = perf_counter()
        context = self._context_builder.build(results)
        prompt = self._prompt_builder.build(
            question=normalized_question,
            context=context,
            query_understanding=understood_query,
            conversation_history=cleaned_history,
            response_language=response_language,
            variation_profile=variation_profile,
        )
        context_finished = perf_counter()

        generation_started = perf_counter()
        uncertainty_instruction = (
            self._evidence_strength_classifier.prompt_instruction(
                evidence_strength["level"],
                response_language,
            )
        )
        generation = await self._generation_provider.generate(
            system_prompt=(
                prompt.system_prompt
                + "\n\nEVIDENCE STRENGTH LANGUAGE POLICY:\n"
                + uncertainty_instruction
            ),
            user_prompt=prompt.user_prompt,
            temperature=effective_temp,
            max_output_tokens=max_output_tokens,
            top_p=effective_top_p,
        )
        generation_finished = perf_counter()


        raw_generated_answer = self._normalize_source_citations(
            generation.text.strip()
        )
        refusal_category, clean_generated_answer = (
            RefusalPolicy.parse_marked_answer(raw_generated_answer)
        )
        generation_refused = refusal_category is not None
        if generation_refused:
            raw_generated_answer = clean_generated_answer

        citation_repair_started = perf_counter()
        if generation_refused:
            generated_answer = raw_generated_answer
            citation_repair = {
                "attempted": False,
                "repaired": False,
                "initial_passed": True,
                "final_passed": True,
                "reason": "generation_refusal",
            }
        else:
            repair_result = (
                await self._citation_repair_service.repair_if_needed(
                    raw_generated_answer,
                    sources=context.sources,
                    refusal_sentences=refusal_sentences,
                    response_language=response_language,
                )
            )
            generated_answer = self._normalize_source_citations(
                repair_result.answer
            )
            citation_repair = {
                "attempted": repair_result.attempted,
                "repaired": repair_result.repaired,
                "initial_passed": (
                    repair_result.initial_decision.passed
                ),
                "final_passed": repair_result.final_decision.passed,
                "reason": repair_result.final_decision.reason,
            }
        citation_repair_finished = perf_counter()

        citation_compliance_failed = (
            not generation_refused
            and not citation_repair["final_passed"]
        )
        used_sources = (
            []
            if generation_refused or citation_compliance_failed
            else self._select_sources(
                generated_answer,
                context.sources,
            )
        )

        evidence_started = perf_counter()
        evidence_items = (
            []
            if generation_refused or citation_compliance_failed
            else self._evidence_builder.build(
                used_sources=used_sources,
                retrieval_results=results,
            )
        )
        evidence = [asdict(item) for item in evidence_items]
        evidence_finished = perf_counter()

        claim_validation_started = perf_counter()
        extracted_claims = []
        support_results = []
        if generation_refused or citation_compliance_failed:
            claims: list[dict[str, Any]] = []
            claim_results: list[dict[str, Any]] = []
            claim_validation = self._empty_claim_validation(
                reason=(
                    "generation_refusal"
                    if generation_refused
                    else "citation_repair_failed"
                )
            )
        else:
            extracted_claims = self._claim_extractor.extract(
                generated_answer,
                refusal_sentences=refusal_sentences,
            )
            support_results = (
                await self._claim_support_evaluator.evaluate(
                    extracted_claims,
                    evidence=evidence,
                )
            )

            recovery_generation_attempted = False
            recovery_generation_passed = False
            if support_results and not any(result.supported for result in support_results):
                recovery_generation_attempted = True
                recovered = await self._recover_all_unsupported_claims(
                    question=normalized_question,
                    context=context,
                    results=results,
                    query_understanding=understood_query,
                    conversation_history=cleaned_history,
                    response_language=response_language,
                    refusal_sentences=refusal_sentences,
                    max_output_tokens=max_output_tokens,
                )
                if recovered is not None:
                    generated_answer, used_sources, evidence, extracted_claims, support_results, citation_repair = recovered
                    recovery_generation_passed = True
            has_supported_claims = any(result.supported for result in support_results)
            has_unsupported_claims = any(not result.supported for result in support_results)
            claims_pruned = False

            if has_supported_claims and has_unsupported_claims:
                pruned_answer = self._unsupported_claim_pruner.prune(
                    extracted_claims,
                    support_results,
                )

                if pruned_answer:
                    generated_answer = pruned_answer
                    claims_pruned = True
                    claims_rebuilt = False
                    claim_rebuild_error = None

                    if self._supported_answer_rebuilder is not None:
                        try:
                            rebuilt_answer = (
                                await self._supported_answer_rebuilder.rebuild(
                                    question=normalized_question,
                                    supported_answer=pruned_answer,
                                    response_language=response_language,
                                    variation_profile=variation_profile,
                                )
                            )
                            generated_answer = self._normalize_source_citations(
                                rebuilt_answer
                            )
                            claims_rebuilt = True

                        except Exception as exc:
                            claim_rebuild_error = (
                                f"{type(exc).__name__}: {exc}"
                            )
                            generated_answer = pruned_answer

                    used_sources = self._select_sources(
                        generated_answer,
                        context.sources,
                    )
                    evidence_items = self._evidence_builder.build(
                        used_sources=used_sources,
                        retrieval_results=results,
                    )
                    evidence = [
                        asdict(item)
                        for item in evidence_items
                    ]

                    extracted_claims = self._claim_extractor.extract(
                        generated_answer,
                        refusal_sentences=refusal_sentences,
                    )
                    support_results = (
                        await self._claim_support_evaluator.evaluate(
                            extracted_claims,
                            evidence=evidence,
                        )
                    )

            decision = self._post_generation_safety_gate.evaluate(
                support_results
            )
            claims = [asdict(claim) for claim in extracted_claims]
            claim_results = [
                asdict(result)
                for result in support_results
            ]
            claim_validation = asdict(decision)
            claim_validation["claims_pruned"] = claims_pruned
            claim_validation["claims_rebuilt"] = (
                claims_rebuilt if claims_pruned else False
            )
            claim_validation["claim_rebuild_error"] = (
                claim_rebuild_error if claims_pruned else None
            )
            claim_validation["all_claims_unsupported_initially"] = recovery_generation_attempted
            claim_validation["recovery_generation_attempted"] = recovery_generation_attempted
            claim_validation["recovery_generation_passed"] = recovery_generation_passed
        claim_validation_finished = perf_counter()

        citation_evaluation_started = perf_counter()
        if generation_refused:
            citation_evaluation = self._empty_citation_evaluation(
                reason="generation_refusal"
            )
        elif citation_compliance_failed:
            citation_evaluation = self._empty_citation_evaluation(
                reason="citation_repair_failed"
            )
        else:
            citation_report = (
                self._citation_accuracy_evaluator.evaluate(
                    claims=extracted_claims,
                    claim_results=support_results,
                    sources=used_sources,
                    evidence=evidence,
                )
            )
            citation_evaluation = asdict(citation_report)
        citation_evaluation_finished = perf_counter()

        refusal_guidance: dict[str, Any] | None = None
        if generation_refused:
            answer = generated_answer
            refused = True
            grounded = False
            refusal = {
                "reason": refusal_category or "generation_refusal",
                "stage": "generation",
                "generation_skipped": False,
            }
            refusal_guidance = {
                "category": refusal_category or "generation_refusal",
                "requires_professional": refusal_category in {
                    "professional_care",
                    "personalized_treatment",
                    "urgent_help",
                },
                "urgent": refusal_category == "urgent_help",
            }
            safety_flagged = False
            returned_sources: list[dict[str, Any]] = []
            returned_evidence: list[dict[str, Any]] = []
        elif citation_compliance_failed:
            answer = self._post_generation_refusal(response_language)
            refused = True
            grounded = False
            refusal = {
                "reason": "citation_repair_failed",
                "stage": "post_generation",
                "generation_skipped": False,
            }
            safety_flagged = True
            returned_sources = []
            returned_evidence = []
        elif not claim_validation["passed"]:
            answer = self._post_generation_refusal(response_language)
            refused = True
            grounded = False
            refusal = {
                "reason": claim_validation["reason"],
                "stage": "post_generation",
                "generation_skipped": False,
            }
            safety_flagged = True
            returned_sources = []
            returned_evidence = []
        elif not citation_evaluation["passed"]:
            answer = self._post_generation_refusal(response_language)
            refused = True
            grounded = False
            refusal = {
                "reason": "citation_accuracy_failed",
                "stage": "post_generation",
                "generation_skipped": False,
            }
            safety_flagged = True
            returned_sources = []
            returned_evidence = []
        else:
            answer = self._prepare_final_medical_answer(
                generated_answer,
                response_language,
            )
            refused = False
            grounded = bool(evidence)
            refusal = None
            safety_flagged = False
            returned_sources = used_sources
            returned_evidence = evidence

        total_finished = perf_counter()
        return {
            "question": normalized_question,
            "answer": answer,
            "recommendation": answer,
            "answer_language": response_language,
            "grounded": grounded,
            "refused": refused,
            "safety_flagged": safety_flagged,
            "refusal": refusal,
            "refusal_guidance": refusal_guidance,
            "relevance": relevance,
            "evidence_strength": evidence_strength,
            "provider": generation.provider,
            "model": generation.model,
            "request_id": generation.request_id,
            "sources": returned_sources,
            "evidence": returned_evidence,
            "claims": claims,
            "claim_results": claim_results,
            "citation_repair": citation_repair,
            "claim_validation": claim_validation,
            "citation_evaluation": citation_evaluation,
            "retrieval": retrieval,
            "context_characters": context.total_characters,
            "generation_config": {
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            },
            "timings_ms": {
                "retrieval": self._milliseconds(
                    retrieval_started,
                    retrieval_finished,
                ),
                "context_building": self._milliseconds(
                    context_started,
                    context_finished,
                ),
                "generation": self._milliseconds(
                    generation_started,
                    generation_finished,
                ),
                "evidence_building": self._milliseconds(
                    evidence_started,
                    evidence_finished,
                ),
                "citation_repair": self._milliseconds(
                    citation_repair_started,
                    citation_repair_finished,
                ),
                "claim_validation": self._milliseconds(
                    claim_validation_started,
                    claim_validation_finished,
                ),
                "citation_evaluation": self._milliseconds(
                    citation_evaluation_started,
                    citation_evaluation_finished,
                ),
                "total": self._milliseconds(
                    total_started,
                    total_finished,
                ),
            },
        }

    @staticmethod
    def _empty_claim_validation(reason: str) -> dict[str, Any]:
        return {
            "passed": True,
            "reason": reason,
            "total_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": 0,
            "faithfulness": 1.0,
            "minimum_faithfulness": 0.90,
            "unsupported_claim_ids": (),
        }

    @staticmethod
    def _empty_citation_evaluation(reason: str) -> dict[str, Any]:
        return {
            "total_claims": 0,
            "cited_claims": 0,
            "uncited_claims": 0,
            "citation_completeness": 1.0,
            "total_citation_links": 0,
            "correct_citation_links": 0,
            "incorrect_citation_links": 0,
            "citation_accuracy": None,
            "unique_source_count": 0,
            "invalid_source_ids": (),
            "metadata_accuracy": None,
            "claim_support_accuracy": None,
            "passed": True,
            "minimum_citation_accuracy": 0.95,
            "minimum_citation_completeness": 1.0,
            "items": (),
            "reason": reason,
        }

    @staticmethod
    def _refusal_output(
        *,
        question: str,
        answer: str,
        language: str,
        retrieval: dict[str, Any],
        relevance: dict[str, Any],
        refusal_guidance: dict[str, Any] | None = None,
        evidence_strength: dict[str, Any],
        refusal_reason: str,
        refusal_stage: str,
        generation_skipped: bool,
        safety_flagged: bool,
        citation_repair: dict[str, Any],
        citation_evaluation: dict[str, Any],
        claim_validation: dict[str, Any],
        claims: list[dict[str, Any]],
        timings_ms: dict[str, float],
    ) -> dict[str, Any]:
        return {
            "question": question,
            "answer": answer,
            "recommendation": answer,
            "answer_language": language,
            "grounded": False,
            "refused": True,
            "safety_flagged": safety_flagged,
            "refusal": {
                "reason": refusal_reason,
                "stage": refusal_stage,
                "generation_skipped": generation_skipped,
            },
            "refusal_guidance": refusal_guidance,
            "relevance": relevance,
            "evidence_strength": evidence_strength,
            "provider": None,
            "model": None,
            "request_id": None,
            "sources": [],
            "evidence": [],
            "claims": claims,
            "claim_results": [],
            "citation_repair": citation_repair,
            "claim_validation": claim_validation,
            "citation_evaluation": citation_evaluation,
            "retrieval": retrieval,
            "context_characters": 0,
            "generation_config": None,
            "timings_ms": timings_ms,
        }

    async def ask_ephemeral_document(
        self,
        *,
        question: str,
        candidates: list[dict[str, Any]],
        conversation_history: list[dict[str, str]] | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 1200,
        session: AsyncSession | None = None,
        include_global_knowledge: bool = False,
        retrieval_limit: int = 5,
    ) -> dict[str, Any]:
        """Run a temporary document alone or merge it with the global knowledge base."""
        from src.services.EphemeralDocumentService import EphemeralDocumentService

        total_started = perf_counter()
        normalized_question = self._normalize_question(question)
        cleaned_history = self._clean_conversation_history(conversation_history)
        understood_query = await self._query_understanding.understand(
            normalized_question,
            conversation_history=cleaned_history,
        )
        response_language = LanguageDetector.detect(normalized_question).code
        refusal_sentences = self._refusal_sentences()

        if understood_query.intent == "social":
            answer = (
                understood_query.clarification_message
                or self._social_fallback(response_language)
            )
            return self._social_output(
                question=normalized_question,
                answer=answer,
                language=response_language,
                query_understanding=understood_query.as_dict(),
                total_ms=self._milliseconds(total_started, perf_counter()),
            )

        pre_reason = understood_query.safety_reason
        if pre_reason is None and understood_query.intent == "out_of_scope":
            pre_reason = "out_of_scope"
        if pre_reason is not None or understood_query.ambiguous:
            if understood_query.ambiguous:
                answer = (
                    understood_query.clarification_message
                    or RefusalPolicy.decision(
                        normalized_question,
                        reason="insufficient_evidence",
                    ).message
                )
                category = "insufficient_evidence"
                requires_professional = False
                urgent = False
            else:
                decision = RefusalPolicy.decision(
                    normalized_question,
                    reason=pre_reason,
                )
                answer = decision.message
                category = decision.reason
                requires_professional = decision.requires_professional
                urgent = decision.urgent

            finished = perf_counter()
            retrieval = {
                "results": [],
                "pre_dedup_count": 0,
                "post_dedup_count": 0,
                "removed_duplicates": [],
                "cross_language_keyword_used": False,
                "effective_keyword_query": None,
                "semantic_query": understood_query.semantic_query,
                "rerank_query": understood_query.semantic_query,
                "query_understanding": understood_query.as_dict(),
                "ephemeral_document": True,
            }
            relevance = {
                "passed": False,
                "reason": "pre_retrieval_policy_decision",
                "top_score": None,
                "second_score": None,
                "score_margin": None,
                "threshold": 0.320982,
                "qualified_chunk_count": 0,
                "minimum_qualified_chunks": 1,
            }
            strength = {
                "level": "insufficient",
                "top_score": None,
                "relevance_threshold": 0.320982,
                "strong_threshold": 0.533,
                "language_policy": "generation_not_run",
                "answer_allowed": False,
                "rationale": "Handled before retrieval by policy.",
            }
            return self._refusal_output(
                question=normalized_question,
                answer=answer,
                language=response_language,
                retrieval=retrieval,
                relevance=relevance,
                refusal_guidance={
                    "category": category,
                    "requires_professional": requires_professional,
                    "urgent": urgent,
                },
                evidence_strength=strength,
                refusal_reason=category,
                refusal_stage="pre_generation",
                generation_skipped=True,
                safety_flagged=urgent,
                citation_repair={
                    "attempted": False,
                    "repaired": False,
                    "initial_passed": True,
                    "final_passed": True,
                    "reason": "generation_not_run",
                },
                citation_evaluation=self._empty_citation_evaluation("generation_not_run"),
                claim_validation=self._empty_claim_validation("generation_not_run"),
                claims=[],
                timings_ms={
                    "retrieval": 0.0,
                    "context_building": 0.0,
                    "generation": 0.0,
                    "evidence_building": 0.0,
                    "citation_repair": 0.0,
                    "claim_validation": 0.0,
                    "citation_evaluation": 0.0,
                    "total": self._milliseconds(total_started, finished),
                },
            )

        retrieval_started = perf_counter()
        ranker = EphemeralDocumentService()
        ranked_candidates = ranker.rank_chunks_for_query(
            chunks=candidates,
            query=understood_query.semantic_query,
            limit=8,
        )
        retrieval_finished = perf_counter()

        positive_candidates = [
            item for item in ranked_candidates
            if float(item.get("rerank_score") or 0.0) > 0.0
        ]
        uploaded_results = positive_candidates or ranked_candidates
        for item in uploaded_results:
            item["retrieval_scope"] = "uploaded_document"
            item["ephemeral_document"] = True

        global_retrieval: dict[str, Any] | None = None
        global_results: list[dict[str, Any]] = []
        if include_global_knowledge:
            if session is None:
                raise ValueError(
                    "session is required when global knowledge is included"
                )
            global_retrieval = await self._retrieval_pipeline.search(
                session=session,
                query=understood_query.semantic_query,
                limit=max(retrieval_limit, 5),
                project_id=None,
                asset_id=None,
                use_deduplication=True,
                use_reranking=True,
                semantic_query=understood_query.semantic_query,
                keyword_hints=understood_query.keyword_hints,
            )
            global_results = [dict(item) for item in global_retrieval["results"]]
            for item in global_results:
                item["retrieval_scope"] = "global_knowledge_base"
                item["ephemeral_document"] = False

        # Merge both evidence pools without storing the temporary document.
        # Keep unique asset/chunk pairs and unique temporary chunk IDs.
        merged: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        for item in global_results + uploaded_results:
            key = (
                item.get("retrieval_scope"),
                item.get("asset_id"),
                item.get("chunk_id"),
                str(item.get("text") or item.get("content") or "")[:160],
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)

        merged.sort(
            key=lambda item: float(item.get("rerank_score") or 0.0),
            reverse=True,
        )
        results = merged[: max(retrieval_limit, 5)]
        retrieval = {
            "results": results,
            "pre_dedup_count": (
                len(candidates)
                + (global_retrieval.get("pre_dedup_count", 0) if global_retrieval else 0)
            ),
            "post_dedup_count": len(merged),
            "removed_duplicates": [],
            "cross_language_keyword_used": (
                global_retrieval.get("cross_language_keyword_used", False)
                if global_retrieval else False
            ),
            "effective_keyword_query": (
                global_retrieval.get("effective_keyword_query")
                if global_retrieval else None
            ),
            "semantic_query": understood_query.semantic_query,
            "rerank_query": understood_query.semantic_query,
            "query_understanding": understood_query.as_dict(),
            "ephemeral_document": True,
            "combined_search": include_global_knowledge,
            "global_result_count": len(global_results),
            "uploaded_document_result_count": len(uploaded_results),
            "score_type": "mixed_global_rerank_and_uploaded_lexical",
        }

        relevance = self._relevance_gate.evaluate_as_dict(results)
        evidence_strength = self._evidence_strength_classifier.classify_as_dict(
            relevance.get("top_score")
        )
        if not results or not relevance["passed"]:
            decision = RefusalPolicy.decision(
                normalized_question,
                low_relevance=True,
                response_language=response_language,
            )
            finished = perf_counter()
            return self._refusal_output(
                question=normalized_question,
                answer=decision.message,
                language=response_language,
                retrieval=retrieval,
                relevance=relevance,
                refusal_guidance={
                    "category": decision.reason,
                    "requires_professional": decision.requires_professional,
                    "urgent": decision.urgent,
                },
                evidence_strength=evidence_strength,
                refusal_reason=decision.reason,
                refusal_stage="pre_generation",
                generation_skipped=True,
                safety_flagged=False,
                citation_repair={
                    "attempted": False,
                    "repaired": False,
                    "initial_passed": True,
                    "final_passed": True,
                    "reason": "generation_not_run",
                },
                citation_evaluation=self._empty_citation_evaluation("generation_not_run"),
                claim_validation=self._empty_claim_validation("generation_not_run"),
                claims=[],
                timings_ms={
                    "retrieval": self._milliseconds(retrieval_started, retrieval_finished),
                    "context_building": 0.0,
                    "generation": 0.0,
                    "evidence_building": 0.0,
                    "citation_repair": 0.0,
                    "claim_validation": 0.0,
                    "citation_evaluation": 0.0,
                    "total": self._milliseconds(total_started, finished),
                },
            )

        variation_profiles = ["direct_answer", "key_point_first", "practical_structure", "educational_explanation"]
        profile_temperatures = {
            "direct_answer": 0.22,
            "key_point_first": 0.28,
            "practical_structure": 0.32,
            "educational_explanation": 0.36,
        }
        seed_hash = abs(hash(f"{normalized_question}_{total_started}")) % len(variation_profiles)
        variation_profile = variation_profiles[seed_hash]

        effective_temp = temperature
        effective_top_p = None
        if settings.RAG_ENABLE_RESPONSE_VARIATION and temperature == 0.0:
            effective_temp = profile_temperatures.get(variation_profile, settings.RAG_DEFAULT_TEMPERATURE)
            effective_top_p = settings.RAG_TOP_P

        context_started = perf_counter()
        context = self._context_builder.build(results)
        prompt = self._prompt_builder.build(
            question=normalized_question,
            context=context,
            query_understanding=understood_query,
            conversation_history=cleaned_history,
            response_language=response_language,
            variation_profile=variation_profile,
        )
        context_finished = perf_counter()

        generation_started = perf_counter()
        uncertainty_instruction = self._evidence_strength_classifier.prompt_instruction(
            evidence_strength["level"],
            response_language,
        )
        generation = await self._generation_provider.generate(
            system_prompt=(
                prompt.system_prompt
                + "\n\nEVIDENCE STRENGTH LANGUAGE POLICY:\n"
                + uncertainty_instruction
            ),
            user_prompt=prompt.user_prompt,
            temperature=effective_temp,
            max_output_tokens=max_output_tokens,
            top_p=effective_top_p,
        )
        generation_finished = perf_counter()


        raw_answer = self._normalize_source_citations(generation.text.strip())
        refusal_category, cleaned_answer = RefusalPolicy.parse_marked_answer(raw_answer)
        generation_refused = refusal_category is not None
        if generation_refused:
            raw_answer = cleaned_answer

        citation_repair_started = perf_counter()
        if generation_refused:
            generated_answer = raw_answer
            citation_repair = {
                "attempted": False,
                "repaired": False,
                "initial_passed": True,
                "final_passed": True,
                "reason": "generation_refusal",
            }
        else:
            repaired = await self._citation_repair_service.repair_if_needed(
                raw_answer,
                sources=context.sources,
                refusal_sentences=refusal_sentences,
                response_language=response_language,
            )
            generated_answer = self._normalize_source_citations(repaired.answer)
            citation_repair = {
                "attempted": repaired.attempted,
                "repaired": repaired.repaired,
                "initial_passed": repaired.initial_decision.passed,
                "final_passed": repaired.final_decision.passed,
                "reason": repaired.final_decision.reason,
            }
        citation_repair_finished = perf_counter()
        citation_compliance_failed = (
            not generation_refused and not citation_repair["final_passed"]
        )

        used_sources = (
            [] if generation_refused or citation_compliance_failed
            else self._select_sources(generated_answer, context.sources)
        )
        evidence_started = perf_counter()
        evidence_items = (
            [] if generation_refused or citation_compliance_failed
            else self._evidence_builder.build(
                used_sources=used_sources,
                retrieval_results=results,
            )
        )
        evidence = [asdict(item) for item in evidence_items]
        evidence_finished = perf_counter()

        claim_validation_started = perf_counter()
        extracted_claims = []
        support_results = []
        claims_pruned = False
        claims_rebuilt = False
        claim_rebuild_error = None
        if generation_refused or citation_compliance_failed:
            claims = []
            claim_results = []
            claim_validation = self._empty_claim_validation(
                "generation_refusal" if generation_refused else "citation_repair_failed"
            )
        else:
            extracted_claims = self._claim_extractor.extract(
                generated_answer,
                refusal_sentences=refusal_sentences,
            )
            support_results = await self._claim_support_evaluator.evaluate(
                extracted_claims,
                evidence=evidence,
            )
            recovery_generation_attempted = False
            recovery_generation_passed = False
            if support_results and not any(item.supported for item in support_results):
                recovery_generation_attempted = True
                recovered = await self._recover_all_unsupported_claims(
                    question=normalized_question,
                    context=context,
                    results=results,
                    query_understanding=understood_query,
                    conversation_history=cleaned_history,
                    response_language=response_language,
                    refusal_sentences=refusal_sentences,
                    max_output_tokens=max_output_tokens,
                )
                if recovered is not None:
                    generated_answer, used_sources, evidence, extracted_claims, support_results, citation_repair = recovered
                    recovery_generation_passed = True
            has_supported = any(item.supported for item in support_results)
            has_unsupported = any(not item.supported for item in support_results)
            if has_supported and has_unsupported:
                pruned = self._unsupported_claim_pruner.prune(
                    extracted_claims,
                    support_results,
                )
                if pruned:
                    generated_answer = pruned
                    claims_pruned = True
                    if self._supported_answer_rebuilder is not None:
                        try:
                            generated_answer = self._normalize_source_citations(
                                await self._supported_answer_rebuilder.rebuild(
                                    question=normalized_question,
                                    supported_answer=pruned,
                                    response_language=response_language,
                                    variation_profile=variation_profile,
                                )
                            )
                            claims_rebuilt = True

                        except Exception as exc:
                            claim_rebuild_error = f"{type(exc).__name__}: {exc}"
                            generated_answer = pruned
                    used_sources = self._select_sources(generated_answer, context.sources)
                    evidence_items = self._evidence_builder.build(
                        used_sources=used_sources,
                        retrieval_results=results,
                    )
                    evidence = [asdict(item) for item in evidence_items]
                    extracted_claims = self._claim_extractor.extract(
                        generated_answer,
                        refusal_sentences=refusal_sentences,
                    )
                    support_results = await self._claim_support_evaluator.evaluate(
                        extracted_claims,
                        evidence=evidence,
                    )

            validation_decision = self._post_generation_safety_gate.evaluate(
                support_results
            )
            claims = [asdict(item) for item in extracted_claims]
            claim_results = [asdict(item) for item in support_results]
            claim_validation = asdict(validation_decision)
            claim_validation["claims_pruned"] = claims_pruned
            claim_validation["claims_rebuilt"] = claims_rebuilt
            claim_validation["claim_rebuild_error"] = claim_rebuild_error
            claim_validation["all_claims_unsupported_initially"] = recovery_generation_attempted
            claim_validation["recovery_generation_attempted"] = recovery_generation_attempted
            claim_validation["recovery_generation_passed"] = recovery_generation_passed
        claim_validation_finished = perf_counter()

        citation_evaluation_started = perf_counter()
        if generation_refused:
            citation_evaluation = self._empty_citation_evaluation("generation_refusal")
        elif citation_compliance_failed:
            citation_evaluation = self._empty_citation_evaluation("citation_repair_failed")
        else:
            citation_evaluation = asdict(
                self._citation_accuracy_evaluator.evaluate(
                    claims=extracted_claims,
                    claim_results=support_results,
                    sources=used_sources,
                    evidence=evidence,
                )
            )
        citation_evaluation_finished = perf_counter()

        refusal_guidance = None
        if generation_refused:
            answer = generated_answer
            refused = True
            grounded = False
            refusal = {
                "reason": refusal_category or "generation_refusal",
                "stage": "generation",
                "generation_skipped": False,
            }
            refusal_guidance = {
                "category": refusal_category or "generation_refusal",
                "requires_professional": refusal_category in {
                    "professional_care", "personalized_treatment", "urgent_help"
                },
                "urgent": refusal_category == "urgent_help",
            }
            safety_flagged = False
            returned_sources = []
            returned_evidence = []
        elif citation_compliance_failed:
            answer = self._post_generation_refusal(response_language)
            refused = True
            grounded = False
            refusal = {
                "reason": "citation_repair_failed",
                "stage": "post_generation",
                "generation_skipped": False,
            }
            safety_flagged = True
            returned_sources = []
            returned_evidence = []
        elif not claim_validation["passed"]:
            answer = self._post_generation_refusal(response_language)
            refused = True
            grounded = False
            refusal = {
                "reason": claim_validation["reason"],
                "stage": "post_generation",
                "generation_skipped": False,
            }
            safety_flagged = True
            returned_sources = []
            returned_evidence = []
        elif not citation_evaluation["passed"]:
            answer = self._post_generation_refusal(response_language)
            refused = True
            grounded = False
            refusal = {
                "reason": "citation_accuracy_failed",
                "stage": "post_generation",
                "generation_skipped": False,
            }
            safety_flagged = True
            returned_sources = []
            returned_evidence = []
        else:
            answer = self._prepare_final_medical_answer(
                generated_answer,
                response_language,
            )
            refused = False
            grounded = bool(evidence)
            refusal = None
            safety_flagged = False
            returned_sources = used_sources
            returned_evidence = evidence

        finished = perf_counter()
        return {
            "question": normalized_question,
            "answer": answer,
            "recommendation": answer,
            "answer_language": response_language,
            "grounded": grounded,
            "refused": refused,
            "safety_flagged": safety_flagged,
            "refusal": refusal,
            "refusal_guidance": refusal_guidance,
            "relevance": relevance,
            "evidence_strength": evidence_strength,
            "provider": generation.provider,
            "model": generation.model,
            "request_id": generation.request_id,
            "sources": returned_sources,
            "evidence": returned_evidence,
            "claims": claims,
            "claim_results": claim_results,
            "citation_repair": citation_repair,
            "citation_evaluation": citation_evaluation,
            "claim_validation": claim_validation,
            "retrieval": retrieval,
            "context_characters": context.total_characters,
            "generation_config": {
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            },
            "timings_ms": {
                "retrieval": self._milliseconds(retrieval_started, retrieval_finished),
                "context_building": self._milliseconds(context_started, context_finished),
                "generation": self._milliseconds(generation_started, generation_finished),
                "evidence_building": self._milliseconds(evidence_started, evidence_finished),
                "citation_repair": self._milliseconds(citation_repair_started, citation_repair_finished),
                "claim_validation": self._milliseconds(claim_validation_started, claim_validation_finished),
                "citation_evaluation": self._milliseconds(citation_evaluation_started, citation_evaluation_finished),
                "total": self._milliseconds(total_started, finished),
            },
        }

    async def close(self) -> None:
        await self._retrieval_pipeline.close()
        await self._generation_provider.close()
        if self._claim_judge_provider is not self._generation_provider:
            await self._claim_judge_provider.close()

    