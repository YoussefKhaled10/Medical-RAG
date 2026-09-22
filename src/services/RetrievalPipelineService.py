from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src.services.CandidateDeduplicator import CandidateDeduplicator
from src.services.CohereReranker import CohereReranker
from src.services.CrossLanguageKeywordTranslator import (
    CrossLanguageKeywordTranslator,
)
from src.services.HybridRetrievalService import (
    HybridRetrievalService,
    SearchType,
)


class RetrievalPipelineService:
    def __init__(
        self,
        hybrid_service: HybridRetrievalService,
        deduplicator: CandidateDeduplicator,
        reranker: CohereReranker,
        fused_candidate_limit: int = 20,
    ) -> None:
        self._hybrid_service = hybrid_service
        self._deduplicator = deduplicator
        self._reranker = reranker
        self._fused_candidate_limit = fused_candidate_limit

    async def search(
        self,
        session: AsyncSession,
        query: str,
        limit: int = 5,
        project_id: int | list[int] | tuple[int, ...] | None = None,
        asset_id: int | None = None,
        use_deduplication: bool = True,
        use_reranking: bool = True,
        use_cross_language_keyword: bool | None = None,
        semantic_query: str | None = None,
        keyword_hints: tuple[str, ...] | list[str] | None = None,
        search_type: SearchType | str = SearchType.HYBRID,
    ) -> dict[str, Any]:

        effective_semantic_query = " ".join(
            str(semantic_query or query).split()
        ).strip()
        if not effective_semantic_query:
            raise ValueError("semantic retrieval query must not be empty")

        if use_cross_language_keyword is None:
            use_cross_language_keyword = (
                CrossLanguageKeywordTranslator.contains_arabic(query)
            )

        try:
            effective_search_type = search_type if isinstance(search_type, SearchType) else SearchType(str(search_type).strip().lower())
        except ValueError as exc:
            raise ValueError("search_type must be semantic, keyword, or hybrid") from exc

        candidates, rewritten, effective_keyword_query = (
            await self._hybrid_service.search(
                session=session,
                query=query,
                search_type=effective_search_type,
                limit=self._fused_candidate_limit,
                project_id=project_id,
                asset_id=asset_id,
                use_query_rewriting=True,
                use_cross_language_keyword=use_cross_language_keyword,
                semantic_query=effective_semantic_query,
                keyword_hints=keyword_hints,
            )
        )

        # Retrieval near-deduplication is disabled by design. RRF still merges
        # the identical asset_id + chunk_id returned by both branches.
        pre_dedup_count = len(candidates)
        removed_duplicates: list[dict[str, Any]] = []
        post_dedup_count = pre_dedup_count
        rerank_query = effective_semantic_query
        reranking_fallback = False
        reranking_error = None
        if use_reranking:
            try:
                final_results = await self._reranker.rerank(
                    query=rerank_query,
                    candidates=candidates,
                    top_n=limit,
                )
            except Exception as exc:
                # Retrieval remains available when an external reranker is rate
                # limited or temporarily unavailable. RRF order is deterministic.
                reranking_fallback = True
                reranking_error = f"{type(exc).__name__}: {exc}"
                print(
                    "[RetrievalPipelineService] Cohere reranking fallback to RRF: "
                    f"{reranking_error}",
                    flush=True,
                )
                final_results = [dict(item) for item in candidates[:limit]]
                for rank, item in enumerate(final_results, start=1):
                    item["rank"] = rank
                    item["pre_rerank_rank"] = rank
                    item["rerank_score"] = None
        else:
            final_results = [dict(item) for item in candidates[:limit]]
            for rank, item in enumerate(final_results, start=1):
                item["rank"] = rank
                item["pre_rerank_rank"] = None
                item["rerank_score"] = None

        return {
            "results": final_results,
            "pre_dedup_count": pre_dedup_count,
            "post_dedup_count": post_dedup_count,
            "removed_duplicates": removed_duplicates,
            "cross_language_keyword_used": use_cross_language_keyword,
            "effective_keyword_query": effective_keyword_query,
            "original_query": rewritten.original_query,
            "semantic_query": rewritten.semantic_query,
            "query_expansions": list(rewritten.expansions),
            "rerank_query": rerank_query,
            "reranking_fallback": reranking_fallback,
            "reranking_error": reranking_error,
            "deduplication_enabled": False,
            "candidates_sent_to_reranker": len(candidates),
            **self._hybrid_service.last_diagnostics,
        }

    async def close(self) -> None:
        await self._hybrid_service.close()
        await self._reranker.close()
