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
        project_id: int | None = None,
        asset_id: int | None = None,
        use_deduplication: bool = True,
        use_reranking: bool = True,
        use_cross_language_keyword: bool | None = None,
    ) -> dict[str, Any]:
        # Auto-enable only for Arabic. English remains on the exact evaluated path.
        if use_cross_language_keyword is None:
            use_cross_language_keyword = (
                CrossLanguageKeywordTranslator.contains_arabic(query)
            )

        candidates, _, effective_keyword_query = (
            await self._hybrid_service.search(
                session=session,
                query=query,
                search_type=SearchType.HYBRID,
                limit=self._fused_candidate_limit,
                project_id=project_id,
                asset_id=asset_id,
                use_query_rewriting=False,
                use_cross_language_keyword=use_cross_language_keyword,
            )
        )

        pre_dedup_count = len(candidates)
        removed_duplicates: list[dict[str, Any]] = []
        if use_deduplication:
            candidates, removed_duplicates = (
                self._deduplicator.deduplicate(candidates)
            )

        post_dedup_count = len(candidates)
        if use_reranking:
            final_results = await self._reranker.rerank(
                query=query,
                candidates=candidates,
                top_n=limit,
            )
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
        }

    async def close(self) -> None:
        await self._hybrid_service.close()
        await self._reranker.close()
