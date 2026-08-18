from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src.models import ChunkModel
from src.services.CrossLanguageKeywordTranslator import (
    CrossLanguageKeywordTranslator,
)
from src.services.QueryRewriter import QueryRewriter, RewrittenQuery
from src.stores.llm.LLMInterface import LLMInterface
from src.stores.vectordb.VectorDBInterface import VectorDBInterface


class SearchType(StrEnum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


@dataclass(slots=True)
class _Candidate:
    key: str
    asset_id: int
    project_id: int | None
    chunk_id: str
    document_name: str
    section_title: str
    page_number: int
    text: str
    semantic_rank: int | None = None
    semantic_score: float | None = None
    keyword_rank: int | None = None
    keyword_score: float | None = None
    rrf_score: float = 0.0


class HybridRetrievalService:
    def __init__(
        self,
        embedding_provider: LLMInterface,
        vector_db: VectorDBInterface,
        semantic_fetch_limit: int = 20,
        keyword_fetch_limit: int = 20,
        rrf_k: int = 60,
        query_rewriter: QueryRewriter | None = None,
        keyword_translator: CrossLanguageKeywordTranslator | None = None,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_db = vector_db
        self._semantic_fetch_limit = semantic_fetch_limit
        self._keyword_fetch_limit = keyword_fetch_limit
        self._rrf_k = rrf_k
        self._query_rewriter = query_rewriter or QueryRewriter()
        self._keyword_translator = keyword_translator

    async def _semantic_search(
        self, query: str, project_id: int | None, asset_id: int | None
    ) -> list[_Candidate]:
        filters: dict[str, Any] = {}
        if project_id is not None:
            filters["project_id"] = project_id
        if asset_id is not None:
            filters["asset_id"] = asset_id

        await self._vector_db.initialize()
        embedding = await self._embedding_provider.embed_query(query)
        results = await self._vector_db.similarity_search(
            query_embedding=embedding,
            limit=self._semantic_fetch_limit,
            filters=filters or None,
        )

        candidates: list[_Candidate] = []
        for rank, result in enumerate(results, start=1):
            metadata = result.metadata or {}
            result_asset_id = metadata.get("asset_id")
            chunk_id = metadata.get("chunk_id")
            if result_asset_id is None or not chunk_id:
                continue
            candidates.append(_Candidate(
                key=f"{result_asset_id}:{chunk_id}",
                asset_id=int(result_asset_id),
                project_id=metadata.get("project_id"),
                chunk_id=str(chunk_id),
                document_name=str(metadata.get("document_name") or ""),
                section_title=str(metadata.get("section_title") or ""),
                page_number=int(metadata.get("page_number") or 1),
                text=result.text,
                semantic_rank=rank,
                semantic_score=float(result.score),
            ))
        return candidates

    async def _keyword_search(
        self,
        session: AsyncSession,
        query: str,
        project_id: int | None,
        asset_id: int | None,
    ) -> list[_Candidate]:
        rows = await ChunkModel.keyword_search(
            session=session,
            query=query,
            limit=self._keyword_fetch_limit,
            project_id=project_id,
            asset_id=asset_id,
        )
        return [
            _Candidate(
                key=f"{row['asset_id']}:{row['chunk_id']}",
                asset_id=row["asset_id"],
                project_id=row.get("project_id"),
                chunk_id=row["chunk_id"],
                document_name=row["document_name"],
                section_title=row["section_title"],
                page_number=row["page_number"],
                text=row["text"],
                keyword_rank=rank,
                keyword_score=float(row["keyword_score"]),
            )
            for rank, row in enumerate(rows, start=1)
        ]

    def _fuse(
        self,
        semantic: list[_Candidate],
        keyword: list[_Candidate],
    ) -> list[_Candidate]:
        fused: dict[str, _Candidate] = {}
        for candidate in semantic:
            candidate.rrf_score = 1.0 / (
                self._rrf_k + candidate.semantic_rank
            )
            fused[candidate.key] = candidate

        for candidate in keyword:
            contribution = 1.0 / (
                self._rrf_k + candidate.keyword_rank
            )
            if candidate.key in fused:
                existing = fused[candidate.key]
                existing.keyword_rank = candidate.keyword_rank
                existing.keyword_score = candidate.keyword_score
                existing.rrf_score += contribution
            else:
                candidate.rrf_score = contribution
                fused[candidate.key] = candidate

        return sorted(
            fused.values(),
            key=lambda item: (
                item.rrf_score,
                item.semantic_score or 0.0,
                item.keyword_score or 0.0,
            ),
            reverse=True,
        )

    @staticmethod
    def _to_output(candidate: _Candidate, rank: int) -> dict[str, Any]:
        return {
            "rank": rank,
            "asset_id": candidate.asset_id,
            "project_id": candidate.project_id,
            "chunk_id": candidate.chunk_id,
            "document_name": candidate.document_name,
            "section_title": candidate.section_title,
            "page_number": candidate.page_number,
            "text": candidate.text,
            "semantic_rank": candidate.semantic_rank,
            "semantic_score": (
                round(candidate.semantic_score, 6)
                if candidate.semantic_score is not None else None
            ),
            "keyword_rank": candidate.keyword_rank,
            "keyword_score": (
                round(candidate.keyword_score, 6)
                if candidate.keyword_score is not None else None
            ),
            "rrf_score": round(candidate.rrf_score, 8),
        }

    async def search(
        self,
        session: AsyncSession,
        query: str,
        search_type: SearchType = SearchType.HYBRID,
        limit: int = 5,
        project_id: int | None = None,
        asset_id: int | None = None,
        use_query_rewriting: bool = False,
        use_cross_language_keyword: bool = False,
    ) -> tuple[list[dict[str, Any]], RewrittenQuery, str]:
        rewritten = self._query_rewriter.rewrite(query)

        # Semantic retrieval always receives the original query. This preserves
        # the evaluated English baseline and the successful Arabic cross-lingual path.
        semantic_query = rewritten.original_query
        keyword_query = (
            rewritten.keyword_query
            if use_query_rewriting
            else rewritten.original_query
        )

        if (
            use_cross_language_keyword
            and self._keyword_translator is not None
        ):
            keyword_query = (
                await self._keyword_translator.translate_for_keyword_search(
                    keyword_query
                )
            )

        if search_type == SearchType.SEMANTIC:
            candidates = await self._semantic_search(
                semantic_query, project_id, asset_id
            )
            for item in candidates:
                item.rrf_score = 1.0 / (
                    self._rrf_k + item.semantic_rank
                )
        elif search_type == SearchType.KEYWORD:
            candidates = await self._keyword_search(
                session, keyword_query, project_id, asset_id
            )
            for item in candidates:
                item.rrf_score = 1.0 / (
                    self._rrf_k + item.keyword_rank
                )
        else:
            semantic = await self._semantic_search(
                semantic_query, project_id, asset_id
            )
            keyword = await self._keyword_search(
                session, keyword_query, project_id, asset_id
            )
            candidates = self._fuse(semantic, keyword)

        output = [
            self._to_output(item, rank)
            for rank, item in enumerate(candidates[:limit], start=1)
        ]
        return output, rewritten, keyword_query

    async def close(self) -> None:
        await self._embedding_provider.close()
        await self._vector_db.close()
        if self._keyword_translator is not None:
            await self._keyword_translator.close()
