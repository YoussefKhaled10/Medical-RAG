import asyncio
from typing import Any

import cohere


class CohereReranker:
    """Second-stage reranker backed by Cohere Rerank v2."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "rerank-v3.5",
        max_tokens_per_doc: int = 4096,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Cohere API key must not be empty")
        if not model_name.strip():
            raise ValueError("Rerank model name must not be empty")
        if max_tokens_per_doc <= 0:
            raise ValueError("max_tokens_per_doc must be greater than zero")

        self._client = cohere.ClientV2(api_key=api_key)
        self._model_name = model_name
        self._max_tokens_per_doc = max_tokens_per_doc

    @staticmethod
    def _document_text(candidate: dict[str, Any]) -> str:
        section = str(candidate.get("section_title") or "").strip()
        text = str(candidate.get("text") or "").strip()
        return f"Section: {section}\nContent: {text}".strip()

    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        if not query.strip():
            raise ValueError("query must not be empty")

        top_n = max(1, min(top_n, len(candidates)))
        documents = [self._document_text(item) for item in candidates]

        response = await asyncio.to_thread(
            self._client.rerank,
            model=self._model_name,
            query=query.strip(),
            documents=documents,
            top_n=top_n,
            max_tokens_per_doc=self._max_tokens_per_doc,
        )

        output: list[dict[str, Any]] = []
        for final_rank, rerank_result in enumerate(response.results, start=1):
            candidate = dict(candidates[rerank_result.index])
            candidate["pre_rerank_rank"] = candidate.get("rank")
            candidate["rank"] = final_rank
            candidate["rerank_score"] = round(
                float(rerank_result.relevance_score),
                6,
            )
            output.append(candidate)
        return output

    async def close(self) -> None:
        return None
