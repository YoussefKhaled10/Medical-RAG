from src.helpers.config import settings
from src.services.CandidateDeduplicator import CandidateDeduplicator
from src.services.CohereReranker import CohereReranker
from src.services.RetrievalPipelineService import RetrievalPipelineService
from src.services.hybrid_retrieval_factory import create_hybrid_retrieval_service


def create_retrieval_pipeline_service() -> RetrievalPipelineService:
    hybrid_service = create_hybrid_retrieval_service()
    deduplicator = CandidateDeduplicator(
        token_jaccard_threshold=0.90,
        minimum_tokens_for_similarity=20,
    )
    reranker = CohereReranker(
        api_key=settings.COHERE_API_KEY,
        model_name=getattr(
            settings,
            "COHERE_RERANK_MODEL",
            "rerank-v4.0-pro",
        ),
        max_tokens_per_doc=4096,
    )
    return RetrievalPipelineService(
        hybrid_service=hybrid_service,
        deduplicator=deduplicator,
        reranker=reranker,
        fused_candidate_limit=20,
    )
