import asyncio
import math

from src.helpers.config import settings
from src.stores.llm.LLMFactory import LLMFactory


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(value * value for value in vector_a))
    norm_b = math.sqrt(sum(value * value for value in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


async def run_test() -> None:
    provider = LLMFactory.create(
        provider=settings.EMBEDDING_BACKEND,
        api_key=settings.COHERE_API_KEY,
        model_name=settings.COHERE_EMBEDDING_MODEL,
        embedding_dimension=settings.EMBEDDING_MODEL_SIZE,
        batch_size=settings.COHERE_EMBEDDING_BATCH_SIZE,
        truncate=settings.COHERE_TRUNCATE,
    )

    try:
        print("[1/5] Checking Cohere configuration...")
        print(f"  backend: {settings.EMBEDDING_BACKEND}")
        print(f"  model: {settings.COHERE_EMBEDDING_MODEL}")
        print(f"  expected dimension: {settings.EMBEDDING_MODEL_SIZE}")

        print("[2/5] Checking Cohere API health...")
        is_healthy = await provider.health_check()
        assert is_healthy, (
            "Cohere health check failed. Check COHERE_API_KEY, model name, "
            "internet connection, and provider implementation."
        )

        print("[3/5] Creating document embeddings...")
        documents = [
            "Metformin is commonly used in diabetes management.",
            "Blood pressure management is important in cardiology.",
            "Balanced nutrition supports general health.",
        ]
        document_embeddings = await provider.embed_documents(documents)

        assert len(document_embeddings) == len(documents), (
            "Unexpected number of document embeddings"
        )
        assert all(
            len(vector) == settings.EMBEDDING_MODEL_SIZE
            for vector in document_embeddings
        ), "Unexpected document embedding dimension"

        print("[4/5] Creating query embedding...")
        query = "What medicine is commonly used for diabetes management?"
        query_embedding = await provider.embed_query(query)

        assert len(query_embedding) == settings.EMBEDDING_MODEL_SIZE, (
            "Unexpected query embedding dimension"
        )

        print("[5/5] Calculating similarities...")
        scores = [
            cosine_similarity(query_embedding, document_embedding)
            for document_embedding in document_embeddings
        ]

        ranked_results = sorted(
            zip(documents, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        for position, (document, score) in enumerate(ranked_results, start=1):
            print(f"  {position}. score={score:.6f} text={document}")

        assert ranked_results[0][0] == documents[0], (
            "The diabetes document was expected to be the most relevant result"
        )

        print("SUCCESS: Cohere embedding integration test passed.")

    finally:
        await provider.close()


if __name__ == "__main__":
    asyncio.run(run_test())
