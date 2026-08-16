import asyncio
import sys
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.helpers.config import settings
from src.stores.vectordb.providers.PGVector_Provider import PGVector_Provider
from src.stores.vectordb.VectorDBInterface import VectorDocument


def create_vector(dimension: int, first: float, second: float) -> list[float]:
    """Create a deterministic vector with the configured dimension."""
    vector = [0.0] * dimension
    vector[0] = first
    vector[1] = second
    return vector


async def run_test() -> None:
    test_prefix = f"vector-test-{uuid4().hex[:8]}"
    test_ids = [
        f"{test_prefix}-medical",
        f"{test_prefix}-cardiology",
        f"{test_prefix}-nutrition",
    ]

    provider = PGVector_Provider(
        database_url=settings.POSTGRES_URL,
        embedding_dimension=settings.EMBEDDING_MODEL_SIZE,
        table_name=settings.VECTOR_DB_TABLE,
        min_pool_size=settings.VECTOR_DB_MIN_POOL_SIZE,
        max_pool_size=settings.VECTOR_DB_MAX_POOL_SIZE,
    )

    try:
        print("[1/6] Initializing pgvector provider...")
        await provider.initialize()

        print("[2/6] Checking database health...")
        is_healthy = await provider.health_check()
        assert is_healthy, "Vector database health check failed"

        print("[3/6] Inserting test vectors...")
        documents = [
            VectorDocument(
                id=test_ids[0],
                text="Metformin is commonly used in diabetes management.",
                embedding=create_vector(
                    settings.EMBEDDING_MODEL_SIZE,
                    first=1.0,
                    second=0.0,
                ),
                metadata={
                    "test_run": test_prefix,
                    "topic": "diabetes",
                    "page_number": 1,
                },
            ),
            VectorDocument(
                id=test_ids[1],
                text="Blood pressure management is important in cardiology.",
                embedding=create_vector(
                    settings.EMBEDDING_MODEL_SIZE,
                    first=0.8,
                    second=0.2,
                ),
                metadata={
                    "test_run": test_prefix,
                    "topic": "cardiology",
                    "page_number": 2,
                },
            ),
            VectorDocument(
                id=test_ids[2],
                text="Balanced nutrition supports general health.",
                embedding=create_vector(
                    settings.EMBEDDING_MODEL_SIZE,
                    first=0.0,
                    second=1.0,
                ),
                metadata={
                    "test_run": test_prefix,
                    "topic": "nutrition",
                    "page_number": 3,
                },
            ),
        ]
        await provider.upsert(documents)

        print("[4/6] Running cosine similarity search...")
        query_vector = create_vector(
            settings.EMBEDDING_MODEL_SIZE,
            first=1.0,
            second=0.0,
        )
        results = await provider.similarity_search(
            query_embedding=query_vector,
            limit=3,
            filters={"test_run": test_prefix},
        )

        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        assert results[0].id == test_ids[0], (
            f"Unexpected top result: {results[0].id}"
        )
        assert results[0].score > 0.99, (
            f"Unexpected top similarity score: {results[0].score}"
        )

        print("[5/6] Search results:")
        for position, result in enumerate(results, start=1):
            print(
                f"  {position}. id={result.id} "
                f"score={result.score:.6f} "
                f"topic={result.metadata.get('topic')}"
            )

        print("[6/6] Cleaning test data...")
        deleted_rows = await provider.delete_by_ids(test_ids)
        assert deleted_rows == 3, f"Expected to delete 3 rows, got {deleted_rows}"

        print("SUCCESS: pgvector integration test passed.")

    finally:
        # Cleanup is safe even when an earlier assertion fails.
        try:
            await provider.delete_by_ids(test_ids)
        except Exception:
            pass
        await provider.close()


if __name__ == "__main__":
    asyncio.run(run_test())
