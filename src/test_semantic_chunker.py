import asyncio

from src.chunkers import SemanticChunker
from src.schemas.ingestion import DocumentSection, ParsedElement
from src.stores.llm.LLMFactory import LLMFactory
from src.helpers.config import settings

async def main() -> None:
    embedding_provider = LLMFactory.create(
        provider=settings.EMBEDDING_BACKEND,
        api_key=settings.COHERE_API_KEY,
        model_name=settings.COHERE_EMBEDDING_MODEL,
        embedding_dimension=settings.EMBEDDING_MODEL_SIZE,
        batch_size=settings.COHERE_EMBEDDING_BATCH_SIZE,
        truncate=settings.COHERE_TRUNCATE,
    )

    section = DocumentSection(
        document_name="Alcohol-use disorders",
        section_title="Contents",
        elements=[
            ParsedElement(
                element_index=0,
                text="Validated questionnaires help identify harmful alcohol use.",
                category="NarrativeText",
                page_number=2,
            ),
            ParsedElement(
                element_index=1,
                text="Community support networks can assist recovery.",
                category="NarrativeText",
                page_number=2,
            ),
            ParsedElement(
                element_index=2,
                text="Quality measures describe structure, process and outcomes.",
                category="NarrativeText",
                page_number=3,
            ),
        ],
    )

    chunker = SemanticChunker(
        embedding_provider=embedding_provider,
        similarity_threshold=0.55,
        minimum_tokens=5,
        target_tokens=15,
        maximum_tokens=50,
    )

    try:
        chunks = await chunker.chunk([section])
        assert chunks, "No semantic chunks were generated"

        for chunk in chunks:
            print(chunk.model_dump_json(indent=2))

        assert chunks[0].chunk_id == "chunk_0001"
        assert all(chunk.document_name == section.document_name for chunk in chunks)
        print(f"SUCCESS: Generated {len(chunks)} semantic chunk(s).")
    finally:
        await embedding_provider.close()


if __name__ == "__main__":
    asyncio.run(main())
