import asyncio
from pathlib import Path
from sqlalchemy.ext.asyncio import AsyncSession
from src.chunkers import ChunkerInterface
from src.models import AssetModel, ChunkModel
from src.parsers import PDFParserInterface, SectionBuilder, TextParser
from src.schemas.ingestion import SemanticChunk
from src.stores.llm.LLMInterface import LLMInterface
from src.stores.vectordb.VectorDBInterface import (
    VectorDBInterface,
    VectorDocument,
)


class IngestionService:
    def __init__(
        self,
        parser: PDFParserInterface,
        section_builder: SectionBuilder,
        semantic_chunker: ChunkerInterface,
        embedding_provider: LLMInterface,
        vector_db: VectorDBInterface,
        text_parser: TextParser | None = None,
    ) -> None:
        self._parser = parser
        self._text_parser = text_parser or TextParser()
        self._section_builder = section_builder
        self._semantic_chunker = semantic_chunker
        self._embedding_provider = embedding_provider
        self._vector_db = vector_db

    async def ingest_asset(self, session: AsyncSession, asset_id: int) -> list[SemanticChunk]:
        if asset_id <= 0:
            raise ValueError("asset_id must be greater than zero")
        asset = await AssetModel.get_by_id(session=session, asset_id=asset_id)
        if asset is None:
            raise ValueError(f"Asset was not found: {asset_id}")
        file_path = Path(asset.file_path).expanduser().resolve()
        if not file_path.is_file():
            raise FileNotFoundError(f"Asset file was not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        if suffix not in {".pdf", ".txt"}:
            raise ValueError(f"Asset file must be a PDF or TXT, got: {suffix}")
        
        active_parser = self._text_parser if suffix == ".txt" else self._parser
        await AssetModel.update_status(session=session, asset_id=asset_id, status="PROCESSING")
        vector_ids = []
        try:
            elements = await asyncio.to_thread(active_parser.parse, file_path)
            sections = self._section_builder.build(document_name=asset.document_name, elements=elements)
            if not sections:
                raise ValueError("No document sections were generated")
            chunks = await self._semantic_chunker.chunk(sections)
            if not chunks:
                raise ValueError("No semantic chunks were generated")
            await ChunkModel.replace_asset_chunks(
                session=session,
                asset_id=asset_id,
                chunks=chunks,
                metadata={
                    "parser": active_parser.__class__.__name__,
                    "chunker": self._semantic_chunker.__class__.__name__,
                },
                commit=False,
            )
            embeddings = await self._embedding_provider.embed_documents([chunk.text for chunk in chunks])
            if len(embeddings) != len(chunks):
                raise RuntimeError("Embedding count does not match chunk count")
            documents = []
            for chunk, embedding in zip(chunks, embeddings):
                vector_id = f"asset_{asset_id}_{chunk.chunk_id}"
                vector_ids.append(vector_id)
                documents.append(
                    VectorDocument(
                        id=vector_id,
                        text=chunk.text,
                        embedding=embedding,
                        metadata={
                            "asset_id": asset_id,
                            "project_id": asset.project_id,
                            "chunk_id": chunk.chunk_id,
                            "document_name": chunk.document_name,
                            "section_title": chunk.section_title,
                            "page_number": chunk.page_number,
                        },
                    )
                )
            await self._vector_db.initialize()
            await self._vector_db.delete_by_asset_id(asset_id)
            await self._vector_db.upsert(documents)
            total_pages = max((e.page_number for e in elements), default=1)
            await AssetModel.mark_completed(
                session=session,
                asset_id=asset_id,
                total_pages=total_pages,
                total_chunks=len(chunks),
                commit=False,
            )
            await session.commit()
            return chunks
        except Exception as exc:
            await session.rollback()
            try:
                if vector_ids:
                    await self._vector_db.delete_by_ids(vector_ids)
            except Exception:
                pass
            await AssetModel.mark_failed(session=session, asset_id=asset_id, error_message=str(exc))
            raise

    async def get_asset_output(self, session: AsyncSession, asset_id: int) -> list[dict[str, str | int]]:
        records = await ChunkModel.get_by_asset(session=session, asset_id=asset_id)
        return [record.to_output_dict() for record in records]

    async def remove_asset_index(self, session: AsyncSession, asset_id: int) -> int:
        await self._vector_db.initialize()
        deleted = await self._vector_db.delete_by_asset_id(asset_id)
        await ChunkModel.delete_by_asset(session=session, asset_id=asset_id)
        return deleted

    async def close(self) -> None:
        await self._embedding_provider.close()
        await self._vector_db.close()
