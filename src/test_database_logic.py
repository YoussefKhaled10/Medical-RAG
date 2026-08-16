import asyncio
from uuid import uuid4

from sqlalchemy import select

from src.models import (
    AssetModel,
    AsyncSessionLocal,
    ChunkModel,
    ProjectModel,
    close_database,
)
from src.models.db_schemes.medical_rag import Asset, Chunk, Project
from src.schemas.ingestion import SemanticChunk


async def run_test() -> None:
    test_suffix = uuid4().hex[:8]
    project_name = f"db-logic-test-{test_suffix}"
    project_id: int | None = None

    try:
        async with AsyncSessionLocal() as session:
            print("[1/8] Creating test project...")
            project = await ProjectModel.create(
                session=session,
                name=project_name,
                description="Temporary project for database logic testing",
                source_name="NICE",
                source_url="https://www.nice.org.uk/",
                source_version="2025",
                license_name="NICE notice of rights",
            )
            project_id = project.id

            assert project.id is not None
            assert project.name == project_name

            print("[2/8] Reading project from PostgreSQL...")
            stored_project = await ProjectModel.get_by_id(
                session=session,
                project_id=project.id,
            )
            assert stored_project is not None
            assert stored_project.name == project_name

            print("[3/8] Creating test asset...")
            asset = await AssetModel.create(
                session=session,
                project_id=project.id,
                document_name="Alcohol-use disorders",
                file_name="Alcohol-use disorders.pdf",
                file_path=f"/tmp/{test_suffix}/Alcohol-use disorders.pdf",
                file_type="application/pdf",
                file_size=1024,
                file_checksum=f"{test_suffix:0<64}"[:64],
            )

            assert asset.id is not None
            assert asset.project_id == project.id
            assert asset.processing_status == "PENDING"

            print("[4/8] Updating asset status to PROCESSING...")
            processing_asset = await AssetModel.update_status(
                session=session,
                asset_id=asset.id,
                status="PROCESSING",
            )
            assert processing_asset is not None
            assert processing_asset.processing_status == "PROCESSING"

            print("[5/8] Replacing asset chunks...")
            semantic_chunks = [
                SemanticChunk(
                    chunk_id="chunk_0001",
                    document_name="Alcohol-use disorders",
                    section_title="Contents",
                    page_number=2,
                    text=(
                        "Quality statement 1: Use of validated alcohol "
                        "questionnaires."
                    ),
                ),
                SemanticChunk(
                    chunk_id="chunk_0002",
                    document_name="Alcohol-use disorders",
                    section_title=(
                        "Quality statement 2: Community support networks "
                        "and self-help groups"
                    ),
                    page_number=10,
                    text=(
                        "People who misuse alcohol are given information "
                        "about community support networks and self-help groups."
                    ),
                ),
            ]

            chunk_records = await ChunkModel.replace_asset_chunks(
                session=session,
                asset_id=asset.id,
                chunks=semantic_chunks,
                metadata={
                    "test_run": test_suffix,
                    "parser": "UnstructuredPDFParser",
                    "chunker": "SemanticChunker",
                },
            )

            assert len(chunk_records) == 2
            assert chunk_records[0].chunk_id == "chunk_0001"
            assert chunk_records[1].chunk_id == "chunk_0002"

            print("[6/8] Reading and validating stored chunks...")
            stored_chunks = await ChunkModel.get_by_asset(
                session=session,
                asset_id=asset.id,
            )
            assert len(stored_chunks) == 2

            public_output = [chunk.to_output_dict() for chunk in stored_chunks]
            assert public_output == [
                chunk.model_dump() for chunk in semantic_chunks
            ]

            count = await ChunkModel.count_by_asset(
                session=session,
                asset_id=asset.id,
            )
            assert count == 2

            one_chunk = await ChunkModel.get_by_chunk_id(
                session=session,
                asset_id=asset.id,
                chunk_id="chunk_0001",
            )
            assert one_chunk is not None
            assert one_chunk.section_title == "Contents"
            assert one_chunk.chunk_metadata["test_run"] == test_suffix

            print("[7/8] Marking asset as COMPLETED...")
            completed_asset = await AssetModel.mark_completed(
                session=session,
                asset_id=asset.id,
                total_pages=12,
                total_chunks=count,
            )
            assert completed_asset is not None
            assert completed_asset.processing_status == "COMPLETED"
            assert completed_asset.total_pages == 12
            assert completed_asset.total_chunks == 2
            assert completed_asset.parsed_at is not None

            print("[8/8] Public chunk output:")
            for item in public_output:
                print(item)

            print("SUCCESS: Database logic integration test passed.")

    finally:
        if project_id is not None:
            async with AsyncSessionLocal() as cleanup_session:
                print("Cleaning test project, asset and chunks...")
                await ProjectModel.delete(
                    session=cleanup_session,
                    project_id=project_id,
                )

                project_exists = await cleanup_session.scalar(
                    select(Project.id).where(Project.id == project_id)
                )
                orphan_asset = await cleanup_session.scalar(
                    select(Asset.id).where(Asset.project_id == project_id)
                )
                orphan_chunk = await cleanup_session.scalar(
                    select(Chunk.id)
                    .join(Asset, Chunk.asset_id == Asset.id)
                    .where(Asset.project_id == project_id)
                )

                assert project_exists is None
                assert orphan_asset is None
                assert orphan_chunk is None

        await close_database()


if __name__ == "__main__":
    asyncio.run(run_test())
