import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from src.models import (
    AssetModel,
    AsyncSessionLocal,
    ChunkModel,
    ProjectModel,
    close_database,
)
from src.services.ingestion_factory import create_ingestion_service


def calculate_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the complete PDF ingestion pipeline integration test."
    )
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument(
        "--output",
        default="ipv_ingestion_output.json",
        help="Path for the public chunk JSON output",
    )
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Keep the temporary project, asset, chunks, and vectors",
    )
    return parser


async def run_test(
    pdf_path: Path,
    output_path: Path,
    keep_data: bool,
) -> None:
    suffix = uuid4().hex[:8]
    project_id: int | None = None
    asset_id: int | None = None
    service = create_ingestion_service()

    try:
        print("[1/9] Validating source PDF...")
        assert pdf_path.is_file(), f"PDF was not found: {pdf_path}"
        assert pdf_path.suffix.lower() == ".pdf", "Input must be a PDF"
        print(f"  path: {pdf_path}")
        print(f"  size: {pdf_path.stat().st_size:,} bytes")

        async with AsyncSessionLocal() as session:
            print("[2/9] Creating temporary project...")
            project = await ProjectModel.create(
                session=session,
                name=f"IPV ingestion test {suffix}",
                description="Temporary full-pipeline integration test",
                source_name="USPSTF / JAMA",
                source_version="2025",
                commit=True,
            )
            project_id = project.id

            print("[3/9] Creating temporary asset...")
            asset = await AssetModel.create(
                session=session,
                project_id=project.id,
                document_name=pdf_path.stem,
                file_name=pdf_path.name,
                file_path=str(pdf_path),
                file_type="application/pdf",
                file_size=pdf_path.stat().st_size,
                file_checksum=calculate_sha256(pdf_path),
                commit=True,
            )
            asset_id = asset.id
            print(f"  project_id: {project_id}")
            print(f"  asset_id: {asset_id}")

            print("[4/9] Running complete ingestion pipeline...")
            chunks = await service.ingest_asset(
                session=session,
                asset_id=asset.id,
            )
            assert chunks, "Ingestion returned no semantic chunks"

            print("[5/9] Validating public output contract...")
            required_keys = {
                "chunk_id",
                "document_name",
                "section_title",
                "page_number",
                "text",
            }
            public_output = [chunk.model_dump() for chunk in chunks]

            for index, item in enumerate(public_output, start=1):
                assert set(item) == required_keys, (
                    f"Chunk {index} has unexpected fields: {set(item)}"
                )
                assert item["chunk_id"] == f"chunk_{index:04d}", (
                    f"Unexpected chunk ID at position {index}"
                )
                assert item["document_name"] == pdf_path.stem
                assert item["page_number"] >= 1
                assert item["section_title"].strip()
                assert item["text"].strip()

            print("[6/9] Validating PostgreSQL records...")
            stored_chunks = await ChunkModel.get_by_asset(
                session=session,
                asset_id=asset.id,
            )
            assert len(stored_chunks) == len(chunks)
            assert [item.to_output_dict() for item in stored_chunks] == public_output

            stored_asset = await AssetModel.get_by_id(
                session=session,
                asset_id=asset.id,
            )
            assert stored_asset is not None
            assert stored_asset.processing_status == "COMPLETED"
            assert stored_asset.total_pages is not None and stored_asset.total_pages > 0, (
                f"Expected positive total_pages, got {stored_asset.total_pages}"
            )
            assert stored_asset.total_chunks == len(chunks)

            print("[7/9] Validating PGVector retrieval...")
            query_embedding = await service._embedding_provider.embed_query(
                "Which patients should clinicians screen for intimate partner violence?"
            )
            results = await service._vector_db.similarity_search(
                query_embedding=query_embedding,
                limit=min(5, len(chunks)),
                filters={"asset_id": asset.id},
            )
            assert results, "PGVector returned no search results"
            assert all(result.metadata["asset_id"] == asset.id for result in results)

            print("[8/9] Saving public JSON output...")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as file:
                json.dump(public_output, file, ensure_ascii=False, indent=2)

            with output_path.open("r", encoding="utf-8") as file:
                saved_output = json.load(file)
            assert saved_output == public_output

            print("[9/9] Results summary")
            print(f"  pages: {stored_asset.total_pages}")
            print(f"  sections represented: {len({c.section_title for c in chunks})}")
            print(f"  chunks: {len(chunks)}")
            print(f"  first chunk: {chunks[0].chunk_id} / page {chunks[0].page_number}")
            print(f"  last chunk: {chunks[-1].chunk_id} / page {chunks[-1].page_number}")
            print(f"  output: {output_path}")
            print("  top vector results:")
            for position, result in enumerate(results, start=1):
                print(
                    f"    {position}. score={result.score:.4f} "
                    f"page={result.metadata.get('page_number')} "
                    f"section={result.metadata.get('section_title')}"
                )

            print("SUCCESS: Full ingestion pipeline test passed.")

    finally:
        if not keep_data and project_id is not None:
            print("Cleaning temporary database and vector data...")
            if asset_id is not None:
                try:
                    await service._vector_db.initialize()
                    await service._vector_db.delete_by_asset_id(asset_id)
                except Exception as cleanup_error:
                    print(f"  vector cleanup warning: {cleanup_error}")

            async with AsyncSessionLocal() as cleanup_session:
                await ProjectModel.delete(
                    session=cleanup_session,
                    project_id=project_id,
                )
        elif keep_data:
            print("Temporary data was kept because --keep-data was supplied.")
            print(f"  project_id: {project_id}")
            print(f"  asset_id: {asset_id}")

        await service.close()
        await close_database()


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(
        run_test(
            pdf_path=Path(args.pdf_path).expanduser().resolve(),
            output_path=Path(args.output).expanduser().resolve(),
            keep_data=args.keep_data,
        )
    )


if __name__ == "__main__":
    main()
