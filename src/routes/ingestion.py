import hashlib
from pathlib import Path
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import AssetModel, ProjectModel, get_db_session
from src.services.ingestion_factory import create_ingestion_service


UPLOAD_ROOT = Path(__file__).resolve().parents[2] / "uploads"
MAX_PDF_SIZE = 50 * 1024 * 1024


ingestion_router = APIRouter(
    prefix="/api/v1/ingestion",
    tags=["Ingestion"],
)


class ChunkResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_name: str
    section_title: str
    page_number: int
    text: str


class IngestionResponse(BaseModel):
    project_id: int
    asset_id: int
    document_name: str
    processing_status: str
    total_pages: int
    total_chunks: int
    chunks: list[ChunkResponse]


class AssetChunksResponse(BaseModel):
    asset_id: int
    total_chunks: int
    chunks: list[ChunkResponse]


def _safe_pdf_name(original_name: str | None) -> str:
    if not original_name:
        return "document.pdf"

    file_name = Path(original_name).name
    if Path(file_name).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )
    return file_name


async def _save_pdf(
    upload_file: UploadFile,
    project_id: int,
) -> tuple[Path, int, str]:
    file_name = _safe_pdf_name(upload_file.filename)
    upload_directory = UPLOAD_ROOT / str(project_id)
    upload_directory.mkdir(parents=True, exist_ok=True)

    stored_path = upload_directory / f"{uuid4().hex}_{file_name}"
    file_size = 0
    digest = hashlib.sha256()

    try:
        with stored_path.open("wb") as destination:
            while chunk := await upload_file.read(1024 * 1024):
                file_size += len(chunk)
                if file_size > MAX_PDF_SIZE:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="PDF exceeds the 50 MB upload limit.",
                    )
                digest.update(chunk)
                destination.write(chunk)
    except Exception:
        stored_path.unlink(missing_ok=True)
        raise
    finally:
        await upload_file.close()

    if file_size == 0:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded PDF is empty.",
        )

    with stored_path.open("rb") as file:
        if file.read(5) != b"%PDF-":
            stored_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded file is not a valid PDF.",
            )

    return stored_path, file_size, digest.hexdigest()


@ingestion_router.post(
    "/upload-index",
    response_model=IngestionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a PDF",
)
async def upload_and_index_pdf(
    project_id: int = Form(..., gt=0),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_db_session),
) -> IngestionResponse:
    project = await ProjectModel.get_by_id(
        session=session,
        project_id=project_id,
    )
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} was not found.",
        )

    saved_path: Path | None = None
    asset_id: int | None = None
    service = None

    try:
        saved_path, file_size, checksum = await _save_pdf(
            upload_file=file,
            project_id=project_id,
        )

        existing_asset = await AssetModel.get_by_checksum(
            session=session,
            project_id=project_id,
            file_checksum=checksum,
        )
        if existing_asset is not None:
            saved_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "message": "This PDF already exists in the project.",
                    "asset_id": existing_asset.id,
                    "processing_status": existing_asset.processing_status,
                },
            )

        original_name = _safe_pdf_name(file.filename)
        asset = await AssetModel.create(
            session=session,
            project_id=project_id,
            document_name=Path(original_name).stem,
            file_name=original_name,
            file_path=str(saved_path),
            file_type="application/pdf",
            file_size=file_size,
            file_checksum=checksum,
        )
        asset_id = asset.id

        service = create_ingestion_service()
        chunks = await service.ingest_asset(
            session=session,
            asset_id=asset.id,
        )

        completed_asset = await AssetModel.get_by_id(
            session=session,
            asset_id=asset.id,
        )
        if completed_asset is None:
            raise RuntimeError("Asset disappeared after ingestion.")

        return IngestionResponse(
            project_id=project_id,
            asset_id=asset.id,
            document_name=asset.document_name,
            processing_status=completed_asset.processing_status,
            total_pages=completed_asset.total_pages or 0,
            total_chunks=completed_asset.total_chunks,
            chunks=[ChunkResponse(**chunk.model_dump()) for chunk in chunks],
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "PDF ingestion failed.",
                "asset_id": asset_id,
                "error": str(exc),
            },
        ) from exc
    finally:
        if service is not None:
            await service.close()


@ingestion_router.get(
    "/assets/{asset_id}/chunks",
    response_model=AssetChunksResponse,
    status_code=status.HTTP_200_OK,
    summary="Get indexed chunks for an asset",
)
async def get_asset_chunks(
    asset_id: int,
    session: AsyncSession = Depends(get_db_session),
) -> AssetChunksResponse:
    asset = await AssetModel.get_by_id(
        session=session,
        asset_id=asset_id,
    )
    if asset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Asset {asset_id} was not found.",
        )

    service = create_ingestion_service()
    try:
        output = await service.get_asset_output(
            session=session,
            asset_id=asset_id,
        )
    finally:
        await service.close()

    return AssetChunksResponse(
        asset_id=asset_id,
        total_chunks=len(output),
        chunks=[ChunkResponse(**item) for item in output],
    )
