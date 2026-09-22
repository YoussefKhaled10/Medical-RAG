import hashlib
from pathlib import Path
from typing import Any
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
from src.helpers.config import settings
from src.services.ingestion_factory import create_ingestion_service


UPLOAD_ROOT = Path(__file__).resolve().parents[2] / "uploads"
MAX_FILE_SIZE = 50 * 1024 * 1024
MAX_PDF_SIZE = MAX_FILE_SIZE


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


def _safe_file_name(original_name: str | None) -> str:
    if not original_name:
        return "document.pdf"

    file_name = Path(original_name).name
    suffix = Path(file_name).suffix.lower()
    if suffix not in {".pdf", ".txt"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF (.pdf) and Text (.txt) files are supported.",
        )
    return file_name


# Alias for backward compatibility
_safe_pdf_name = _safe_file_name


async def _save_file(
    upload_file: UploadFile,
    project_id: int,
) -> tuple[Path, int, str, str]:
    file_name = _safe_file_name(upload_file.filename)
    suffix = Path(file_name).suffix.lower()
    upload_directory = UPLOAD_ROOT / str(project_id)
    upload_directory.mkdir(parents=True, exist_ok=True)

    stored_path = upload_directory / f"{uuid4().hex}_{file_name}"
    file_size = 0
    digest = hashlib.sha256()

    try:
        with stored_path.open("wb") as destination:
            while chunk := await upload_file.read(1024 * 1024):
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="File exceeds the 50 MB upload limit.",
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
            detail="The uploaded file is empty.",
        )

    file_type = "application/pdf" if suffix == ".pdf" else "text/plain"
    if suffix == ".pdf":
        with stored_path.open("rb") as file:
            if file.read(5) != b"%PDF-":
                stored_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="The uploaded file is not a valid PDF.",
                )
    elif suffix == ".txt":
        try:
            with stored_path.open("rb") as file:
                sample = file.read(4096)
                if b"\x00" in sample:
                    stored_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="The uploaded file appears to be a binary file, not valid text.",
                    )
        except HTTPException:
            raise
        except Exception as exc:
            stored_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to validate text file: {exc}",
            )

    return stored_path, file_size, digest.hexdigest(), file_type


_save_pdf = _save_file


from src.dependencies.auth import get_current_admin, get_current_user, get_optional_current_user
from src.models.db_schemes.medical_rag import User


@ingestion_router.post(
    "/upload-index",
    response_model=IngestionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a document (PDF or TXT) into user's private vault",
)
async def upload_and_index_document(
    project_id: int | None = Form(default=None),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> IngestionResponse:
    # Uploads always target the authenticated user's private vault.
    if current_user is not None:
        target_project_id = current_user.private_project_id
        if project_id is not None and project_id != current_user.private_project_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You can only upload files into your own Private Vault.",
            )
        if not target_project_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User does not have an active Private Vault assigned.",
            )
    else:
        if project_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="project_id is required for unauthenticated uploads.",
            )
        target_project_id = project_id

    project = await ProjectModel.get_by_id(
        session=session,
        project_id=target_project_id,
    )
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {target_project_id} was not found.",
        )


    saved_path: Path | None = None
    asset_id: int | None = None
    service = None

    try:
        saved_path, file_size, checksum, file_type = await _save_file(
            upload_file=file,
            project_id=target_project_id,
        )

        existing_asset = await AssetModel.get_by_checksum(
            session=session,
            project_id=target_project_id,
            file_checksum=checksum,
        )
        if existing_asset is not None:
            if existing_asset.processing_status == "COMPLETED":
                saved_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail={
                        "message": "This document already exists in the project.",
                        "asset_id": existing_asset.id,
                        "processing_status": existing_asset.processing_status,
                    },
                )
            else:
                # If previously failed or incomplete, clean up old index/records and allow re-upload
                cleanup_service = create_ingestion_service()
                try:
                    await cleanup_service.remove_asset_index(
                        session=session,
                        asset_id=existing_asset.id,
                    )
                except Exception:
                    pass
                finally:
                    await cleanup_service.close()

                if existing_asset.file_path and existing_asset.file_path != str(saved_path):
                    Path(existing_asset.file_path).unlink(missing_ok=True)
                
                await AssetModel.delete(
                    session=session,
                    asset_id=existing_asset.id,
                    commit=True,
                )

        original_name = _safe_file_name(file.filename)
        asset = await AssetModel.create(
            session=session,
            project_id=target_project_id,
            document_name=Path(original_name).stem,
            file_name=original_name,
            file_path=str(saved_path),
            file_type=file_type,
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
            project_id=target_project_id,
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
                "message": "Document ingestion failed.",
                "asset_id": asset_id,
                "error": str(exc),
            },
        ) from exc
    finally:
        if service is not None:
            await service.close()




@ingestion_router.post(
    "/system-upload",
    response_model=IngestionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a document into the system knowledge base (admin only)",
)
async def upload_and_index_system_document(
    file: UploadFile = File(...),
    current_admin: User = Depends(get_current_admin),
    session: AsyncSession = Depends(get_db_session),
) -> IngestionResponse:
    # Index a document in the server-configured global project.
    global_project_id = int(settings.GLOBAL_PROJECT_ID)
    if global_project_id <= 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GLOBAL_PROJECT_ID is not configured correctly.",
        )

    # Reuse the existing validated ingestion path without duplicating logic.
    original_private_project_id = current_admin.private_project_id
    try:
        current_admin.private_project_id = global_project_id
        return await upload_and_index_document(
            project_id=global_project_id,
            file=file,
            current_user=current_admin,
            session=session,
        )
    finally:
        current_admin.private_project_id = original_private_project_id


upload_and_index_pdf = upload_and_index_document


@ingestion_router.get("/my-assets", summary="List the authenticated user's private files")
async def list_my_assets(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> list[dict[str, Any]]:
    if not current_user.private_project_id:
        return []
    assets = await AssetModel.list_by_project(session, current_user.private_project_id)
    return [
        {
            "id": item.id,
            "file_name": item.file_name,
            "document_name": item.document_name,
            "status": item.processing_status,
            "created_at": item.created_at.isoformat() if item.created_at else None,
        }
        for item in assets
    ]


@ingestion_router.get(
    "/assets/{asset_id}/chunks",
    response_model=AssetChunksResponse,
    status_code=status.HTTP_200_OK,
    summary="Get indexed chunks for an asset",
)
async def get_asset_chunks(
    asset_id: int,
    current_user: User = Depends(get_current_user),
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

    # Users may view only the configured system KB or their own Private Vault.
    global_project_id = int(settings.GLOBAL_PROJECT_ID)
    if asset.project_id not in {global_project_id, current_user.private_project_id}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You do not have permission to view this document.",
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


@ingestion_router.delete(
    "/assets/{asset_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete an asset and its vector documents",
)
async def delete_asset(
    asset_id: int,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, Any]:
    asset = await AssetModel.get_by_id(
        session=session,
        asset_id=asset_id,
    )
    if asset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Asset {asset_id} was not found.",
        )

    if asset.project_id != current_user.private_project_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You can only delete documents from your own Private Vault.",
        )

    service = create_ingestion_service()
    try:
        await service.remove_asset_index(session=session, asset_id=asset_id)
        if asset.file_path:
            Path(asset.file_path).unlink(missing_ok=True)
        await AssetModel.delete(session=session, asset_id=asset_id, commit=True)
    finally:
        await service.close()

    return {"success": True, "message": f"Asset {asset_id} deleted successfully."}

