import json
import re
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import AssetModel, ChunkModel, get_db_session


export_router = APIRouter(
    prefix="/api/v1/ingestion",
    tags=["Ingestion"],
)


def _safe_export_name(document_name: str) -> str:
    normalized = re.sub(
        r"[^A-Za-z0-9._ -]+",
        "_",
        document_name,
    ).strip()

    normalized = normalized or "document"

    return f"{normalized}_chunks.json"


@export_router.get(
    "/assets/{asset_id}/export",
    response_class=Response,
    status_code=status.HTTP_200_OK,
    summary="Download indexed chunks as a JSON file",
)
async def export_asset_chunks(
    asset_id: int,
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    asset = await AssetModel.get_by_id(
        session=session,
        asset_id=asset_id,
    )

    if asset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Asset {asset_id} was not found.",
        )

    chunk_records = await ChunkModel.get_by_asset(
        session=session,
        asset_id=asset_id,
    )

    if not chunk_records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Asset {asset_id} has no indexed chunks.",
        )

    chunks = [
        record.to_output_dict()
        for record in chunk_records
    ]

    json_bytes = json.dumps(
        chunks,
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")

    file_name = _safe_export_name(
        asset.document_name,
    )

    encoded_file_name = quote(file_name)

    return Response(
        content=json_bytes,
        media_type="application/json; charset=utf-8",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{file_name}"; '
                f"filename*=UTF-8''{encoded_file_name}"
            ),
            "Content-Length": str(len(json_bytes)),
            "X-Asset-ID": str(asset_id),
            "X-Total-Chunks": str(len(chunks)),
        },
    )