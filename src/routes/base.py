from fastapi import APIRouter, status

from src.helpers.config import settings


base_router = APIRouter(
    prefix="/api/v1",
    tags=["Base"],
)


@base_router.get(
    "/",
    status_code=status.HTTP_200_OK,
    summary="Get application information",
)
async def get_app_info() -> dict[str, str]:
    return {
        "app_name": settings.APP_NAME,
        "app_version": settings.APP_VERSION,
    }
