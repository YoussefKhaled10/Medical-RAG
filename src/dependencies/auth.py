from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from src.helpers.security import decode_access_token
from src.models import UserModel, get_db_session
from src.models.db_schemes.medical_rag import User


http_bearer = HTTPBearer(auto_error=False)


def _unauthorized(detail: str) -> HTTPException:
    """Build a consistent HTTP 401 response."""

    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _extract_access_token_payload(
    credentials: HTTPAuthorizationCredentials | None,
) -> dict:
    """Validate the Bearer credentials and decode an access token."""

    if credentials is None:
        raise _unauthorized("Missing authentication token.")

    if credentials.scheme.lower() != "bearer":
        raise _unauthorized("Invalid authentication scheme.")

    token = credentials.credentials.strip()

    if not token:
        raise _unauthorized("Missing authentication token.")

    payload = decode_access_token(token)

    if not payload:
        raise _unauthorized("Token is invalid or expired.")

    if "sub" not in payload:
        raise _unauthorized("Token payload is missing the user identifier.")

    token_type = str(payload.get("type") or "").strip().lower()

    if token_type != "access":
        raise _unauthorized("An access token is required.")

    return payload


async def _get_user_from_payload(
    payload: dict,
    session: AsyncSession,
) -> User:
    """Load and validate the user referenced by the token payload."""

    try:
        user_id = int(payload["sub"])
    except (TypeError, ValueError):
        raise _unauthorized("Malformed token identifier.")

    user = await UserModel.get_by_id(
        session,
        user_id,
    )

    if user is None:
        raise _unauthorized("User no longer exists.")

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated.",
        )

    return user


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(http_bearer),
    session: AsyncSession = Depends(get_db_session),
) -> User:
    """
    Require a valid access token and return the authenticated user.

    Used by protected routes such as:
    - /auth/me
    - private uploads
    - system uploads
    - private conversations
    """

    payload = _extract_access_token_payload(credentials)

    return await _get_user_from_payload(
        payload,
        session,
    )


async def get_optional_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(http_bearer),
    session: AsyncSession = Depends(get_db_session),
) -> User | None:
    """
    Support routes that allow both guests and authenticated users.

    Behavior:
    - No Authorization header: return None and continue as Guest.
    - Valid access token: return the authenticated user.
    - Invalid, expired, or wrong token type: return HTTP 401.

    A malformed token must never silently downgrade to Guest access.
    """

    if credentials is None:
        return None

    payload = _extract_access_token_payload(credentials)

    return await _get_user_from_payload(
        payload,
        session,
    )


async def get_current_admin(
    current_user: User = Depends(get_current_user),
) -> User:
    """Require an active authenticated administrator."""

    if not bool(getattr(current_user, "is_admin", False)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator privileges are required.",
        )

    return current_user