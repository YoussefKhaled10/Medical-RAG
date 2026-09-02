import base64
import hashlib
import hmac
import json
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from src.helpers.config import settings


def hash_password(password: str) -> str:
    """Hash password using PBKDF2-HMAC-SHA256 with a unique salt."""
    salt = secrets.token_bytes(16)
    key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        100_000,
    )
    return f"pbkdf2_sha256$100000${base64.b64encode(salt).decode('ascii')}${base64.b64encode(key).decode('ascii')}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against PBKDF2 hash."""
    try:
        parts = hashed_password.split("$")
        if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
            return False
        iterations = int(parts[1])
        salt = base64.b64decode(parts[2].encode("ascii"))
        expected_key = base64.b64decode(parts[3].encode("ascii"))
        actual_key = hashlib.pbkdf2_hmac(
            "sha256",
            plain_password.encode("utf-8"),
            salt,
            iterations,
        )
        return hmac.compare_digest(expected_key, actual_key)
    except Exception:
        return False


def _base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _base64url_decode(data: str) -> bytes:
    padding = 4 - (len(data) % 4)
    if padding != 4:
        data += "=" * padding
    return base64.urlsafe_b64decode(data.encode("ascii"))


def create_access_token(
    data: dict[str, Any], expires_delta: timedelta | None = None
) -> str:
    """Create a signed JWT access token using HMAC-SHA256."""
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": int(expire.timestamp()), "iat": int(now.timestamp())})

    header = {"alg": "HS256", "typ": "JWT"}
    header_b64 = _base64url_encode(
        json.dumps(header, separators=(",", ":")).encode("utf-8")
    )
    payload_b64 = _base64url_encode(
        json.dumps(to_encode, separators=(",", ":")).encode("utf-8")
    )

    message = f"{header_b64}.{payload_b64}"
    signature = hmac.new(
        settings.JWT_SECRET_KEY.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    sig_b64 = _base64url_encode(signature)

    return f"{message}.{sig_b64}"


def decode_access_token(token: str) -> dict[str, Any] | None:
    """Decode, verify signature and check expiration of a JWT access token."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b64, payload_b64, sig_b64 = parts
        message = f"{header_b64}.{payload_b64}"
        expected_sig = hmac.new(
            settings.JWT_SECRET_KEY.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        actual_sig = _base64url_decode(sig_b64)

        if not hmac.compare_digest(expected_sig, actual_sig):
            return None

        payload = json.loads(_base64url_decode(payload_b64).decode("utf-8"))
        exp = payload.get("exp")
        if exp is not None:
            now = datetime.now(timezone.utc).timestamp()
            if now > exp:
                return None  # Token expired

        return payload
    except Exception:
        return None


def create_refresh_token(user_id: int) -> tuple[str, datetime]:
    """Generate a random cryptographic refresh token and calculate its expiration date."""
    token = secrets.token_urlsafe(64)
    expires_at = datetime.now(timezone.utc) + timedelta(
        days=settings.REFRESH_TOKEN_EXPIRE_DAYS
    )
    return token, expires_at
