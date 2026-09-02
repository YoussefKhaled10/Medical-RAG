from collections.abc import Sequence
from datetime import datetime, timezone, timedelta
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.db_schemes.medical_rag import User, EmailVerification, RefreshToken, Project


class UserModel:
    @staticmethod
    async def create_user(
        session: AsyncSession,
        email: str,
        username: str,
        hashed_password: str,
        full_name: str | None = None,
        private_project_id: int | None = None,
        is_verified: bool = False,
        commit: bool = True,
    ) -> User:
        user = User(
            email=email.strip().lower(),
            username=username.strip(),
            hashed_password=hashed_password,
            full_name=full_name.strip() if full_name else None,
            private_project_id=private_project_id,
            is_verified=is_verified,
            is_active=True,
        )
        session.add(user)
        if commit:
            await session.commit()
            await session.refresh(user)
        else:
            await session.flush()
        return user

    @staticmethod
    async def get_by_id(session: AsyncSession, user_id: int) -> User | None:
        return await session.get(User, user_id)

    @staticmethod
    async def get_by_email(session: AsyncSession, email: str) -> User | None:
        result = await session.execute(
            select(User).where(User.email == email.strip().lower())
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_username(session: AsyncSession, username: str) -> User | None:
        result = await session.execute(
            select(User).where(User.username == username.strip())
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_email_or_username(
        session: AsyncSession, identifier: str
    ) -> User | None:
        clean = identifier.strip()
        result = await session.execute(
            select(User).where(
                (User.email == clean.lower()) | (User.username == clean)
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def assign_private_project(
        session: AsyncSession,
        user_id: int,
        project_id: int,
        commit: bool = True,
    ) -> User | None:
        user = await session.get(User, user_id)
        if user:
            user.private_project_id = project_id
            if commit:
                await session.commit()
                await session.refresh(user)
            else:
                await session.flush()
        return user

    @staticmethod
    async def set_verified(
        session: AsyncSession, user_id: int, commit: bool = True
    ) -> User | None:
        user = await session.get(User, user_id)
        if user:
            user.is_verified = True
            if commit:
                await session.commit()
                await session.refresh(user)
            else:
                await session.flush()
        return user

    # ================= Email Verification OTPs =================

    @staticmethod
    async def create_verification_code(
        session: AsyncSession,
        email: str,
        code: str,
        validity_minutes: int = 15,
        commit: bool = True,
    ) -> EmailVerification:
        # Invalidate previous unused codes for this email
        await session.execute(
            update(EmailVerification)
            .where(
                EmailVerification.email == email.strip().lower(),
                EmailVerification.is_used == False,
            )
            .values(is_used=True)
        )

        expires_at = datetime.now(timezone.utc) + timedelta(minutes=validity_minutes)
        verification = EmailVerification(
            email=email.strip().lower(),
            code=code.strip(),
            expires_at=expires_at,
            is_used=False,
        )
        session.add(verification)
        if commit:
            await session.commit()
            await session.refresh(verification)
        else:
            await session.flush()
        return verification

    @staticmethod
    async def verify_code(
        session: AsyncSession,
        email: str,
        code: str,
        consume: bool = True,
        commit: bool = True,
    ) -> bool:
        clean_email = email.strip().lower()
        clean_code = code.strip()
        now = datetime.now(timezone.utc)

        result = await session.execute(
            select(EmailVerification).where(
                EmailVerification.email == clean_email,
                EmailVerification.code == clean_code,
                EmailVerification.is_used == False,
                EmailVerification.expires_at >= now,
            ).order_by(EmailVerification.id.desc())
        )
        record = result.scalar_one_or_none()
        if not record:
            return False

        if consume:
            record.is_used = True
            if commit:
                await session.commit()
            else:
                await session.flush()
        return True

    # ================= Refresh Tokens =================

    @staticmethod
    async def store_refresh_token(
        session: AsyncSession,
        user_id: int,
        token: str,
        expires_at: datetime,
        commit: bool = True,
    ) -> RefreshToken:
        rt = RefreshToken(
            user_id=user_id,
            token=token,
            expires_at=expires_at,
            revoked=False,
        )
        session.add(rt)
        if commit:
            await session.commit()
            await session.refresh(rt)
        else:
            await session.flush()
        return rt

    @staticmethod
    async def get_valid_refresh_token(
        session: AsyncSession, token: str
    ) -> RefreshToken | None:
        now = datetime.now(timezone.utc)
        result = await session.execute(
            select(RefreshToken).where(
                RefreshToken.token == token,
                RefreshToken.revoked == False,
                RefreshToken.expires_at >= now,
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def revoke_refresh_token(
        session: AsyncSession, token: str, commit: bool = True
    ) -> bool:
        result = await session.execute(
            update(RefreshToken)
            .where(RefreshToken.token == token)
            .values(revoked=True)
        )
        if commit:
            await session.commit()
        return bool(result.rowcount)
