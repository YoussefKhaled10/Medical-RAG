from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Integer, String, func

from .base import SQLAlchemyBase


class EmailVerification(SQLAlchemyBase):
    __tablename__ = "email_verifications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), nullable=False, index=True)
    code = Column(String(10), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    is_used = Column(Boolean, nullable=False, default=False, server_default="false")

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        return f"<EmailVerification(id={self.id}, email={self.email!r}, is_used={self.is_used})>"
