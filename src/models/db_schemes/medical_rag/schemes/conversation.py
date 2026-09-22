from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    func,
)
from sqlalchemy.orm import relationship

from .base import SQLAlchemyBase


class Conversation(SQLAlchemyBase):
    """Persisted chat conversation owned by one user."""

    __tablename__ = "conversations"
    __table_args__ = (
        Index(
            "ix_conversations_user_updated",
            "user_id",
            "updated_at",
        ),
    )

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    title = Column(
        String(160),
        nullable=False,
        default="New Conversation",
        server_default="New Conversation",
    )

    title_source = Column(
        String(16),
        nullable=False,
        default="auto",
        server_default="auto",
    )

    message_count = Column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    last_message_preview = Column(
        String(280),
        nullable=True,
    )

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    user = relationship(
        "User",
        foreign_keys=[user_id],
    )

    messages = relationship(
        "ChatMessage",
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ChatMessage.created_at",
    )

    def __repr__(self) -> str:
        return (
            f"<Conversation(id={self.id}, user_id={self.user_id}, "
            f"title={self.title!r}, message_count={self.message_count})>"
        )
