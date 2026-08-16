from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from .base import SQLAlchemyBase


class Chunk(SQLAlchemyBase):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint(
            "asset_id",
            "chunk_id",
            name="uq_chunks_asset_chunk_id",
        ),
        CheckConstraint(
            "chunk_index > 0",
            name="ck_chunks_chunk_index_positive",
        ),
        CheckConstraint(
            "page_number > 0",
            name="ck_chunks_page_number_positive",
        ),
        CheckConstraint(
            "token_count >= 0",
            name="ck_chunks_token_count_non_negative",
        ),
        Index("ix_chunks_asset_page", "asset_id", "page_number"),
        Index("ix_chunks_asset_section", "asset_id", "section_title"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)

    asset_id = Column(
        Integer,
        ForeignKey("assets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Public chunk output fields.
    chunk_id = Column(String(50), nullable=False)
    document_name = Column(String(255), nullable=False, index=True)
    section_title = Column(String(500), nullable=False, default="Introduction")
    page_number = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)

    # Internal ingestion and retrieval fields.
    chunk_index = Column(Integer, nullable=False)
    token_count = Column(Integer, nullable=False, default=0, server_default="0")
    page_end = Column(Integer, nullable=True)
    chunk_metadata = Column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
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

    asset = relationship(
        "Asset",
        back_populates="chunks",
    )

    def to_output_dict(self) -> dict[str, str | int]:
        """Return the exact public parsing/chunking output contract."""
        return {
            "chunk_id": self.chunk_id,
            "document_name": self.document_name,
            "section_title": self.section_title,
            "page_number": self.page_number,
            "text": self.text,
        }

    def __repr__(self) -> str:
        return (
            f"<Chunk(id={self.id}, asset_id={self.asset_id}, "
            f"chunk_id={self.chunk_id!r})>"
        )
