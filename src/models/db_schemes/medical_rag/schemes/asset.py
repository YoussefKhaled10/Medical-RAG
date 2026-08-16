from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import relationship

from .base import SQLAlchemyBase


class Asset(SQLAlchemyBase):
    __tablename__ = "assets"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "file_checksum",
            name="uq_assets_project_checksum",
        ),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)

    project_id = Column(
        Integer,
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    document_name = Column(String(255), nullable=False, index=True)
    file_name = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False)
    file_type = Column(String(100), nullable=True)
    file_size = Column(BigInteger, nullable=True)
    file_checksum = Column(String(64), nullable=True)

    processing_status = Column(
        String(50),
        nullable=False,
        default="PENDING",
        server_default="PENDING",
        index=True,
    )
    error_message = Column(Text, nullable=True)
    total_pages = Column(Integer, nullable=True)
    total_chunks = Column(Integer, nullable=False, default=0, server_default="0")
    parsed_at = Column(DateTime(timezone=True), nullable=True)

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

    project = relationship(
        "Project",
        back_populates="assets",
    )
    chunks = relationship(
        "Chunk",
        back_populates="asset",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Chunk.chunk_index",
    )

    def __repr__(self) -> str:
        return (
            f"<Asset(id={self.id}, document_name={self.document_name!r}, "
            f"status={self.processing_status!r})>"
        )
