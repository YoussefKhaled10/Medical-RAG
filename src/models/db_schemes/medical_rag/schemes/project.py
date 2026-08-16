from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text, func
from sqlalchemy.orm import relationship

from .base import SQLAlchemyBase


class Project(SQLAlchemyBase):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)

    source_name = Column(String(255), nullable=True)
    source_url = Column(Text, nullable=True)
    source_version = Column(String(100), nullable=True)
    license_name = Column(String(255), nullable=True)

    status = Column(
        String(50),
        nullable=False,
        default="ACTIVE",
        server_default="ACTIVE",
        index=True,
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

    assets = relationship(
        "Asset",
        back_populates="project",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"<Project(id={self.id}, name={self.name!r})>"
