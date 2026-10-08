from datetime import datetime
from typing import Any

from sqlalchemy import ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.common import created_at_column, prefixed_id


class Sandbox(Base):
    """A workspace for one CNN effort, with a pinned environment (PLAN §4)."""

    __tablename__ = "sandboxes"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=prefixed_id("sbx"))
    owner_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    template: Mapped[str] = mapped_column(String(64), nullable=False)
    # Pinned by the first run or editor session; later ones use the same engine snapshot.
    engine_snapshot: Mapped[str | None] = mapped_column(String(64), nullable=True)
    engine_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = created_at_column()

    def pin(self, engine_snapshot: str, engine_version: str) -> None:
        """Pin the engine the first time an environment is built for this sandbox (PLAN P9)."""
        if not self.engine_snapshot:
            self.engine_snapshot, self.engine_version = engine_snapshot, engine_version


class Pipeline(Base):
    __tablename__ = "pipelines"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=prefixed_id("pl"))
    sandbox_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("sandboxes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    created_at: Mapped[datetime] = created_at_column()


class PipelineVersion(Base):
    """An immutable saved state of a pipeline (the canvas JSON)."""

    __tablename__ = "pipeline_versions"
    __table_args__ = (UniqueConstraint("pipeline_id", "number"),)

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=prefixed_id("plv"))
    pipeline_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("pipelines.id", ondelete="CASCADE"), nullable=False, index=True
    )
    number: Mapped[int] = mapped_column(Integer, nullable=False)
    spec: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = created_at_column()
