"""Sweeps (PLAN Phase 8): many runs of one pipeline version with chosen values changed."""

from datetime import datetime
from typing import Any

from sqlalchemy import ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.common import created_at_column, prefixed_id


class Sweep(Base):
    __tablename__ = "sweeps"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=prefixed_id("swp"))
    owner_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sandbox_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("sandboxes.id", ondelete="CASCADE"), nullable=False, index=True
    )
    pipeline_version_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("pipeline_versions.id", ondelete="CASCADE"), nullable=False
    )
    # [{"path": "trainer.optimizer.lr", "values": [0.001, 0.0001]}, …]
    parameters: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    mode: Mapped[str] = mapped_column(String(16), nullable=False)  # grid | random
    run_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    created_at: Mapped[datetime] = created_at_column()
