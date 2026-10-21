"""The worker registry (PLAN Phase 8): which machines are up, what they serve, their URL."""

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.common import created_at_column


class WorkerNode(Base):
    __tablename__ = "workers"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)  # WORKER_ID
    # Internal API (editor sessions, deployments), e.g. http://worker:9000.
    url: Mapped[str] = mapped_column(Text, nullable=False)
    tiers: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    capacity: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    # Set for a dedicated machine: it serves only this account.
    account_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    running: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    started_at: Mapped[datetime] = created_at_column()
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
