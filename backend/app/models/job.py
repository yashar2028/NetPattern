"""Jobs (the work queue), their live events, and pinned sandbox environments."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from sqlalchemy import BigInteger, Boolean, DateTime
from sqlalchemy import Enum as SQLEnum
from sqlalchemy import ForeignKey, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


def new_job_id() -> str:
    return f"job_{uuid4().hex}"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


FINISHED_STATUSES = (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)


class Job(Base):
    """One unit of sandbox work: a training run, a sanity check, a dataset profile, …"""

    __tablename__ = "jobs"
    __table_args__ = (Index("ix_jobs_claim", "status", "hardware_tier", "priority", "created_at"),)

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=new_job_id)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[JobStatus] = mapped_column(
        SQLEnum(
            JobStatus,
            name="job_status",
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
        default=JobStatus.QUEUED,
    )
    hardware_tier: Mapped[str] = mapped_column(String(32), nullable=False, default="cpu")
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Set for jobs started through the API; CLI jobs have no owner.
    owner_id: Mapped[str | None] = mapped_column(
        String(40), ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True
    )
    sandbox_id: Mapped[str | None] = mapped_column(
        String(40), ForeignKey("sandboxes.id", ondelete="CASCADE"), nullable=True, index=True
    )
    pipeline_version_id: Mapped[str | None] = mapped_column(
        String(40), ForeignKey("pipeline_versions.id", ondelete="SET NULL"), nullable=True
    )
    # The engine request (pipeline, dataset, options) without the run id.
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    # Requested environment: {"template": ..., "packs": [...]} (packs are added as needed).
    environment: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    environment_id: Mapped[str | None] = mapped_column(
        String(40), ForeignKey("environments.id"), nullable=True
    )
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    worker_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    run_dir: Mapped[str | None] = mapped_column(Text, nullable=True)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class JobEvent(Base):
    """One event streamed from the engine (batch, epoch_end, evaluation, …)."""

    __tablename__ = "job_events"
    __table_args__ = (UniqueConstraint("job_id", "seq"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class EnvironmentStatus(str, Enum):
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"


class Environment(Base):
    """A pinned sandbox environment: template + packs + engine snapshot (PLAN §4.2)."""

    __tablename__ = "environments"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    template: Mapped[str] = mapped_column(String(64), nullable=False)
    packs: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    system: Mapped[str] = mapped_column(String(32), nullable=False)
    engine_version: Mapped[str] = mapped_column(String(32), nullable=False)
    engine_snapshot: Mapped[str] = mapped_column(String(64), nullable=False)
    flake_nix: Mapped[str] = mapped_column(Text, nullable=False)
    flake_lock: Mapped[str | None] = mapped_column(Text, nullable=True)
    store_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    # The hash part of the store path: what users see as the "environment hash".
    env_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=EnvironmentStatus.BUILDING.value
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # The engine's JSON Schema + capabilities, for rendering forms (PLAN P6).
    engine_schema: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    built_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
