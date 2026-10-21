"""Registered models, their deployments and the inference calls they serve (PLAN §8.5)."""

from datetime import datetime
from typing import Any

from sqlalchemy import Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.common import created_at_column, prefixed_id


class Model(Base):
    __tablename__ = "models"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=prefixed_id("mdl"))
    owner_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sandbox_id: Mapped[str | None] = mapped_column(
        String(40), ForeignKey("sandboxes.id", ondelete="SET NULL"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    created_at: Mapped[datetime] = created_at_column()


class ModelVersion(Base):
    """A trained checkpoint frozen with everything needed to serve it."""

    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("model_id", "number"),)

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=prefixed_id("mdv"))
    model_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("models.id", ondelete="CASCADE"), nullable=False, index=True
    )
    number: Mapped[int] = mapped_column(Integer, nullable=False)
    run_id: Mapped[str | None] = mapped_column(
        String(40), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True
    )
    # The run's pinned environment: deployments serve the model in exactly this environment.
    environment_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("environments.id"), nullable=False
    )
    # Folder in the storage volume: model.json, best.pt, parity samples.
    path: Mapped[str] = mapped_column(Text, nullable=False)
    task: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    input: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = created_at_column()


class DeploymentMode:
    SERVERLESS = "serverless"  # loaded on the first call, stopped when idle
    DEDICATED = "dedicated"  # kept loaded


class DeploymentStatus:
    STARTING = "starting"
    ACTIVE = "active"
    PAUSED = "paused"
    FAILED = "failed"


class Deployment(Base):
    __tablename__ = "deployments"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=prefixed_id("dep"))
    owner_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_version_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("model_versions.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    mode: Mapped[str] = mapped_column(String(16), nullable=False, default=DeploymentMode.SERVERLESS)
    hardware_tier: Mapped[str] = mapped_column(String(32), nullable=False, default="cpu")
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=DeploymentStatus.STARTING
    )
    # The worker that runs the model process (workers.id); reassigned if that worker is gone.
    worker_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # Result of the parity check run when the deployment started.
    parity: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = created_at_column()


class InferenceRecord(Base):
    """One predict call: the meter for inference (PLAN §11.2) and the request log."""

    __tablename__ = "inference_records"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=prefixed_id("inf"))
    deployment_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("deployments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    owner_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    api_key_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    images: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    # Time spent in the model process; billed as compute (PLAN §11.2).
    compute_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = created_at_column()
