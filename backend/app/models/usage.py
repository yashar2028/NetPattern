from datetime import datetime

from sqlalchemy import Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.common import created_at_column, prefixed_id


class UsageRecord(Base):
    """Append-only ledger (PLAN §11.4): compute time of a finished job or of a dedicated
    deployment, and its compute units (tier rate × seconds). No currency yet."""

    __tablename__ = "usage_records"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=prefixed_id("use"))
    owner_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sandbox_id: Mapped[str | None] = mapped_column(
        String(40), ForeignKey("sandboxes.id", ondelete="SET NULL"), nullable=True
    )
    # One of the two: the finished job, or the dedicated deployment that stayed loaded.
    job_id: Mapped[str | None] = mapped_column(String(40), nullable=True, unique=True)
    deployment_id: Mapped[str | None] = mapped_column(
        String(40), ForeignKey("deployments.id", ondelete="SET NULL"), nullable=True, index=True
    )
    hardware_tier: Mapped[str] = mapped_column(String(32), nullable=False)
    seconds: Mapped[float] = mapped_column(Float, nullable=False)
    units: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")
    created_at: Mapped[datetime] = created_at_column()
