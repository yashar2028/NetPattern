from datetime import datetime

from sqlalchemy import Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.common import created_at_column, prefixed_id


class UsageRecord(Base):
    """Append-only ledger: compute time per finished job (PLAN §11.4). No prices yet."""

    __tablename__ = "usage_records"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=prefixed_id("use"))
    owner_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sandbox_id: Mapped[str | None] = mapped_column(
        String(40), ForeignKey("sandboxes.id", ondelete="SET NULL"), nullable=True
    )
    job_id: Mapped[str] = mapped_column(String(40), nullable=False, unique=True)
    hardware_tier: Mapped[str] = mapped_column(String(32), nullable=False)
    seconds: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = created_at_column()
