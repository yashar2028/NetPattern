from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.common import created_at_column, prefixed_id

# A version row is created before its archive is extracted, so a crash in between
# leaves an "uploading" row (swept later) instead of a folder without a row.
# "purged": deleted, retention period over, files removed; the row keeps its number taken.
UPLOADING, READY, FAILED, PURGED = "uploading", "ready", "failed", "purged"


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=prefixed_id("ds"))
    owner_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    created_at: Mapped[datetime] = created_at_column()
    # Soft delete: restorable until the retention period ends, then purged.
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class DatasetVersion(Base):
    """One uploaded archive, extracted once and never changed afterwards."""

    __tablename__ = "dataset_versions"
    __table_args__ = (UniqueConstraint("dataset_id", "number"),)

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=prefixed_id("dsv"))
    dataset_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    number: Mapped[int] = mapped_column(Integer, nullable=False)
    # Folder relative to the datasets root, e.g. users/usr_…/ds_…/v1
    path: Mapped[str] = mapped_column(Text, nullable=False)
    files: Mapped[int] = mapped_column(Integer, nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, server_default=READY)
    created_at: Mapped[datetime] = created_at_column()
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
