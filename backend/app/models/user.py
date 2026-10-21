from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text, false
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.common import created_at_column, prefixed_id


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=prefixed_id("usr"))
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    # Dedicated machine (PLAN §11.6): the account's jobs run only on its own worker
    # (WORKER_ACCOUNT); shared workers skip them.
    dedicated_worker: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=false())
    created_at: Mapped[datetime] = created_at_column()


class ApiKey(Base):
    """A key for programmatic access. Only a hash is stored; the key is shown once."""

    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=prefixed_id("key"))
    user_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    lookup: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    key_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    # "all": the whole API as the user; "inference": only schema/predict of deployments.
    scope: Mapped[str] = mapped_column(String(16), nullable=False, server_default="all")
    # Optional: an inference key limited to one deployment.
    deployment_id: Mapped[str | None] = mapped_column(
        String(40), ForeignKey("deployments.id", ondelete="CASCADE"), nullable=True
    )
    created_at: Mapped[datetime] = created_at_column()
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
