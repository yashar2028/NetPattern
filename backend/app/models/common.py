from datetime import datetime
from uuid import uuid4

from sqlalchemy import DateTime, func
from sqlalchemy.orm import Mapped, mapped_column


def prefixed_id(prefix: str):
    """Default factory for ids like `sbx_3f2a…` (readable in logs and URLs)."""

    def factory() -> str:
        return f"{prefix}_{uuid4().hex}"

    return factory


def created_at_column() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
