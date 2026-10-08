"""Cursor pagination (PLAN §8.1): `?limit=&cursor=` -> `{"data": [...], "next_cursor": ...}`.

Lists are ordered newest first; the cursor encodes the last item's (created_at, id).
"""

import base64
import json
from datetime import datetime
from typing import Any, Generic, TypeVar

from fastapi import status
from pydantic import BaseModel
from sqlalchemy import Select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import ApiError

T = TypeVar("T")
DEFAULT_LIMIT = 20
MAX_LIMIT = 100


class Page(BaseModel, Generic[T]):
    data: list[T]
    next_cursor: str | None = None


def _encode(created_at: datetime, item_id: str) -> str:
    raw = json.dumps([created_at.isoformat(), item_id]).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode(cursor: str) -> tuple[datetime, str]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        created_at, item_id = json.loads(base64.urlsafe_b64decode(padded))
        return datetime.fromisoformat(created_at), str(item_id)
    except (ValueError, TypeError) as error:
        raise ApiError(status.HTTP_400_BAD_REQUEST, "invalid_cursor", "Invalid cursor") from error


async def paginate(
    db: AsyncSession, query: Select, model: Any, limit: int, cursor: str | None
) -> tuple[list[Any], str | None]:
    limit = max(1, min(limit, MAX_LIMIT))
    query = query.order_by(model.created_at.desc(), model.id.desc())
    if cursor:
        created_at, item_id = _decode(cursor)
        query = query.where(tuple_(model.created_at, model.id) < tuple_(created_at, item_id))
    rows = list((await db.execute(query.limit(limit + 1))).scalars().all())
    if len(rows) <= limit:
        return rows, None
    last = rows[limit - 1]
    return rows[:limit], _encode(last.created_at, last.id)
