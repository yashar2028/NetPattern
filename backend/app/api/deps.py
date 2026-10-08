"""Shared API dependencies: the current user from a JWT or an API key."""

import hmac
from datetime import UTC, datetime

from fastapi import Depends, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import ApiError
from app.db.session import get_db
from app.models.user import ApiKey, User
from app.services.security import (
    TokenError,
    api_key_lookup,
    decode_access_token,
    hash_api_key,
    is_api_key,
)

bearer_scheme = HTTPBearer(auto_error=False)


def _unauthorized(detail: str) -> ApiError:
    return ApiError(status.HTTP_401_UNAUTHORIZED, "unauthorized", "Authentication required", detail)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    if credentials is None:
        raise _unauthorized("send `Authorization: Bearer <token or API key>`")
    credential = credentials.credentials

    if is_api_key(credential):
        candidates = (
            await db.execute(
                select(ApiKey).where(
                    ApiKey.lookup == api_key_lookup(credential), ApiKey.revoked_at.is_(None)
                )
            )
        ).scalars()
        digest = hash_api_key(credential)
        key = next((c for c in candidates if hmac.compare_digest(c.key_hash, digest)), None)
        if key is None:
            raise _unauthorized("the API key is invalid or revoked")
        key.last_used_at = datetime.now(UTC)
        await db.commit()
        user = await db.get(User, key.user_id)
    else:
        try:
            payload = decode_access_token(credential)
        except TokenError as error:
            raise _unauthorized(str(error)) from error
        user = await db.get(User, payload["sub"])

    if user is None:
        raise _unauthorized("the user no longer exists")
    return user


def not_found(what: str) -> ApiError:
    return ApiError(status.HTTP_404_NOT_FOUND, "not_found", f"{what} not found")
