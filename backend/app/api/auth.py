"""Accounts (email + password, PLAN D21) and API keys for programmatic access."""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, not_found
from app.core.errors import ApiError
from app.db.session import get_db
from app.models.user import ApiKey, User
from app.schemas.api import (
    ApiKeyCreate,
    ApiKeyCreated,
    ApiKeyOut,
    AuthResponse,
    LoginRequest,
    RegisterRequest,
    UserOut,
)
from app.services.security import (
    api_key_lookup,
    create_access_token,
    generate_api_key,
    hash_api_key,
    hash_password,
    verify_password,
)

router = APIRouter(prefix="/v1", tags=["auth"])


@router.post("/auth/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)) -> AuthResponse:
    if await db.scalar(select(User.id).where(User.email == body.email)):
        raise ApiError(status.HTTP_409_CONFLICT, "email_taken", "A user with this email exists")
    user = User(
        email=body.email, password_hash=hash_password(body.password), full_name=body.full_name
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return AuthResponse(
        access_token=create_access_token(user.id), user=UserOut.model_validate(user)
    )


@router.post("/auth/login", response_model=AuthResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)) -> AuthResponse:
    user = await db.scalar(select(User).where(User.email == body.email))
    if user is None or not verify_password(body.password, user.password_hash):
        raise ApiError(
            status.HTTP_401_UNAUTHORIZED, "invalid_credentials", "Invalid email or password"
        )
    return AuthResponse(
        access_token=create_access_token(user.id), user=UserOut.model_validate(user)
    )


@router.get("/auth/me", response_model=UserOut)
async def me(user: User = Depends(get_current_user)) -> User:
    return user


@router.get("/api-keys", response_model=list[ApiKeyOut], tags=["api keys"])
async def list_keys(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    query = select(ApiKey).where(ApiKey.user_id == user.id).order_by(ApiKey.created_at.desc())
    return (await db.execute(query)).scalars().all()


@router.post(
    "/api-keys",
    response_model=ApiKeyCreated,
    status_code=status.HTTP_201_CREATED,
    tags=["api keys"],
)
async def create_key(
    body: ApiKeyCreate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> ApiKeyCreated:
    key = generate_api_key()
    row = ApiKey(
        user_id=user.id, name=body.name, lookup=api_key_lookup(key), key_hash=hash_api_key(key)
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return ApiKeyCreated(**ApiKeyOut.model_validate(row).model_dump(), key=key)


@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["api keys"])
async def revoke_key(
    key_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> Response:
    row = await db.get(ApiKey, key_id)
    if row is None or row.user_id != user.id:
        raise not_found("API key")
    row.revoked_at = row.revoked_at or datetime.now(UTC)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
