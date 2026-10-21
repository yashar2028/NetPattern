import os
import shutil
import uuid

# Settings are read at import time. Inside docker compose the real values are
# already set; these defaults only let unit tests import the app elsewhere.
os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://netpattern:netpattern@localhost:5434/netpattern_dev"
)
os.environ.setdefault("AUTH_JWT_SECRET", "test-secret-not-for-production")

import pytest  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy import delete, text  # noqa: E402

from app.db.session import AsyncSessionLocal  # noqa: E402
from app.main import app  # noqa: E402
from app.models.user import User  # noqa: E402
from app.services import storage  # noqa: E402


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        yield http_client
    app.dependency_overrides.clear()


@pytest.fixture
def register(client):
    """Register a fresh user; returns (headers, user json)."""

    async def _register() -> tuple[dict[str, str], dict]:
        email = f"user-{uuid.uuid4().hex[:10]}@example.test"
        response = await client.post(
            "/v1/auth/register", json={"email": email, "password": "correct horse battery"}
        )
        assert response.status_code == 201, response.text
        body = response.json()
        return {"Authorization": f"Bearer {body['access_token']}"}, body["user"]

    return _register


@pytest.fixture(scope="session", autouse=True)
async def cleanup_test_users(request):
    yield
    # Unit-only runs (CI) have no database.
    if not any(item.get_closest_marker("integration") for item in request.session.items):
        return
    async with AsyncSessionLocal() as db:
        users = (
            await db.execute(User.__table__.select().where(User.email.like("%@example.test")))
        ).all()
        for user in users:
            shutil.rmtree(storage.datasets_root() / "users" / user.id, ignore_errors=True)
        await db.execute(delete(User).where(User.email.like("%@example.test")))
        # Teams whose members were all test users are left without members.
        await db.execute(
            text("DELETE FROM teams WHERE id NOT IN (SELECT team_id FROM team_members)")
        )
        await db.commit()
