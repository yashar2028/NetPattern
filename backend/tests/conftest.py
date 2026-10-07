import os

# Settings are read at import time. Inside docker compose the real values are
# already set; these defaults only let unit tests import the app elsewhere.
os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://netpattern:netpattern@localhost:5434/netpattern_dev"
)

import pytest  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as http_client:
        yield http_client
    app.dependency_overrides.clear()
