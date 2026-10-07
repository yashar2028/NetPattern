import pytest

from app import __version__
from app.core.config import settings
from app.db.session import get_db
from app.main import app


class _HealthySession:
    async def execute(self, statement):
        return None


class _BrokenSession:
    async def execute(self, statement):
        raise ConnectionRefusedError("connection refused")


def _override_db(session):
    async def _get_db():
        yield session

    return _get_db


async def test_info_reports_service_and_api_version(client):
    response = await client.get("/v1/info")

    assert response.status_code == 200
    assert response.json() == {
        "name": "NetPattern",
        "version": __version__,
        "api_version": "v1",
        "environment": settings.APP_ENV,
    }
    assert response.headers["X-Request-Id"].startswith("req_")


async def test_health_ok_when_database_answers(client):
    app.dependency_overrides[get_db] = _override_db(_HealthySession())

    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok", "version": __version__}


async def test_health_returns_problem_json_when_database_is_down(client):
    app.dependency_overrides[get_db] = _override_db(_BrokenSession())

    response = await client.get("/health")

    assert response.status_code == 503
    assert response.headers["content-type"] == "application/problem+json"
    body = response.json()
    assert body["code"] == "database_unavailable"
    assert body["status"] == 503
    assert body["request_id"] == response.headers["X-Request-Id"]


async def test_unknown_route_returns_problem_json(client):
    response = await client.get("/v1/does-not-exist")

    assert response.status_code == 404
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["code"] == "not_found"


@pytest.mark.integration
async def test_health_against_real_database(client):
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["database"] == "ok"
