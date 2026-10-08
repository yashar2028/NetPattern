"""Editor session forwarding: API -> worker internal API -> session manager."""

from types import SimpleNamespace

import httpx
import pytest
from httpx import ASGITransport

from app.api import sessions as sessions_api
from app.sandbox.session import SessionError
from app.workers import sessions as sessions_module
from app.workers.sessions import SessionManager, internal_api

pytestmark = pytest.mark.integration


class FakeSessions:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    async def call(self, sandbox_id, method, params):
        self.calls.append((sandbox_id, method, params))
        if method == "validate_architecture" and params.get("broken"):
            raise SessionError(
                4001, "invalid spec", {"issues": [{"node_id": "add", "message": "mismatch"}]}
            )
        return {"ok": True, "method": method}


@pytest.fixture
def fake_worker(monkeypatch):
    fake = FakeSessions()
    app = internal_api(fake)
    monkeypatch.setattr(
        sessions_api,
        "worker_client",
        lambda: httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://worker"),
    )
    return fake


async def _sandbox(client, headers):
    return (await client.post("/v1/sandboxes", json={"name": "s"}, headers=headers)).json()


async def test_editor_calls_reach_the_sandbox_session(client, register, fake_worker):
    headers, _ = await register()
    sandbox = await _sandbox(client, headers)

    response = await client.post(
        f"/v1/sandboxes/{sandbox['id']}/session/validate_architecture",
        json={"x": 1},
        headers=headers,
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True, "method": "validate_architecture"}
    assert fake_worker.calls == [(sandbox["id"], "validate_architecture", {"x": 1})]


async def test_session_errors_become_problem_json_with_issues(client, register, fake_worker):
    headers, _ = await register()
    sandbox = await _sandbox(client, headers)

    response = await client.post(
        f"/v1/sandboxes/{sandbox['id']}/session/validate_architecture",
        json={"broken": True},
        headers=headers,
    )

    assert response.status_code == 422
    assert response.json()["errors"] == [{"node_id": "add", "message": "mismatch"}]


async def test_only_editor_methods_on_own_sandboxes(client, register, fake_worker):
    alice, _ = await register()
    bob, _ = await register()
    sandbox = await _sandbox(client, alice)

    unknown = await client.post(f"/v1/sandboxes/{sandbox['id']}/session/shutdown", headers=alice)
    foreign = await client.post(f"/v1/sandboxes/{sandbox['id']}/session/schema", headers=bob)

    assert unknown.status_code == 404
    assert foreign.status_code == 404
    assert fake_worker.calls == []


async def test_worker_down_gives_503(client, register, monkeypatch):
    headers, _ = await register()
    sandbox = await _sandbox(client, headers)
    monkeypatch.setattr(
        sessions_api,
        "worker_client",
        lambda: httpx.AsyncClient(base_url="http://127.0.0.1:9", timeout=2),
    )

    response = await client.post(f"/v1/sandboxes/{sandbox['id']}/session/schema", headers=headers)

    assert response.status_code == 503 and response.json()["code"] == "worker_unavailable"


async def test_engine_upgrade_restarts_the_editor_session(client, register, monkeypatch):
    headers, _ = await register()
    sandbox = await _sandbox(client, headers)
    newest = {"snapshot": "a" * 64}
    started = []

    class FakeClient:
        def __init__(self, *args) -> None:
            self.process = None

        async def start(self):
            self.process = SimpleNamespace(returncode=None)
            started.append(self)

        async def close(self):
            self.process.returncode = 0

        async def call(self, method, params, timeout):
            return {"pong": True}

    class FakeEnvironments:
        async def ensure(self, db, template, packs, snapshot_id=None):
            snapshot = snapshot_id or newest["snapshot"]
            return SimpleNamespace(
                id="env", store_path="/nix/store/x", engine_snapshot=snapshot, engine_version="0"
            )

    monkeypatch.setattr(sessions_module, "SessionClient", FakeClient)
    manager = SessionManager(FakeEnvironments())

    await manager.call(sandbox["id"], "ping", {})
    await manager.call(sandbox["id"], "ping", {})
    newest["snapshot"] = "b" * 64
    await client.post(f"/v1/sandboxes/{sandbox['id']}/environment/upgrade", headers=headers)
    await manager.call(sandbox["id"], "ping", {})
    await manager.stop_all()

    assert len(started) == 2
    assert started[0].process.returncode == 0
    assert manager.sessions == {}
