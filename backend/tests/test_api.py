"""Platform API against the real PostgreSQL (run inside docker compose).

Runs are created on the "gpu" tier so the CPU worker of the dev stack leaves them alone.
"""

import io
import json
import shutil
import zipfile
from datetime import UTC, datetime

import pytest
from sqlalchemy import delete, update

from app.db.session import AsyncSessionLocal
from app.models.job import Job, JobStatus
from app.models.sandbox import Sandbox
from app.models.user import User
from app.services import storage
from app.workers import queue

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session", autouse=True)
async def cleanup_test_users():
    yield
    async with AsyncSessionLocal() as db:
        users = (
            await db.execute(User.__table__.select().where(User.email.like("%@example.test")))
        ).all()
        for user in users:
            shutil.rmtree(storage.datasets_root() / "users" / user.id, ignore_errors=True)
        await db.execute(delete(User).where(User.email.like("%@example.test")))
        await db.commit()


def _zip(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


SPEC = {
    "task": {"type": "classification.single_label"},
    "nodes": [
        {
            "id": "data",
            "type": "dataset",
            "params": {"format": "csv", "dataset_version_id": None, "subpath": "pets"},
        },
        {
            "id": "model",
            "type": "model",
            "params": {
                "architecture": {
                    "kind": "pretrained",
                    "base": {"source": "torchvision", "name": "resnet18"},
                }
            },
        },
        {"id": "train", "type": "trainer", "params": {"epochs": 1}},
    ],
}


async def _sandbox(client, headers, name="cnn"):
    response = await client.post("/v1/sandboxes", json={"name": name}, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()


async def _dataset_version(client, headers) -> dict:
    dataset = (await client.post("/v1/datasets", json={"name": "pets"}, headers=headers)).json()
    archive = _zip({"export/pets/manifest.csv": b"path,label\n", "export/pets/a.png": b"x"})
    response = await client.post(
        f"/v1/datasets/{dataset['id']}/versions",
        files={"file": ("pets.zip", archive, "application/zip")},
        headers=headers,
    )
    assert response.status_code == 201, response.text
    return response.json()


async def _pipeline_version(client, headers, sandbox_id, version_id) -> dict:
    spec = json.loads(json.dumps(SPEC))
    spec["nodes"][0]["params"]["dataset_version_id"] = version_id
    pipeline = (
        await client.post(
            f"/v1/sandboxes/{sandbox_id}/pipelines",
            json={"name": "p", "spec": spec},
            headers=headers,
        )
    ).json()
    versions = (
        await client.get(f"/v1/pipelines/{pipeline['id']}/versions", headers=headers)
    ).json()
    return versions[0]


# --------------------------------------------------------------------------- auth


async def test_register_login_and_me(client, register):
    headers, user = await register()

    me = await client.get("/v1/auth/me", headers=headers)
    duplicate = await client.post(
        "/v1/auth/register", json={"email": user["email"], "password": "another password"}
    )
    wrong = await client.post("/v1/auth/login", json={"email": user["email"], "password": "nope"})
    login = await client.post(
        "/v1/auth/login", json={"email": user["email"].upper(), "password": "correct horse battery"}
    )

    assert me.json()["id"] == user["id"]
    assert duplicate.status_code == 409 and duplicate.json()["code"] == "email_taken"
    assert wrong.status_code == 401
    assert login.status_code == 200 and login.json()["access_token"]


async def test_requests_without_credentials_get_problem_json(client):
    response = await client.get("/v1/sandboxes")

    assert response.status_code == 401
    assert response.headers["content-type"] == "application/problem+json"


async def test_api_keys_authenticate_until_revoked(client, register):
    headers, user = await register()
    created = (await client.post("/v1/api-keys", json={"name": "ci"}, headers=headers)).json()
    key_headers = {"Authorization": f"Bearer {created['key']}"}

    assert created["key"].startswith("np_")
    assert (await client.get("/v1/auth/me", headers=key_headers)).json()["id"] == user["id"]
    listed = (await client.get("/v1/api-keys", headers=headers)).json()
    assert "key" not in listed[0] and listed[0]["last_used_at"] is not None

    await client.delete(f"/v1/api-keys/{created['id']}", headers=headers)
    assert (await client.get("/v1/auth/me", headers=key_headers)).status_code == 401


# --------------------------------------------------------------------------- sandboxes & pipelines


async def test_sandboxes_are_private_and_paginated(client, register):
    alice, _ = await register()
    bob, _ = await register()
    for i in range(3):
        await _sandbox(client, alice, name=f"s{i}")

    first = (await client.get("/v1/sandboxes?limit=2", headers=alice)).json()
    second = (
        await client.get(f"/v1/sandboxes?limit=2&cursor={first['next_cursor']}", headers=alice)
    ).json()
    foreign = await client.get(f"/v1/sandboxes/{first['data'][0]['id']}", headers=bob)

    assert [s["name"] for s in first["data"] + second["data"]] == ["s2", "s1", "s0"]
    assert second["next_cursor"] is None
    assert foreign.status_code == 404
    assert (await client.get("/v1/sandboxes", headers=bob)).json()["data"] == []


async def test_pipeline_versions_are_immutable_and_numbered(client, register):
    headers, _ = await register()
    sandbox = await _sandbox(client, headers)
    pipeline = (
        await client.post(
            f"/v1/sandboxes/{sandbox['id']}/pipelines",
            json={"name": "p", "spec": SPEC},
            headers=headers,
        )
    ).json()

    second = await client.post(
        f"/v1/pipelines/{pipeline['id']}/versions",
        json={"spec": {**SPEC, "edges": []}},
        headers=headers,
    )
    invalid = await client.post(
        f"/v1/pipelines/{pipeline['id']}/versions", json={"spec": {"nodes": "x"}}, headers=headers
    )
    first = (await client.get(f"/v1/pipelines/{pipeline['id']}/versions/1", headers=headers)).json()

    assert pipeline["latest_version"] == 1
    assert second.json()["number"] == 2
    assert invalid.status_code == 422
    assert "edges" not in first["spec"]


# --------------------------------------------------------------------------- datasets


async def test_zip_upload_creates_a_version_and_strips_the_top_folder(client, register):
    headers, user = await register()

    version = await _dataset_version(client, headers)

    assert version["number"] == 1 and version["files"] == 2
    assert version["top_level"] == ["pets/"]
    folder = storage.datasets_root() / "users" / user["id"] / version["dataset_id"] / "v1" / "pets"
    assert (folder / "manifest.csv").is_file()


async def test_unsafe_archives_are_rejected(client, register):
    headers, _ = await register()
    dataset = (await client.post("/v1/datasets", json={"name": "bad"}, headers=headers)).json()
    url = f"/v1/datasets/{dataset['id']}/versions"

    escaping = await client.post(
        url,
        files={"file": ("x.zip", _zip({"../../evil.txt": b"x"}), "application/zip")},
        headers=headers,
    )
    not_zip = await client.post(
        url, files={"file": ("x.zip", b"plain text", "application/zip")}, headers=headers
    )

    assert escaping.status_code == 400 and escaping.json()["code"] == "invalid_archive"
    assert not_zip.status_code == 400


# --------------------------------------------------------------------------- runs


async def test_run_resolves_datasets_and_uses_the_pinned_engine(client, register):
    headers, user = await register()
    sandbox = await _sandbox(client, headers)
    version = await _dataset_version(client, headers)
    pipeline_version = await _pipeline_version(client, headers, sandbox["id"], version["id"])
    async with AsyncSessionLocal() as db:
        await db.execute(
            update(Sandbox).where(Sandbox.id == sandbox["id"]).values(engine_snapshot="ab" * 32)
        )
        await db.commit()

    response = await client.post(
        f"/v1/sandboxes/{sandbox['id']}/runs",
        json={"pipeline_version_id": pipeline_version["id"], "hardware_tier": "gpu"},
        headers=headers,
    )

    assert response.status_code == 202, response.text
    async with AsyncSessionLocal() as db:
        job = await db.get(Job, response.json()["id"])
    params = job.payload["pipeline"]["nodes"][0]["params"]
    assert params == {
        "format": "csv",
        "root": f"users/{user['id']}/{version['dataset_id']}/v1/pets",
    }
    assert job.environment == {"template": "pytorch-cpu", "engine_snapshot": "ab" * 32}
    assert job.owner_id == user["id"] and job.sandbox_id == sandbox["id"]


async def test_runs_cannot_reach_other_users_data(client, register):
    alice, _ = await register()
    bob, _ = await register()
    alice_version = await _dataset_version(client, alice)
    bob_sandbox = await _sandbox(client, bob)
    pipeline_version = await _pipeline_version(client, bob, bob_sandbox["id"], alice_version["id"])
    spec_with_root = json.loads(json.dumps(SPEC))
    spec_with_root["nodes"][0]["params"] = {"format": "csv", "root": "users"}
    pipeline = (
        await client.post(
            f"/v1/sandboxes/{bob_sandbox['id']}/pipelines",
            json={"name": "r", "spec": spec_with_root},
            headers=bob,
        )
    ).json()
    root_version = (
        await client.get(f"/v1/pipelines/{pipeline['id']}/versions", headers=bob)
    ).json()[0]

    foreign = await client.post(
        f"/v1/sandboxes/{bob_sandbox['id']}/runs",
        json={"pipeline_version_id": pipeline_version["id"], "hardware_tier": "gpu"},
        headers=bob,
    )
    raw_root = await client.post(
        f"/v1/sandboxes/{bob_sandbox['id']}/runs",
        json={"pipeline_version_id": root_version["id"], "hardware_tier": "gpu"},
        headers=bob,
    )

    assert foreign.status_code == 422 and "not found" in foreign.json()["detail"]
    assert raw_root.status_code == 422


async def test_events_stream_cancel_rerun_and_usage(client, register):
    headers, _ = await register()
    sandbox = await _sandbox(client, headers)
    version = await _dataset_version(client, headers)
    pipeline_version = await _pipeline_version(client, headers, sandbox["id"], version["id"])
    body = {"pipeline_version_id": pipeline_version["id"], "hardware_tier": "gpu"}
    run = (
        await client.post(f"/v1/sandboxes/{sandbox['id']}/runs", json=body, headers=headers)
    ).json()
    queued = (
        await client.post(f"/v1/sandboxes/{sandbox['id']}/runs", json=body, headers=headers)
    ).json()

    async with AsyncSessionLocal() as db:  # play the worker's part
        await db.execute(
            update(Job)
            .where(Job.id == run["id"])
            .values(status=JobStatus.RUNNING, started_at=datetime.now(UTC))
        )
        await db.commit()
        for seq, kind in enumerate(["run_started", "epoch_end", "run_completed"], start=1):
            await queue.add_event(db, run["id"], seq, kind, {"epoch": seq})
        await queue.finish(db, run["id"], JobStatus.COMPLETED, {"status": "completed"})

    events = (await client.get(f"/v1/runs/{run['id']}/events?after=1", headers=headers)).json()
    stream = await client.get(f"/v1/runs/{run['id']}/events/stream", headers=headers)
    cancelled = (await client.post(f"/v1/runs/{queued['id']}/cancel", headers=headers)).json()
    rerun = await client.post(f"/v1/runs/{run['id']}/rerun", headers=headers)
    detail = (await client.get(f"/v1/runs/{run['id']}", headers=headers)).json()
    usage = (await client.get("/v1/usage", headers=headers)).json()
    listed = (
        await client.get(f"/v1/runs?sandbox_id={sandbox['id']}&status=completed", headers=headers)
    ).json()

    assert [e["type"] for e in events] == ["epoch_end", "run_completed"]
    assert stream.headers["content-type"].startswith("text/event-stream")
    assert "id: 1\nevent: run_started" in stream.text and "event: end" in stream.text
    assert cancelled["status"] == "cancelled"
    assert rerun.status_code == 202 and rerun.json()["id"] != run["id"]
    assert detail["status"] == "completed" and detail["result"] == {"status": "completed"}
    assert usage["jobs"] == 1 and usage["by_tier"]["gpu"] >= 0
    assert [r["id"] for r in listed["data"]] == [run["id"]]

    async with AsyncSessionLocal() as db:  # don't leave gpu jobs queued for a GPU worker
        await db.execute(delete(Job).where(Job.sandbox_id == sandbox["id"]))
        await db.commit()


async def test_sandbox_with_active_runs_cannot_be_deleted(client, register):
    headers, _ = await register()
    sandbox = await _sandbox(client, headers)
    version = await _dataset_version(client, headers)
    pipeline_version = await _pipeline_version(client, headers, sandbox["id"], version["id"])
    await client.post(
        f"/v1/sandboxes/{sandbox['id']}/runs",
        json={"pipeline_version_id": pipeline_version["id"], "hardware_tier": "gpu"},
        headers=headers,
    )

    blocked = await client.delete(f"/v1/sandboxes/{sandbox['id']}", headers=headers)
    async with AsyncSessionLocal() as db:
        await db.execute(delete(Job).where(Job.sandbox_id == sandbox["id"]))
        await db.commit()
    deleted = await client.delete(f"/v1/sandboxes/{sandbox['id']}", headers=headers)

    assert blocked.status_code == 409
    assert deleted.status_code == 204
