"""Platform API against the real PostgreSQL (run inside docker compose).

Runs are created on the "gpu" tier so the CPU worker of the dev stack leaves them alone.
"""

import io
import json
import os
import shutil
import time
import zipfile
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import delete, select, update

from app.db.session import AsyncSessionLocal
from app.models.dataset import Dataset, DatasetVersion
from app.models.job import Environment, Job, JobStatus
from app.models.sandbox import Sandbox
from app.models.user import User
from app.services import storage
from app.services.retention import purge_expired
from app.services.uploads import sweep_stale_uploads
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
            shutil.rmtree(storage.predict_inputs_root() / user.id, ignore_errors=True)
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
    assert version["status"] == "ready"
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


async def test_a_folder_left_by_a_crashed_upload_does_not_block_uploads(client, register):
    headers, user = await register()
    dataset = (await client.post("/v1/datasets", json={"name": "pets"}, headers=headers)).json()
    orphan = storage.datasets_root() / "users" / user["id"] / dataset["id"] / "v1"
    orphan.mkdir(parents=True)
    (orphan / "stale.txt").write_text("left behind")

    response = await client.post(
        f"/v1/datasets/{dataset['id']}/versions",
        files={"file": ("pets.zip", _zip({"a.png": b"x"}), "application/zip")},
        headers=headers,
    )

    assert response.status_code == 201, response.text
    assert response.json()["number"] == 1
    assert not (orphan / "stale.txt").exists() and (orphan / "a.png").is_file()


async def test_failed_uploads_keep_their_number_and_are_not_listed(client, register):
    headers, _ = await register()
    dataset = (await client.post("/v1/datasets", json={"name": "pets"}, headers=headers)).json()
    url = f"/v1/datasets/{dataset['id']}/versions"

    bad = await client.post(
        url, files={"file": ("x.zip", b"plain text", "application/zip")}, headers=headers
    )
    good = await client.post(
        url,
        files={"file": ("pets.zip", _zip({"a.png": b"x"}), "application/zip")},
        headers=headers,
    )
    listed = (await client.get(f"/v1/datasets/{dataset['id']}", headers=headers)).json()
    async with AsyncSessionLocal() as db:
        rows = (
            await db.execute(
                select(DatasetVersion.number, DatasetVersion.status)
                .where(DatasetVersion.dataset_id == dataset["id"])
                .order_by(DatasetVersion.number)
            )
        ).all()

    assert bad.status_code == 400
    assert good.status_code == 201 and good.json()["number"] == 2
    assert [v["number"] for v in listed["versions"]] == [2]
    assert [tuple(row) for row in rows] == [(1, "failed"), (2, "ready")]


async def test_sweep_fails_interrupted_uploads_and_removes_their_files(client, register):
    headers, user = await register()
    dataset = (await client.post("/v1/datasets", json={"name": "pets"}, headers=headers)).json()
    relative = f"users/{user['id']}/{dataset['id']}"
    async with AsyncSessionLocal() as db:
        version = DatasetVersion(
            dataset_id=dataset["id"],
            number=1,
            path=f"{relative}/v1",
            files=0,
            size_bytes=0,
            status="uploading",
            created_at=datetime.now(UTC) - timedelta(days=1),
        )
        db.add(version)
        await db.commit()
    partial = storage.datasets_root() / relative / "v1.partial"
    orphan = storage.datasets_root() / relative / "v7"
    for folder in (partial, orphan):
        folder.mkdir(parents=True)
        (folder / "a.png").write_bytes(b"x")
        day_ago = time.time() - 24 * 60 * 60
        os.utime(folder, (day_ago, day_ago))

    async with AsyncSessionLocal() as db:
        failed = await sweep_stale_uploads(db, 60 * 60)
        status = await db.scalar(
            select(DatasetVersion.status).where(DatasetVersion.id == version.id)
        )

    assert version.id in failed and status == "failed"
    assert not partial.exists() and not orphan.exists()


async def test_deleted_datasets_can_be_restored(client, register):
    headers, user = await register()
    version = await _dataset_version(client, headers)
    url = f"/v1/datasets/{version['dataset_id']}"
    folder = storage.datasets_root() / "users" / user["id"] / version["dataset_id"]

    deleted = await client.delete(url, headers=headers)
    live = (await client.get("/v1/datasets", headers=headers)).json()["data"]
    trash = (await client.get("/v1/datasets?deleted=true", headers=headers)).json()["data"]
    upload = await client.post(
        f"{url}/versions",
        files={"file": ("pets.zip", _zip({"a.png": b"x"}), "application/zip")},
        headers=headers,
    )
    restored = await client.post(f"{url}/restore", headers=headers)
    restored_twice = await client.post(f"{url}/restore", headers=headers)

    assert deleted.status_code == 204 and folder.is_dir()  # files stay until purged
    assert live == [] and [d["id"] for d in trash] == [version["dataset_id"]]
    assert trash[0]["deleted_at"] and trash[0]["purge_after"]
    assert upload.status_code == 404
    assert restored.status_code == 200 and restored.json()["deleted_at"] is None
    assert [v["id"] for v in restored.json()["versions"]] == [version["id"]]
    assert restored_twice.status_code == 404


async def test_versions_can_be_deleted_and_restored(client, register):
    headers, _ = await register()
    version = await _dataset_version(client, headers)
    url = f"/v1/datasets/{version['dataset_id']}"

    deleted = await client.delete(f"{url}/versions/1", headers=headers)
    listed = (await client.get(url, headers=headers)).json()
    with_deleted = (await client.get(f"{url}?include_deleted=true", headers=headers)).json()
    profile = await client.post(
        f"{url}/versions/1/profile",
        json={"task": {"type": "classification.single_label"}, "dataset": {"format": "csv"}},
        headers=headers,
    )
    restored = await client.post(f"{url}/versions/1/restore", headers=headers)

    assert deleted.status_code == 204
    assert listed["versions"] == []
    assert with_deleted["versions"][0]["deleted_at"] and with_deleted["versions"][0]["purge_after"]
    assert profile.status_code == 404
    assert restored.status_code == 200 and restored.json()["deleted_at"] is None


async def test_retention_purges_only_what_has_expired(client, register):
    headers, user = await register()
    first = await _dataset_version(client, headers)
    dataset_a = first["dataset_id"]
    second = (
        await client.post(
            f"/v1/datasets/{dataset_a}/versions",
            files={"file": ("pets.zip", _zip({"a.png": b"x"}), "application/zip")},
            headers=headers,
        )
    ).json()
    dataset_b = (await _dataset_version(client, headers))["dataset_id"]
    dataset_c = (await _dataset_version(client, headers))["dataset_id"]
    await client.delete(f"/v1/datasets/{dataset_a}/versions/1", headers=headers)
    for dataset_id in (dataset_b, dataset_c):
        await client.delete(f"/v1/datasets/{dataset_id}", headers=headers)
    # Only data deleted before 2000-01-02 counts as expired, so other data in the dev
    # database is never touched by this test.
    long_ago = datetime(2000, 1, 1, tzinfo=UTC)
    retention = int((datetime.now(UTC) - datetime(2000, 1, 2, tzinfo=UTC)).total_seconds())
    async with AsyncSessionLocal() as db:
        await db.execute(
            update(DatasetVersion)
            .where(DatasetVersion.id == first["id"])
            .values(deleted_at=long_ago)
        )
        await db.execute(update(Dataset).where(Dataset.id == dataset_b).values(deleted_at=long_ago))
        await db.commit()
    inputs = storage.predict_inputs_root() / user["id"]
    old_inputs, new_inputs = inputs / "old", inputs / "new"
    for folder in (old_inputs, new_inputs):
        folder.mkdir(parents=True)
        (folder / "a.png").write_bytes(b"x")
    os.utime(old_inputs, (long_ago.timestamp(), long_ago.timestamp()))

    async with AsyncSessionLocal() as db:
        purged = await purge_expired(db, retention, retention)
        statuses = dict(
            (
                await db.execute(
                    select(DatasetVersion.id, DatasetVersion.status).where(
                        DatasetVersion.dataset_id == dataset_a
                    )
                )
            ).all()
        )
        dataset_b_row = await db.get(Dataset, dataset_b)
        dataset_c_row = await db.get(Dataset, dataset_c)
    restore = await client.post(f"/v1/datasets/{dataset_a}/versions/1/restore", headers=headers)

    root = storage.datasets_root() / "users" / user["id"]
    assert statuses == {first["id"]: "purged", second["id"]: "ready"}
    assert not (root / dataset_a / "v1").exists() and (root / dataset_a / "v2").is_dir()
    assert dataset_b_row is None and not (root / dataset_b).exists()
    assert dataset_c_row is not None and (root / dataset_c).is_dir()
    assert not old_inputs.exists() and new_inputs.is_dir()
    assert {first["id"], dataset_b, f"{user['id']}/old"} <= set(purged)
    assert restore.status_code == 404


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
    assert job.dataset_version_ids == [version["id"]]


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


async def _queued_run(client, headers) -> tuple[dict, dict, dict]:
    """(sandbox, dataset version, run) with the run queued on the gpu tier."""
    sandbox = await _sandbox(client, headers)
    version = await _dataset_version(client, headers)
    pipeline_version = await _pipeline_version(client, headers, sandbox["id"], version["id"])
    run = (
        await client.post(
            f"/v1/sandboxes/{sandbox['id']}/runs",
            json={"pipeline_version_id": pipeline_version["id"], "hardware_tier": "gpu"},
            headers=headers,
        )
    ).json()
    return sandbox, version, run


async def _delete_jobs(sandbox_id: str) -> None:
    async with AsyncSessionLocal() as db:
        await db.execute(delete(Job).where(Job.sandbox_id == sandbox_id))
        await db.commit()


async def test_data_of_active_runs_is_protected_and_reruns_check_their_data(client, register):
    headers, _ = await register()
    sandbox, version, run = await _queued_run(client, headers)
    dataset_url = f"/v1/datasets/{version['dataset_id']}"

    blocked = await client.delete(dataset_url, headers=headers)
    async with AsyncSessionLocal() as db:  # the run finishes
        await db.execute(
            update(Job).where(Job.id == run["id"]).values(status=JobStatus.COMPLETED)
        )
        await db.commit()
    deleted = await client.delete(dataset_url, headers=headers)
    rerun = await client.post(f"/v1/runs/{run['id']}/rerun", headers=headers)
    await client.post(f"{dataset_url}/restore", headers=headers)
    rerun_after_restore = await client.post(f"/v1/runs/{run['id']}/rerun", headers=headers)
    await _delete_jobs(sandbox["id"])

    assert blocked.status_code == 409 and blocked.json()["code"] == "dataset_in_use"
    assert deleted.status_code == 204
    assert rerun.status_code == 409 and rerun.json()["code"] == "inputs_missing"
    assert version["id"] in rerun.json()["detail"]
    assert rerun_after_restore.status_code == 202, rerun_after_restore.text


async def test_checkpoints_can_be_downloaded_by_their_owner(client, register):
    headers, _ = await register()
    other, _ = await register()
    sandbox, _, run = await _queued_run(client, headers)
    checkpoints = storage.run_dir(run["id"]) / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    (checkpoints / "best.pt").write_bytes(b"weights")
    url = f"/v1/runs/{run['id']}/checkpoints"
    try:
        best = await client.get(f"{url}/best", headers=headers)
        last = await client.get(f"{url}/last", headers=headers)
        foreign = await client.get(f"{url}/best", headers=other)
        invalid = await client.get(f"{url}/..%2F..%2Fsecret", headers=headers)
    finally:
        shutil.rmtree(storage.run_dir(run["id"]), ignore_errors=True)
        await _delete_jobs(sandbox["id"])

    assert best.status_code == 200 and best.content == b"weights"
    assert "attachment" in best.headers["content-disposition"]
    assert last.status_code == 404 and foreign.status_code == 404
    assert invalid.status_code in (404, 422)


async def test_predictions_run_in_the_training_environment(client, register):
    headers, user = await register()
    sandbox, version, run = await _queued_run(client, headers)
    environment_id = f"env_test_{uuid4().hex[:12]}"
    async with AsyncSessionLocal() as db:  # play the worker's part: a finished training run
        db.add(
            Environment(
                id=environment_id,
                template="pytorch-cpu",
                packs=[],
                system="x86_64-linux",
                engine_version="0.0.0",
                engine_snapshot="0" * 64,
                flake_nix="",
                status="ready",
            )
        )
        await db.flush()
        await db.execute(
            update(Job)
            .where(Job.id == run["id"])
            .values(status=JobStatus.COMPLETED, environment_id=environment_id)
        )
        await db.commit()
    checkpoints = storage.run_dir(run["id"]) / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    (checkpoints / "best.pt").write_bytes(b"weights")
    url = f"/v1/runs/{run['id']}/predict"
    gpu = {"hardware_tier": "gpu"}  # keeps the dev stack's CPU worker away
    try:
        uploaded = await client.post(
            url,
            files=[
                ("files", ("a b.png", b"png", "image/png")),
                ("files", ("scan.nii.gz", b"nii", "application/gzip")),
            ],
            data=gpu,
            headers=headers,
        )
        from_dataset = await client.post(
            url,
            data={**gpu, "dataset_version_id": version["id"], "subpath": "pets"},
            headers=headers,
        )
        unsupported = await client.post(
            url, files=[("files", ("notes.txt", b"x", "text/plain"))], data=gpu, headers=headers
        )
        both = await client.post(
            url,
            files=[("files", ("a.png", b"x", "image/png"))],
            data={**gpu, "dataset_version_id": version["id"]},
            headers=headers,
        )
        no_checkpoint = await client.post(
            url,
            data={**gpu, "dataset_version_id": version["id"], "checkpoint": "last"},
            headers=headers,
        )
        not_training = await client.post(
            f"/v1/runs/{uploaded.json()['id']}/predict",
            data={**gpu, "dataset_version_id": version["id"]},
            headers=headers,
        )
        async with AsyncSessionLocal() as db:
            upload_job = await db.get(Job, uploaded.json()["id"])
            dataset_job = await db.get(Job, from_dataset.json()["id"])
    finally:
        shutil.rmtree(storage.run_dir(run["id"]), ignore_errors=True)
        await _delete_jobs(sandbox["id"])
        async with AsyncSessionLocal() as db:
            await db.execute(delete(Environment).where(Environment.id == environment_id))
            await db.commit()

    assert uploaded.status_code == 202, uploaded.text
    assert upload_job.kind == "predict"
    assert upload_job.environment == {"environment_id": environment_id}
    assert upload_job.payload["source_run_id"] == run["id"]
    assert upload_job.payload["checkpoint"] == "best"
    assert upload_job.dataset_version_ids == []
    inputs = upload_job.payload["inputs"]
    folder = storage.predict_inputs_root() / inputs["root"]
    assert inputs["source"] == "upload"
    assert sorted(path.name for path in folder.iterdir()) == ["0000_a_b.png", "0001_scan.nii.gz"]
    assert from_dataset.status_code == 202, from_dataset.text
    assert dataset_job.payload["inputs"] == {
        "source": "dataset",
        "root": f"users/{user['id']}/{version['dataset_id']}/v1/pets",
    }
    assert dataset_job.dataset_version_ids == [version["id"]]
    assert unsupported.status_code == 400 and unsupported.json()["code"] == "unsupported_file"
    assert both.status_code == 422
    assert no_checkpoint.status_code == 404
    assert not_training.status_code == 409
