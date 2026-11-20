"""Model registry, deployments and the inference API, with a fake worker."""

import base64
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from fastapi import Body, FastAPI
from httpx import ASGITransport
from sqlalchemy import delete

from app.api import deployments as deployments_api
from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.models.job import Environment, Job, JobStatus
from app.models.model import Model

pytestmark = pytest.mark.integration

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9aw"
    "AAAABJRU5ErkJggg=="
)


class FakeWorker:
    def __init__(self) -> None:
        self.parity: dict = {"passed": True, "samples": 4, "max_difference": 0.0}
        self.calls: list[tuple[str, str, dict]] = []

    def app(self) -> FastAPI:
        app = FastAPI()

        @app.post("/deployments/{deployment_id}/start")
        async def start(deployment_id: str):
            self.calls.append(("start", deployment_id, {}))
            return {"result": self.parity}

        @app.post("/deployments/{deployment_id}/predict")
        async def predict(deployment_id: str, body: dict = Body(default={})):
            self.calls.append(("predict", deployment_id, body))
            files = body["files"]
            assert all(Path(f).is_file() for f in files)  # inputs are on disk for the model
            return {
                "result": {
                    "task": "classification.single_label",
                    "outputs": [{"label": "cat", "score": 0.9} for _ in files],
                    "timing_ms": {"inference": 12.5},
                }
            }

        @app.post("/deployments/{deployment_id}/stop")
        async def stop(deployment_id: str):
            self.calls.append(("stop", deployment_id, {}))
            return {"result": {"stopped": True}}

        return app


@pytest.fixture
def fake_worker(monkeypatch, tmp_path):
    fake = FakeWorker()
    app = fake.app()
    monkeypatch.setattr(
        deployments_api,
        "worker_client",
        lambda *args: httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://worker"),
    )
    monkeypatch.setattr(settings, "STORAGE_ROOT", str(tmp_path))
    return fake


@pytest.fixture
async def trained_run(client, register, tmp_path):
    """A completed training run with a model description, as the engine leaves it."""
    headers, user = await register()
    sandbox = (await client.post("/v1/sandboxes", json={"name": "s"}, headers=headers)).json()
    env_id = f"env_test_{uuid4().hex[:12]}"
    run_dir = tmp_path / "runs" / env_id
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "best.pt").write_bytes(b"weights")
    (run_dir / "model.json").write_text(json.dumps({"format": "netpattern-model/1"}))
    (run_dir / "parity.json").write_text("[]")
    async with AsyncSessionLocal() as db:
        db.add(
            Environment(
                id=env_id,
                template="pytorch-cpu",
                packs=[],
                system="x86_64-linux",
                engine_version="0.3.0",
                engine_snapshot="f" * 64,
                flake_nix="{}",
                store_path="/nix/store/test",
                env_hash="testhash",
                status="ready",
            )
        )
        job = Job(
            kind="train",
            status=JobStatus.COMPLETED,
            hardware_tier="gpu",
            owner_id=user["id"],
            sandbox_id=sandbox["id"],
            environment_id=env_id,
            run_dir=str(run_dir),
            finished_at=datetime.now(UTC),
            result={
                "task": {"type": "classification.single_label", "classes": ["cat", "dog"]},
                "model": {"input": {"channels": 3, "size": [224, 224]}},
                "evaluation": {"test": {"metrics": {"accuracy": 0.9}}},
            },
        )
        db.add(job)
        await db.commit()
        run_id = job.id
    yield headers, run_id
    async with AsyncSessionLocal() as db:
        await db.execute(delete(Model).where(Model.owner_id == user["id"]))
        await db.execute(delete(Job).where(Job.id == run_id))
        await db.execute(delete(Environment).where(Environment.id == env_id))
        await db.commit()


async def _deploy(client, headers, run_id, **extra):
    model = (
        await client.post("/v1/models", json={"run_id": run_id, "name": "pets"}, headers=headers)
    ).json()
    version = model["versions"][0]
    response = await client.post(
        "/v1/deployments",
        json={"model_version_id": version["id"], "name": "pets-api", **extra},
        headers=headers,
    )
    assert response.status_code == 201, response.text
    return model, response.json()


async def test_register_deploy_and_predict_with_an_inference_key(client, fake_worker, trained_run):
    headers, run_id = trained_run
    model, deployment = await _deploy(client, headers, run_id)
    assert model["versions"][0]["metrics"] == {"test": {"accuracy": 0.9}}
    assert Path(settings.STORAGE_ROOT, "models", model["versions"][0]["id"], "best.pt").is_file()
    assert deployment["status"] == "active" and deployment["parity"]["passed"]

    key = (
        await client.post(
            "/v1/api-keys",
            json={"name": "prod", "scope": "inference", "deployment_id": deployment["id"]},
            headers=headers,
        )
    ).json()["key"]
    auth = {"Authorization": f"Bearer {key}"}

    multipart = await client.post(
        f"/v1/deployments/{deployment['id']}/predict",
        files=[("files", ("a.png", PNG, "image/png")), ("files", ("b.png", PNG, "image/png"))],
        data={"options": json.dumps({"top_k": 1})},
        headers=auth,
    )
    as_json = await client.post(
        f"/v1/deployments/{deployment['id']}/predict",
        json={"inputs": [{"image_base64": base64.b64encode(PNG).decode()}]},
        headers=auth,
    )
    schema = await client.get(f"/v1/deployments/{deployment['id']}/schema", headers=auth)
    forbidden = await client.get("/v1/sandboxes", headers=auth)
    log = await client.get(f"/v1/deployments/{deployment['id']}/requests", headers=headers)
    usage = (await client.get("/v1/usage", headers=headers)).json()

    assert multipart.status_code == 200, multipart.text
    body = multipart.json()
    assert body["outputs"] == [{"label": "cat", "score": 0.9}] * 2
    assert body["model_version"] == f"{model['id']}@1" and body["environment_hash"] == "testhash"
    assert fake_worker.calls[-2][2]["options"] == {"top_k": 1}
    assert as_json.status_code == 200 and len(as_json.json()["outputs"]) == 1
    assert schema.json()["classes"] == ["cat", "dog"]
    assert forbidden.status_code == 403 and forbidden.json()["code"] == "insufficient_scope"
    assert [r["images"] for r in log.json()] == [1, 2]
    assert usage["inference_calls"] == 2 and usage["inference_images"] == 3


async def test_bad_inputs_other_deployments_and_rate_limits(
    client, fake_worker, trained_run, monkeypatch
):
    headers, run_id = trained_run
    _, first = await _deploy(client, headers, run_id)
    _, second = await _deploy(client, headers, run_id)
    key = (
        await client.post(
            "/v1/api-keys",
            json={"name": "k", "scope": "inference", "deployment_id": first["id"]},
            headers=headers,
        )
    ).json()["key"]
    auth = {"Authorization": f"Bearer {key}"}

    not_an_image = await client.post(
        f"/v1/deployments/{first['id']}/predict",
        files=[("files", ("x.png", b"plain text", "image/png"))],
        headers=auth,
    )
    other = await client.post(
        f"/v1/deployments/{second['id']}/predict",
        files=[("files", ("a.png", PNG, "image/png"))],
        headers=auth,
    )
    monkeypatch.setattr(settings, "INFERENCE_RATE_LIMIT_PER_MINUTE", 0)
    limited = await client.post(
        f"/v1/deployments/{first['id']}/predict",
        files=[("files", ("a.png", PNG, "image/png"))],
        headers=auth,
    )

    assert not_an_image.status_code == 422 and not_an_image.json()["code"] == "invalid_input"
    assert other.status_code == 403
    assert limited.status_code == 429


async def test_a_failed_parity_check_blocks_traffic(client, fake_worker, trained_run):
    headers, run_id = trained_run
    fake_worker.parity = {"passed": False, "samples": 4, "mismatches": ["parity/0.png"]}
    _, deployment = await _deploy(client, headers, run_id, mode="dedicated")

    response = await client.post(
        f"/v1/deployments/{deployment['id']}/predict",
        files=[("files", ("a.png", PNG, "image/png"))],
        headers=headers,
    )
    paused = await client.post(f"/v1/deployments/{deployment['id']}/pause", headers=headers)

    assert deployment["status"] == "failed" and "parity" in deployment["error"]
    assert response.status_code == 409
    assert paused.json()["status"] == "paused"
    assert ("stop", deployment["id"], {}) in fake_worker.calls


async def test_run_files_are_served_only_from_the_run_directory(client, fake_worker, trained_run):
    headers, run_id = trained_run

    found = await client.get(f"/v1/runs/{run_id}/files/model.json", headers=headers)
    escaped = await client.get(f"/v1/runs/{run_id}/files/%2E%2E/%2E%2E/etc/passwd", headers=headers)
    missing = await client.get(f"/v1/runs/{run_id}/files/nope.png", headers=headers)

    assert found.status_code == 200 and found.json()["format"] == "netpattern-model/1"
    assert escaped.status_code == 404 and missing.status_code == 404


async def test_report_and_reproduce_bundle(client, fake_worker, trained_run):
    headers, run_id = trained_run

    report = await client.get(f"/v1/runs/{run_id}/report.html", headers=headers)
    bundle = await client.get(f"/v1/runs/{run_id}/reproduce.zip", headers=headers)

    assert report.status_code == 200 and "NetPattern run report" in report.text
    assert "accuracy" in report.text
    # The test environment points at no engine snapshot on disk.
    assert bundle.status_code == 410 and bundle.json()["code"] == "snapshot_missing"
