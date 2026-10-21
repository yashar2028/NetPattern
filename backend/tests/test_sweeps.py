"""Sweeps: each combination becomes a labeled training run with the patched spec."""

import pytest
from sqlalchemy import select

from app.api.sweeps import SweepParameter, apply, combinations
from app.core.errors import ApiError
from app.db.session import AsyncSessionLocal
from app.models.job import Job

pytestmark = pytest.mark.integration

SPEC = {
    "task": {"type": "classification.single_label"},
    "nodes": [
        {"id": "data", "type": "dataset", "params": {"format": "synthetic"}},
        {
            "id": "model",
            "type": "model",
            "params": {"architecture": {"kind": "pretrained", "base": {"name": "resnet18"}}},
        },
        {"id": "train", "type": "trainer", "params": {"epochs": 1}},
    ],
    "edges": [["data", "model"], ["model", "train"]],
}


def test_paths_patch_nested_params_and_grids_have_a_limit():
    patched = apply(SPEC, {"trainer.optimizer.lr": 0.01, "dataset.split.fold": 2})
    assert patched["nodes"][2]["params"] == {"epochs": 1, "optimizer": {"lr": 0.01}}
    assert patched["nodes"][0]["params"]["split"] == {"fold": 2}
    assert SPEC["nodes"][2]["params"] == {"epochs": 1}  # the original is untouched

    params = [SweepParameter(path="trainer.epochs", values=[1, 2, 3])]
    assert len(combinations(params * 2, "random", 4)) == 4
    with pytest.raises(ApiError) as caught:
        combinations(params * 2, "grid", 4)
    assert "more than max_runs" in caught.value.detail


async def test_a_sweep_starts_one_labeled_run_per_combination(client, register):
    headers, _ = await register()
    sandbox = (await client.post("/v1/sandboxes", json={"name": "s"}, headers=headers)).json()
    pipeline = (
        await client.post(
            f"/v1/sandboxes/{sandbox['id']}/pipelines",
            json={"name": "p", "spec": SPEC},
            headers=headers,
        )
    ).json()
    version = (
        await client.get(f"/v1/pipelines/{pipeline['id']}/versions/1", headers=headers)
    ).json()

    response = await client.post(
        f"/v1/sandboxes/{sandbox['id']}/sweeps",
        json={
            "pipeline_version_id": version["id"],
            "parameters": [
                {"path": "trainer.optimizer.lr", "values": [0.01, 0.001]},
                {"path": "trainer.batch_size", "values": [8, 16]},
            ],
            "hardware_tier": "gpu",
        },
        headers=headers,
    )
    unknown_node = await client.post(
        f"/v1/sandboxes/{sandbox['id']}/sweeps",
        json={
            "pipeline_version_id": version["id"],
            "parameters": [{"path": "evaluator.splits", "values": [["val"]]}],
        },
        headers=headers,
    )

    assert response.status_code == 201, response.text
    sweep = response.json()
    labels = [run["labels"] for run in sweep["runs"]]
    assert len(labels) == 4 and {"trainer.optimizer.lr": 0.001, "trainer.batch_size": 16} in labels
    async with AsyncSessionLocal() as db:
        jobs = await db.execute(select(Job).where(Job.id.in_([r["id"] for r in sweep["runs"]])))
        trainers = [
            next(n for n in job.payload["pipeline"]["nodes"] if n["type"] == "trainer")["params"]
            for job in jobs.scalars()
        ]
    assert {(t["optimizer"]["lr"], t["batch_size"]) for t in trainers} == {
        (0.01, 8),
        (0.01, 16),
        (0.001, 8),
        (0.001, 16),
    }
    assert unknown_node.status_code == 422 and "no evaluator node" in unknown_node.json()["detail"]
    listed = (await client.get(f"/v1/sandboxes/{sandbox['id']}/sweeps", headers=headers)).json()
    assert [s["id"] for s in listed] == [sweep["id"]]
    for run in sweep["runs"]:
        await client.post(f"/v1/runs/{run['id']}/cancel", headers=headers)
