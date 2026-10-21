"""Teams share sandboxes, datasets and their runs; only owners delete."""

import pytest

pytestmark = pytest.mark.integration

SPEC = {
    "task": {"type": "classification.single_label"},
    "nodes": [
        {"id": "data", "type": "dataset", "params": {"format": "synthetic"}},
        {
            "id": "model",
            "type": "model",
            "params": {
                "architecture": {
                    "kind": "pretrained",
                    "base": {"source": "torchvision", "name": "resnet18", "weights": None},
                }
            },
        },
        {"id": "train", "type": "trainer", "params": {"epochs": 1}},
    ],
    "edges": [["data", "model"], ["model", "train"]],
}


async def test_team_members_share_sandboxes_datasets_and_runs(client, register):
    alice, _ = await register()
    bob, bob_user = await register()
    carol, _ = await register()

    team = (await client.post("/v1/teams", json={"name": "lab"}, headers=alice)).json()
    added = await client.post(
        f"/v1/teams/{team['id']}/members", json={"email": bob_user["email"]}, headers=alice
    )
    not_owner = await client.post(
        f"/v1/teams/{team['id']}/members", json={"email": bob_user["email"]}, headers=bob
    )
    sandbox = (
        await client.post(
            "/v1/sandboxes", json={"name": "shared", "team_id": team["id"]}, headers=alice
        )
    ).json()
    dataset = (
        await client.post(
            "/v1/datasets", json={"name": "shared-data", "team_id": team["id"]}, headers=alice
        )
    ).json()
    pipeline = (
        await client.post(
            f"/v1/sandboxes/{sandbox['id']}/pipelines",
            json={"name": "p", "spec": SPEC},
            headers=bob,
        )
    ).json()
    version = (await client.get(f"/v1/pipelines/{pipeline['id']}/versions/1", headers=bob)).json()
    run = (
        await client.post(
            f"/v1/sandboxes/{sandbox['id']}/runs",
            json={"pipeline_version_id": version["id"], "hardware_tier": "gpu"},
            headers=bob,
        )
    ).json()

    bob_sandboxes = (await client.get("/v1/sandboxes", headers=bob)).json()["data"]
    bob_datasets = (await client.get("/v1/datasets", headers=bob)).json()["data"]
    carol_sandbox = await client.get(f"/v1/sandboxes/{sandbox['id']}", headers=carol)
    alice_sees_run = await client.get(f"/v1/runs/{run['id']}", headers=alice)
    carol_run = await client.get(f"/v1/runs/{run['id']}", headers=carol)
    bob_delete = await client.delete(f"/v1/sandboxes/{sandbox['id']}", headers=bob)
    usage = await client.get(f"/v1/teams/{team['id']}/usage", headers=bob)

    assert added.status_code == 201 and len(added.json()["members"]) == 2
    assert not_owner.status_code == 403
    assert [s["id"] for s in bob_sandboxes] == [sandbox["id"]]
    assert [d["id"] for d in bob_datasets] == [dataset["id"]]
    assert carol_sandbox.status_code == 404 and carol_run.status_code == 404
    assert alice_sees_run.status_code == 200
    assert bob_delete.status_code == 404  # members use, owners delete
    assert usage.status_code == 200 and usage.json()["total_units"] == 0

    await client.post(f"/v1/runs/{run['id']}/cancel", headers=bob)
    removed = await client.delete(f"/v1/teams/{team['id']}/members/{bob_user['id']}", headers=alice)
    after = await client.get(f"/v1/sandboxes/{sandbox['id']}", headers=bob)
    assert removed.status_code == 204 and after.status_code == 404
