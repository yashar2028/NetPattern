from pathlib import Path

import pytest

from app.models.job import Job
from app.sandbox.environments import EngineSnapshot, environment_id, render_flake
from app.sandbox.isolation import SandboxPolicy, sandbox_env, wrap
from app.services import storage
from app.workers.handlers import needs_weights, prepare_request


def test_environment_id_depends_on_template_packs_and_snapshot():
    base = environment_id("pytorch-cpu", ["medical"], "abc", "x86_64-linux")

    assert base == environment_id("pytorch-cpu", ["medical"], "abc", "x86_64-linux")
    assert base == environment_id("pytorch-cpu", ["medical", "medical"][:1], "abc", "x86_64-linux")
    assert base != environment_id("pytorch-cpu", ["medical", "detection"], "abc", "x86_64-linux")
    assert base != environment_id("pytorch-cpu", ["medical"], "abd", "x86_64-linux")
    assert base != environment_id("pytorch-cuda12", ["medical"], "abc", "x86_64-linux")


def test_generated_flake_pins_the_engine_snapshot():
    snapshot = EngineSnapshot("f" * 64, "0.2.0", Path("/data/engine-snapshots/0.2.0-ffffffffffff"))

    flake = render_flake("pytorch-cpu", ["medical", "detection"], snapshot, "x86_64-linux", "env_1")

    assert 'inputs.engine.url = "path:/data/engine-snapshots/0.2.0-ffffffffffff";' in flake
    assert "packages.x86_64-linux.default = engine.lib.mkEnvironment" in flake
    assert 'packs = [ "detection" "medical" ];' in flake


def test_no_isolation_runs_the_command_as_is():
    assert wrap(["netpattern-runtime", "job"], SandboxPolicy(), mode="none") == [
        "netpattern-runtime",
        "job",
    ]


def test_bubblewrap_isolates_and_binds_only_what_is_allowed(tmp_path):
    datasets, run_dir = tmp_path / "datasets", tmp_path / "run"
    datasets.mkdir()

    command = wrap(
        ["netpattern-runtime", "job"],
        SandboxPolicy(read_only=[datasets], read_write=[run_dir], working_dir=run_dir),
        mode="bwrap",
    )

    assert command[0].endswith("bwrap") and "--unshare-all" in command
    assert "--share-net" not in command
    assert ["--ro-bind", str(datasets), str(datasets)] == command[
        command.index(str(datasets)) - 1 : command.index(str(datasets)) + 2
    ]
    assert ["--bind", str(run_dir), str(run_dir)] == command[
        command.index("--bind") : command.index("--bind") + 3
    ]
    assert command[-3:] == ["--", "netpattern-runtime", "job"]


def test_bubblewrap_shares_network_only_when_allowed(tmp_path):
    command = wrap(["x"], SandboxPolicy(network=True), mode="bwrap")

    assert "--share-net" in command


def test_sandbox_environment_inherits_nothing(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql://secret")

    variables = sandbox_env(Path("/nix/store/abc-env"), tmp_path, network=False)

    assert "DATABASE_URL" not in variables
    assert variables["PATH"] == "/nix/store/abc-env/bin"
    assert variables["HF_HUB_OFFLINE"] == "1"
    assert "HF_HUB_OFFLINE" not in sandbox_env(Path("/env"), tmp_path, network=True)


def test_dataset_paths_cannot_escape_the_datasets_folder():
    assert (
        storage.resolve_dataset_path("oxford-pets")
        == (storage.datasets_root() / "oxford-pets").resolve()
    )
    with pytest.raises(ValueError):
        storage.resolve_dataset_path("../runs/secret")
    with pytest.raises(ValueError):
        storage.resolve_dataset_path("/etc")


def _job(payload: dict, kind: str = "train") -> Job:
    return Job(id="job_test", kind=kind, payload=payload, environment={})


def test_prepare_request_resolves_dataset_roots():
    job = _job(
        {
            "pipeline": {
                "nodes": [
                    {"id": "d", "type": "dataset", "params": {"format": "csv", "root": "pets"}}
                ]
            }
        }
    )

    request, readable, writable = prepare_request(job)

    root = request["pipeline"]["nodes"][0]["params"]["root"]
    assert root == str((storage.datasets_root() / "pets").resolve())
    assert readable == [(storage.datasets_root() / "pets").resolve()]  # only this dataset
    assert request["run_id"] == "job_test" and request["kind"] == "train"
    assert writable == []
    assert (
        job.payload["pipeline"]["nodes"][0]["params"]["root"] == "pets"
    )  # the stored payload is untouched


def test_prepare_request_rejects_escaping_dataset_roots():
    job = _job({"dataset": {"format": "csv", "root": "../../etc"}}, kind="profile")

    with pytest.raises(ValueError):
        prepare_request(job)


@pytest.mark.parametrize(
    ("architecture", "expected"),
    [
        (
            {
                "kind": "pretrained",
                "base": {"source": "torchvision", "name": "resnet18", "weights": "DEFAULT"},
            },
            True,
        ),
        (
            {
                "kind": "pretrained",
                "base": {"source": "torchvision", "name": "resnet18", "weights": None},
            },
            False,
        ),
        (
            {
                "kind": "graph",
                "nodes": [{"id": "b", "op": "Backbone", "params": {"name": "resnet18"}}],
            },
            True,
        ),
        (
            {
                "kind": "graph",
                "nodes": [
                    {
                        "id": "b",
                        "op": "Backbone",
                        "params": {"name": "resnet18", "pretrained": False},
                    }
                ],
            },
            False,
        ),
        ({"kind": "graph", "nodes": [{"id": "c", "op": "Conv2d"}]}, False),
    ],
)
def test_needs_weights(architecture, expected):
    request = {
        "pipeline": {
            "nodes": [{"id": "m", "type": "model", "params": {"architecture": architecture}}]
        }
    }

    assert needs_weights(request) is expected
