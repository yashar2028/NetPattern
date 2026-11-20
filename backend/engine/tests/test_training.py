"""Training on synthetic data: an "overfit one batch" sanity check per task, plus a full job."""

import json
from pathlib import Path

import pytest

from netpattern_engine.pipeline import sanity
from netpattern_engine.runtime import main
from netpattern_engine.spec import resolve_pipeline

pytest.importorskip("pycocotools")

TRAINING = {
    "epochs": 2,
    "batch_size": 8,
    "num_workers": 0,
    "log_every_seconds": 0,
    "optimizer": {"lr": 0.01},
}


def _noop(*args, **kwargs):
    pass


def _synthetic(**extra):
    return {"format": "synthetic", "num_samples": 32, "image_size": 32, **extra}


@pytest.mark.parametrize(
    "task",
    ["classification.single_label", "classification.multi_label", "regression"],
)
def test_sanity_overfits_one_batch_with_a_custom_cnn(task, make_pipeline, cnn):
    spec = resolve_pipeline(make_pipeline({"type": task}, _synthetic(), cnn(), TRAINING))

    result = sanity(spec, _noop, steps=60)

    assert result["passed"], result


def test_sanity_overfits_one_batch_for_segmentation(make_pipeline, seg_graph):
    spec = resolve_pipeline(
        make_pipeline({"type": "segmentation.semantic"}, _synthetic(), seg_graph(), TRAINING)
    )

    result = sanity(spec, _noop, steps=80)

    assert result["passed"], result


def test_sanity_uses_the_configured_optimizer(monkeypatch, make_pipeline, cnn):
    from netpattern_engine import pipeline
    from netpattern_engine.training.optim import build_optimizer

    calls = []

    def record_optimizer(model, optimizer_spec, head):
        calls.append(optimizer_spec)
        return build_optimizer(model, optimizer_spec, head)

    monkeypatch.setattr(pipeline, "build_optimizer", record_optimizer)
    training = {
        **TRAINING,
        "optimizer": {"name": "sgd", "lr": 0.001, "momentum": 0.9},
    }
    spec = resolve_pipeline(
        make_pipeline({"type": "classification.single_label"}, _synthetic(), cnn(), training)
    )

    sanity(spec, _noop, steps=2)

    assert len(calls) == 1
    assert calls[0].name == "sgd"
    assert calls[0].lr == 0.001
    assert calls[0].momentum == 0.9


def test_sanity_overfits_one_batch_for_detection(make_pipeline):
    architecture = {
        "kind": "pretrained",
        "base": {
            "source": "torchvision",
            "name": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "weights": None,
            "options": {"min_size": 96, "max_size": 96},
        },
    }
    # A randomly initialized detector needs smaller updates than the toy classifiers.
    # Keep Adam explicit so this exercises the original optimizer with a safer step size.
    training = {**TRAINING, "batch_size": 2, "optimizer": {"name": "adam", "lr": 0.0001}}
    spec = resolve_pipeline(
        make_pipeline(
            {"type": "detection.bbox"},
            _synthetic(num_classes=2, image_size=64),
            architecture,
            training,
        )
    )

    result = sanity(spec, _noop, steps=100)

    assert result["passed"], result


def test_train_job_streams_events_writes_checkpoints_and_metrics(
    tmp_path, capsys, make_pipeline, cnn
):
    request = {
        "run_id": "run_train",
        "kind": "train",
        "pipeline": make_pipeline(
            {"type": "classification.single_label"}, _synthetic(num_samples=48), cnn(), TRAINING
        ),
    }
    request_file, result_file = tmp_path / "request.json", tmp_path / "result.json"
    request_file.write_text(json.dumps(request))

    exit_code = main(["job", "--request", str(request_file), "--output", str(result_file)])

    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    result = json.loads(result_file.read_text())
    types = [event["type"] for event in events]
    assert exit_code == 0, result
    assert types[0] == "run_started" and types[-1] == "run_completed"
    assert types.count("epoch_end") == 2
    assert {"dataset_indexed", "model_built", "batch", "evaluation"} <= set(types)
    assert result["status"] == "completed"
    assert result["training"]["epochs_run"] == 2
    assert set(result["evaluation"]) == {"val", "test"}
    assert "f1_macro" in result["evaluation"]["val"]["metrics"]
    assert Path(tmp_path / result["checkpoints"]["best"]).is_file()
    epoch = next(e for e in events if e["type"] == "epoch_end")
    assert "val/f1_macro" in epoch["metrics"] and "train/loss" in epoch["metrics"]


class _Crash(Exception):
    pass


def test_interrupted_training_resumes_where_it_stopped(tmp_path, make_pipeline, cnn):
    from netpattern_engine.pipeline import train
    from netpattern_engine.training.trainer import CancelToken

    training = {**TRAINING, "epochs": 3, "deterministic": True, "scheduler": {"name": "cosine"}}
    raw = make_pipeline(
        {"type": "classification.single_label"}, _synthetic(num_samples=48), cnn(), training
    )
    reference = train(resolve_pipeline(raw), tmp_path / "reference", _noop, CancelToken())

    def crash_after_first_epoch(kind, **payload):
        if kind == "epoch_end" and payload["epoch"] == 1:
            raise _Crash  # the worker died right after epoch 1 was saved

    run_dir = tmp_path / "run"
    with pytest.raises(_Crash):
        train(resolve_pipeline(raw), run_dir, crash_after_first_epoch, CancelToken())
    events = []
    resumed = train(
        resolve_pipeline(raw),
        run_dir,
        lambda kind, **payload: events.append((kind, payload)),
        CancelToken(),
        resume=True,
    )

    assert ("resumed", {"from_epoch": 2, "epochs_done": 1}) in events
    assert [p["epoch"] for kind, p in events if kind == "epoch_end"] == [2, 3]
    assert resumed["training"]["epochs_run"] == 3
    expected = [h["metrics"]["train/loss"] for h in reference["training"]["history"]]
    actual = [h["metrics"]["train/loss"] for h in resumed["training"]["history"]]
    assert actual == pytest.approx(expected, abs=1e-5)


def test_resume_without_a_checkpoint_starts_from_scratch(tmp_path, make_pipeline, cnn):
    from netpattern_engine.pipeline import train
    from netpattern_engine.training.trainer import CancelToken

    raw = make_pipeline({"type": "classification.single_label"}, _synthetic(), cnn(), TRAINING)
    events = []

    result = train(
        resolve_pipeline(raw),
        tmp_path,
        lambda kind, **payload: events.append(kind),
        CancelToken(),
        resume=True,
    )

    assert "resumed" not in events
    assert result["training"]["epochs_run"] == 2


def test_invalid_pipeline_fails_with_issues(tmp_path, capsys, make_pipeline, cnn):
    raw = cnn()
    raw["nodes"][3]["params"] = {"out_channels": 8}
    request = {
        "run_id": "bad",
        "kind": "train",
        "pipeline": make_pipeline(
            {"type": "classification.single_label"}, _synthetic(), raw, TRAINING
        ),
    }
    request_file, result_file = tmp_path / "request.json", tmp_path / "result.json"
    request_file.write_text(json.dumps(request))

    exit_code = main(["job", "--request", str(request_file), "--output", str(result_file)])

    result = json.loads(result_file.read_text())
    assert exit_code == 1 and result["status"] == "failed"
    assert result["issues"][0]["node_id"] == "add"
