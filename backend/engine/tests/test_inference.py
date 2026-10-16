"""Predictions with a trained checkpoint: the "predict" job kind."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from netpattern_engine.inference import predict
from netpattern_engine.pipeline import train
from netpattern_engine.runtime import main
from netpattern_engine.spec import resolve_pipeline
from netpattern_engine.training.trainer import CancelToken

TRAINING = {
    "epochs": 1,
    "batch_size": 8,
    "num_workers": 0,
    "log_every_seconds": 0,
    "optimizer": {"lr": 0.01},
}


def _noop(*args, **kwargs):
    pass


def _synthetic(**extra):
    return {"format": "synthetic", "num_samples": 32, "image_size": 32, **extra}


def _images(folder: Path, sizes: list[tuple[int, int]]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i, (height, width) in enumerate(sizes):
        pixels = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(folder / f"img_{i}.png")
    return folder


def _trained(tmp_path: Path, raw: dict) -> tuple[object, Path]:
    pipeline = resolve_pipeline(raw)
    train(pipeline, tmp_path / "train", _noop, CancelToken())
    return pipeline, tmp_path / "train" / "checkpoints" / "best.pt"


def test_classification_predictions_rank_the_classes(tmp_path, make_pipeline, cnn):
    pipeline, checkpoint = _trained(
        tmp_path,
        make_pipeline({"type": "classification.single_label"}, _synthetic(), cnn(), TRAINING),
    )
    inputs = _images(tmp_path / "inputs", [(32, 32), (50, 40)])
    (inputs / "notes.txt").write_text("ignored")

    result = predict(pipeline, checkpoint, inputs, {}, tmp_path / "out", _noop, CancelToken())

    assert result["status"] == "completed" and result["samples"] == 2
    first = result["predictions"][0]
    assert first["file"] == "img_0.png"
    assert first["label"] in result["task"]["classes"]
    assert first["top"][0] == {"label": first["label"], "score": first["score"]}
    assert sum(entry["score"] for entry in first["top"]) == pytest.approx(1.0, abs=1e-4)


def test_unreadable_images_are_reported_not_fatal(tmp_path, make_pipeline, cnn):
    pipeline, checkpoint = _trained(
        tmp_path,
        make_pipeline({"type": "classification.single_label"}, _synthetic(), cnn(), TRAINING),
    )
    inputs = _images(tmp_path / "inputs", [(32, 32)])
    (inputs / "broken.png").write_bytes(b"not a png")

    result = predict(pipeline, checkpoint, inputs, {}, tmp_path / "out", _noop, CancelToken())

    by_file = {entry["file"]: entry for entry in result["predictions"]}
    assert "error" in by_file["broken.png"] and "label" in by_file["img_0.png"]
    assert result["warnings"]


def test_regression_predictions_use_the_training_scale(tmp_path, make_pipeline, cnn):
    pipeline, checkpoint = _trained(
        tmp_path, make_pipeline({"type": "regression"}, _synthetic(), cnn(), TRAINING)
    )
    inputs = _images(tmp_path / "inputs", [(32, 32)])

    result = predict(pipeline, checkpoint, inputs, {}, tmp_path / "out", _noop, CancelToken())

    assert set(result["predictions"][0]["values"]) == {"brightness"}
    assert result["task"]["target_std"]


def test_segmentation_masks_have_the_original_image_size(tmp_path, make_pipeline, seg_graph):
    pipeline, checkpoint = _trained(
        tmp_path,
        make_pipeline({"type": "segmentation.semantic"}, _synthetic(), seg_graph(), TRAINING),
    )
    inputs = _images(tmp_path / "inputs", [(40, 48)])
    out = tmp_path / "out"

    result = predict(pipeline, checkpoint, inputs, {}, out, _noop, CancelToken())

    entry = result["predictions"][0]
    mask = Image.open(out / entry["mask"])
    assert mask.size == (48, 40)
    assert np.asarray(mask).max() < len(result["task"]["classes"])
    assert sum(entry["class_fractions"].values()) == pytest.approx(1.0, abs=1e-4)


def test_predict_job_through_the_runtime(tmp_path, capsys, make_pipeline, cnn):
    raw = make_pipeline({"type": "classification.single_label"}, _synthetic(), cnn(), TRAINING)
    _, checkpoint = _trained(tmp_path, raw)
    inputs = _images(tmp_path / "inputs", [(32, 32)])
    run_dir = tmp_path / "predict"
    run_dir.mkdir()
    request = {
        "run_id": "run_predict",
        "kind": "predict",
        "pipeline": raw,
        "checkpoint": str(checkpoint),
        "inputs": {"source": "upload", "root": str(inputs)},
    }
    (run_dir / "request.json").write_text(json.dumps(request))
    capsys.readouterr()

    exit_code = main(
        [
            "job",
            "--request",
            str(run_dir / "request.json"),
            "--output",
            str(run_dir / "result.json"),
        ]
    )

    result = json.loads((run_dir / "result.json").read_text())
    types = [json.loads(line)["type"] for line in capsys.readouterr().out.splitlines()]
    assert exit_code == 0, result
    assert "predict_progress" in types and types[-1] == "run_completed"
    assert result["kind"] == "predict" and result["samples"] == 1


def test_a_checkpoint_for_another_task_is_rejected(tmp_path, make_pipeline, cnn):
    from netpattern_engine.errors import EngineError

    _, checkpoint = _trained(
        tmp_path, make_pipeline({"type": "regression"}, _synthetic(), cnn(), TRAINING)
    )
    other = resolve_pipeline(
        make_pipeline({"type": "classification.single_label"}, _synthetic(), cnn(), TRAINING)
    )
    inputs = _images(tmp_path / "inputs", [(32, 32)])

    with pytest.raises(EngineError, match="checkpoint is for regression"):
        predict(other, checkpoint, inputs, {}, tmp_path / "out", _noop, CancelToken())
