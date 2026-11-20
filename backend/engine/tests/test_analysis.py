"""Error gallery, Grad-CAM explanations and the learning-rate finder."""

import json

import pytest

from netpattern_engine.inference import PARITY_FILE, Predictor
from netpattern_engine.runtime import main

pytest.importorskip("pycocotools")

TRAINING = {"epochs": 1, "batch_size": 8, "num_workers": 0, "log_every_seconds": 0}
DATA = {"format": "synthetic", "num_samples": 40, "image_size": 32}


def _job(tmp_path, capsys, kind, pipeline, options=None):
    request = {
        "run_id": f"run_{kind}",
        "kind": kind,
        "pipeline": pipeline,
        "options": options or {},
    }
    request_file, result_file = tmp_path / "request.json", tmp_path / "result.json"
    request_file.write_text(json.dumps(request))
    exit_code = main(["job", "--request", str(request_file), "--output", str(result_file)])
    capsys.readouterr()
    result = json.loads(result_file.read_text())
    assert exit_code == 0, result
    return result


def test_training_saves_an_error_gallery_and_predictions_can_be_explained(
    tmp_path, capsys, make_pipeline, cnn
):
    pipeline = make_pipeline({"type": "classification.single_label"}, DATA, cnn(), TRAINING)
    result = _job(tmp_path, capsys, "train", pipeline)

    errors = result["errors"]
    assert errors["split"] == "test" and errors["evaluated"] > 0
    for sample in errors["samples"]:
        assert (tmp_path / sample["image"]).is_file() and sample["true"] != sample["predicted"]

    predictor = Predictor.load(tmp_path)
    sample = tmp_path / json.loads((tmp_path / PARITY_FILE).read_text())[0]["file"]
    explanation = predictor.predict([sample], {"explain": True})["outputs"][0]["explanation"]
    assert explanation["method"] == "grad-cam" and explanation["heatmap_png_base64"]
    assert explanation["class"] in predictor.task.classes


def test_lr_finder_returns_a_curve_and_a_suggestion(tmp_path, capsys, make_pipeline, cnn):
    pipeline = make_pipeline({"type": "classification.single_label"}, DATA, cnn(), TRAINING)
    result = _job(tmp_path, capsys, "lr_find", pipeline, {"steps": 30})

    assert len(result["lrs"]) == len(result["losses"]) > 5
    assert result["lrs"][0] < result["lrs"][-1]
    assert result["suggested_lr"] is None or 1e-7 <= result["suggested_lr"] <= 1.0
