"""Predictions from a trained run directory: rebuild, parity check, task-specific outputs."""

import json

import pytest

from netpattern_engine.inference import MODEL_FILE, PARITY_FILE, Predictor, check_parity
from netpattern_engine.runtime import main

pytest.importorskip("pycocotools")

TRAINING = {"epochs": 1, "batch_size": 8, "num_workers": 0, "log_every_seconds": 0}


def _train(tmp_path, capsys, pipeline):
    request = {"run_id": "run_inference", "kind": "train", "pipeline": pipeline}
    request_file, result_file = tmp_path / "request.json", tmp_path / "result.json"
    request_file.write_text(json.dumps(request))
    exit_code = main(["job", "--request", str(request_file), "--output", str(result_file)])
    capsys.readouterr()
    assert exit_code == 0, result_file.read_text()
    return tmp_path


@pytest.mark.parametrize("task", ["classification.single_label", "regression"])
def test_a_trained_run_predicts_and_passes_parity(tmp_path, capsys, make_pipeline, cnn, task):
    data = {"format": "synthetic", "num_samples": 32, "image_size": 32}
    run_dir = _train(tmp_path, capsys, make_pipeline({"type": task}, data, cnn(), TRAINING))
    assert (run_dir / MODEL_FILE).is_file() and (run_dir / PARITY_FILE).is_file()

    predictor = Predictor.load(run_dir)
    parity = check_parity(predictor, run_dir)
    sample = run_dir / json.loads((run_dir / PARITY_FILE).read_text())[0]["file"]
    output = predictor.predict([sample], {"top_k": 2})["outputs"][0]

    assert parity["passed"] and parity["samples"] == 4, parity
    if task == "regression":
        assert set(output["values"]) == set(predictor.task.targets)
    else:
        assert output["label"] in predictor.task.classes and len(output["top_k"]) == 2


def test_segmentation_returns_a_mask_of_the_input_size(tmp_path, capsys, make_pipeline, seg_graph):
    data = {"format": "synthetic", "num_samples": 16, "image_size": 32}
    pipeline = make_pipeline({"type": "segmentation.semantic"}, data, seg_graph(), TRAINING)
    run_dir = _train(tmp_path, capsys, pipeline)

    predictor = Predictor.load(run_dir)
    sample = run_dir / json.loads((run_dir / PARITY_FILE).read_text())[0]["file"]
    output = predictor.predict([sample])["outputs"][0]

    assert check_parity(predictor, run_dir)["passed"]
    assert abs(sum(output["pixel_fractions"].values()) - 1.0) < 1e-3
    assert output["mask_png_base64"]


def test_a_torchvision_model_is_rebuilt_from_its_checkpoint(tmp_path, capsys, make_pipeline):
    data = {"format": "synthetic", "num_samples": 24, "image_size": 32}
    arch = {
        "kind": "pretrained",
        "base": {"source": "torchvision", "name": "resnet18", "weights": None},
    }
    pipeline = make_pipeline({"type": "classification.single_label"}, data, arch, TRAINING)
    run_dir = _train(tmp_path, capsys, pipeline)

    assert check_parity(Predictor.load(run_dir), run_dir)["passed"]
