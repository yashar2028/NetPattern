"""Code block layers and model export."""

import json

import pytest
import torch

from netpattern_engine.export import export_model
from netpattern_engine.graph.ops import ALLOW_CODE_ENV
from netpattern_engine.inference import Predictor
from netpattern_engine.runtime import main
from netpattern_engine.session import validate_architecture

pytest.importorskip("pycocotools")

TASK = {"type": "classification.single_label"}
GOOD = """def build(in_shape):
    return nn.Sequential(nn.Conv2d(in_shape[0], 8, 3, padding=1), nn.ReLU())
"""


def _graph(code):
    return {
        "kind": "graph",
        "input": {"shape": [3, 32, 32]},
        "nodes": [
            {"id": "mine", "op": "Custom", "params": {"code": code}},
            {"id": "pool", "op": "GlobalAvgPool", "params": {}},
            {"id": "fc", "op": "Linear", "params": {"out_features": "$num_outputs"}},
        ],
        "edges": [["input", "mine"], ["mine", "pool"], ["pool", "fc"]],
    }


def _issues(code):
    result = validate_architecture({"architecture": _graph(code), "task": TASK, "num_outputs": 3})
    return result, [
        issue["message"] for issue in result["issues"] if issue.get("node_id") == "mine"
    ]


def test_a_code_block_is_shape_checked_like_any_layer(monkeypatch):
    monkeypatch.setenv(ALLOW_CODE_ENV, "1")
    result, issues = _issues(GOOD)
    assert result["ok"] and not issues
    assert result["nodes"]["mine"]["shape"] == [8, 32, 32] and result["output_shape"] == [3]


@pytest.mark.parametrize(
    ("code", "message"),
    [
        ("def build(in_shape)\n    pass", "syntax error on line 1"),
        ("x = 1", "define a function build"),
        ("def build(in_shape):\n    return 3", "must return an nn.Module"),
        ("def build(in_shape):\n    return nn.Linear(5, 5)", "code block failed"),
    ],
)
def test_code_block_mistakes_become_node_issues(monkeypatch, code, message):
    monkeypatch.setenv(ALLOW_CODE_ENV, "1")
    _, issues = _issues(code)
    assert issues and message in issues[0], issues


def test_code_blocks_can_be_turned_off(monkeypatch):
    monkeypatch.delenv(ALLOW_CODE_ENV, raising=False)
    _, issues = _issues(GOOD)
    assert issues == ["code blocks are turned off on this platform"]


def test_export_writes_weights_and_a_matching_torchscript_model(
    tmp_path, capsys, make_pipeline, cnn
):
    data = {"format": "synthetic", "num_samples": 24, "image_size": 32}
    training = {"epochs": 1, "batch_size": 8, "num_workers": 0, "log_every_seconds": 0}
    request = {
        "run_id": "run_export",
        "kind": "train",
        "pipeline": make_pipeline(TASK, data, cnn(), training),
    }
    (tmp_path / "request.json").write_text(json.dumps(request))
    assert (
        main(
            [
                "job",
                "--request",
                str(tmp_path / "request.json"),
                "--output",
                str(tmp_path / "result.json"),
            ]
        )
        == 0
    )
    capsys.readouterr()

    out = tmp_path / "export-run"
    result = export_model(
        tmp_path, out, ["state_dict", "torchscript", "onnx"], lambda *a, **k: None
    )

    assert {"state_dict", "torchscript", "model.json"} <= set(result["files"])
    predictor = Predictor.load(tmp_path)
    example = torch.rand(1, *result["input"]["shape"])
    scripted = torch.jit.load(str(out / result["files"]["torchscript"]))
    with torch.no_grad():
        assert torch.allclose(scripted(example), predictor.module(example), atol=1e-5)
    assert "onnx" in result["files"] or "onnx" in result["notes"]
