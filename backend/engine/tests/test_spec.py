import json

import pytest

from netpattern_engine.errors import SpecError
from netpattern_engine.schema import schema_document
from netpattern_engine.spec import resolve_pipeline

SYNTHETIC = {"format": "synthetic", "num_samples": 16, "image_size": 32}
CLASSIFICATION = {"type": "classification.single_label"}


def test_resolves_a_canvas_into_typed_parts(make_pipeline, cnn):
    resolved = resolve_pipeline(make_pipeline(CLASSIFICATION, SYNTHETIC, cnn()))

    assert resolved.task.type == "classification.single_label"
    assert resolved.data.format == "synthetic"
    assert resolved.architecture.kind == "graph"
    assert resolved.training.epochs == 2
    assert resolved.evaluation.splits == ["val", "test"]


def test_reports_issues_with_node_id_and_field(make_pipeline, cnn):
    spec = make_pipeline(CLASSIFICATION, SYNTHETIC, cnn(), training={"optimizer": {"lr": -1}})

    with pytest.raises(SpecError) as raised:
        resolve_pipeline(spec)

    issue = raised.value.issues[0]
    assert issue.node_id == "train"
    assert issue.field == "params.optimizer.lr"


def test_missing_node_and_wrong_edges_are_reported(make_pipeline, cnn):
    spec = make_pipeline(CLASSIFICATION, SYNTHETIC, cnn())
    spec["nodes"] = [n for n in spec["nodes"] if n["type"] != "trainer"]
    spec["edges"] = [["data", "model"]]

    with pytest.raises(SpecError) as raised:
        resolve_pipeline(spec)

    messages = " ".join(issue.message for issue in raised.value.issues)
    assert "needs a 'trainer' node" in messages
    assert "cannot connect" in messages


def test_dataset_format_must_support_the_task(make_pipeline, cnn):
    spec = make_pipeline(
        {"type": "detection.bbox"}, {"format": "image_folder", "root": "/tmp"}, cnn()
    )

    with pytest.raises(SpecError) as raised:
        resolve_pipeline(spec)

    assert any(issue.field == "params.format" for issue in raised.value.issues)


def test_split_fractions_must_add_up(make_pipeline, cnn):
    data = {**SYNTHETIC, "split": {"train": 0.5, "val": 0.1, "test": 0.1}}

    with pytest.raises(SpecError):
        resolve_pipeline(make_pipeline(CLASSIFICATION, data, cnn()))


def test_patch_with_alias_is_accepted(make_pipeline):
    architecture = {
        "kind": "pretrained",
        "base": {"source": "torchvision", "name": "resnet18", "weights": None},
        "patches": [
            {
                "op": "replace",
                "target": "fc",
                "with": [{"op": "Linear", "params": {"out_features": 3}}],
            }
        ],
    }

    resolved = resolve_pipeline(make_pipeline(CLASSIFICATION, SYNTHETIC, architecture))

    assert resolved.architecture.patches[0].with_[0].op == "Linear"


def test_schema_document_lists_capabilities():
    document = schema_document()

    assert {"pipeline", "dataset", "architecture", "training"} <= set(document["schemas"])
    ops = {op["op"] for op in document["capabilities"]["graph_ops"]}
    assert {"Conv2d", "Add", "Concat", "BasicBlock", "Backbone"} <= ops
    assert len(document["capabilities"]["tasks"]) == 5
    json.dumps(document)
