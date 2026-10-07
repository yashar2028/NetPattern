import pytest


def graph_cnn(num_classes_symbol: str = "$num_outputs", channels: int = 3, size: int = 32) -> dict:
    """A small residual CNN for classification-style tasks."""
    return {
        "kind": "graph",
        "input": {"shape": [channels, size, size]},
        "nodes": [
            {"id": "c1", "op": "Conv2d", "params": {"out_channels": 16}},
            {"id": "bn1", "op": "BatchNorm2d"},
            {"id": "a1", "op": "ReLU"},
            {"id": "c2", "op": "Conv2d", "params": {"out_channels": 16}},
            {"id": "add", "op": "Add"},
            {"id": "pool", "op": "MaxPool2d"},
            {"id": "gap", "op": "GlobalAvgPool"},
            {"id": "fc", "op": "Linear", "params": {"out_features": num_classes_symbol}},
        ],
        "edges": [
            ["input", "c1"],
            ["c1", "bn1"],
            ["bn1", "a1"],
            ["a1", "c2"],
            ["c2", "add"],
            ["a1", "add"],
            ["add", "pool"],
            ["pool", "gap"],
            ["gap", "fc"],
        ],
    }


def segmentation_graph(size: int = 32) -> dict:
    return {
        "kind": "graph",
        "input": {"shape": [3, size, size]},
        "nodes": [
            {"id": "c1", "op": "Conv2d", "params": {"out_channels": 16}},
            {"id": "a1", "op": "ReLU"},
            {"id": "c2", "op": "Conv2d", "params": {"out_channels": 16}},
            {"id": "a2", "op": "ReLU"},
            {
                "id": "out",
                "op": "Conv2d",
                "params": {"out_channels": "$num_classes", "kernel_size": 1},
            },
        ],
        "edges": [["input", "c1"], ["c1", "a1"], ["a1", "c2"], ["c2", "a2"], ["a2", "out"]],
    }


def pipeline(
    task: dict,
    data: dict,
    architecture: dict,
    training: dict | None = None,
    transforms: dict | None = None,
) -> dict:
    nodes = [
        {"id": "data", "type": "dataset", "params": data},
        {"id": "tf", "type": "transforms", "params": transforms or {"preset": "basic"}},
        {"id": "model", "type": "model", "params": {"architecture": architecture}},
        {
            "id": "train",
            "type": "trainer",
            "params": training
            or {"epochs": 2, "batch_size": 8, "num_workers": 0, "log_every_seconds": 0},
        },
    ]
    return {
        "task": task,
        "nodes": nodes,
        "edges": [["data", "tf"], ["tf", "model"], ["model", "train"]],
    }


@pytest.fixture
def make_pipeline():
    return pipeline


@pytest.fixture
def cnn():
    return graph_cnn


@pytest.fixture
def seg_graph():
    return segmentation_graph
