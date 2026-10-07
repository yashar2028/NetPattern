import torch

from netpattern_engine.graph.compiler import analyze_graph, compile_graph
from netpattern_engine.spec.architecture import GraphArchitecture

SYMBOLS = {"num_outputs": 5, "num_classes": 5}


def _arch(raw: dict) -> GraphArchitecture:
    return GraphArchitecture.model_validate(raw)


def test_shapes_params_and_flops_are_inferred(cnn):
    result = analyze_graph(_arch(cnn()), SYMBOLS)

    assert result.issues == []
    assert result.shapes["c1"] == (16, 32, 32)
    assert result.shapes["pool"] == (16, 16, 16)
    assert result.shapes["gap"] == (16,)
    assert result.output == "fc" and result.output_shape == (5,)
    assert result.params["c1"] == 3 * 16 * 9 + 16
    assert result.params["fc"] == 16 * 5 + 5
    assert result.flops and result.flops > 0


def test_compiled_graph_runs_on_real_tensors(cnn):
    result = compile_graph(_arch(cnn()), SYMBOLS)

    out = result.module(torch.randn(4, 3, 32, 32))
    assert out.shape == (4, 5)
    assert "add" in result.module.code


def test_channel_mismatch_is_reported_at_the_merge_node(cnn):
    raw = cnn()
    raw["nodes"][3]["params"] = {"out_channels": 32}  # c2 now outputs 32 channels

    result = analyze_graph(_arch(raw), SYMBOLS)

    assert [issue.node_id for issue in result.issues] == ["add"]
    assert "shape mismatch at Add: 32×32×32 vs 16×32×32" in result.issues[0].message


def test_linear_on_a_feature_map_suggests_flatten():
    raw = {
        "kind": "graph",
        "input": {"shape": [3, 8, 8]},
        "nodes": [{"id": "fc", "op": "Linear", "params": {"out_features": 2}}],
        "edges": [["input", "fc"]],
    }

    result = analyze_graph(_arch(raw), SYMBOLS)

    assert result.issues[0].node_id == "fc"
    assert "Flatten" in result.issues[0].message


def test_structural_problems_are_reported():
    raw = {
        "kind": "graph",
        "input": {"shape": [3, 8, 8]},
        "nodes": [
            {"id": "a", "op": "ReLU"},
            {"id": "b", "op": "ReLU"},
            {"id": "c", "op": "Nope"},
            {"id": "d", "op": "Add"},
        ],
        "edges": [["input", "a"], ["b", "c"], ["c", "b"], ["a", "d"]],
    }

    result = analyze_graph(_arch(raw), SYMBOLS)
    by_node = {(issue.node_id, issue.field) for issue in result.issues}

    assert ("c", "op") in by_node
    assert any(node == "d" for node, _ in by_node)  # Add with a single input
    assert any(node == "b" for node, _ in by_node)  # cycle / unreachable


def test_invalid_params_point_at_the_field():
    raw = {
        "kind": "graph",
        "input": {"shape": [3, 8, 8]},
        "nodes": [{"id": "c", "op": "Conv2d", "params": {"out_channels": 0}}],
        "edges": [["input", "c"]],
    }

    result = analyze_graph(_arch(raw), SYMBOLS)

    assert result.issues[0].node_id == "c"
    assert result.issues[0].field == "params.out_channels"


def test_composite_blocks_and_concat_build():
    raw = {
        "kind": "graph",
        "input": {"shape": [3, 32, 32]},
        "nodes": [
            {"id": "stem", "op": "Conv2d", "params": {"out_channels": 16, "stride": 2}},
            {"id": "basic", "op": "BasicBlock", "params": {"out_channels": 32, "stride": 2}},
            {"id": "bottle", "op": "Bottleneck", "params": {"out_channels": 32}},
            {"id": "dense", "op": "DenseBlock", "params": {"num_layers": 2, "growth_rate": 8}},
            {"id": "se", "op": "SqueezeExcite"},
            {
                "id": "incep",
                "op": "InceptionBlock",
                "params": {"c1x1": 8, "c3x3": 8, "c5x5": 8, "pool_proj": 8},
            },
            {"id": "mb", "op": "MBConv", "params": {"out_channels": 24}},
            {"id": "cat", "op": "Concat"},
            {"id": "gap", "op": "GlobalAvgPool"},
            {"id": "fc", "op": "Linear", "params": {"out_features": "$num_outputs"}},
        ],
        "edges": [
            ["input", "stem"],
            ["stem", "basic"],
            ["basic", "bottle"],
            ["bottle", "dense"],
            ["dense", "se"],
            ["se", "incep"],
            ["incep", "mb"],
            ["mb", "cat"],
            ["se", "cat"],
            ["cat", "gap"],
            ["gap", "fc"],
        ],
    }

    result = compile_graph(_arch(raw), SYMBOLS)

    assert result.shapes["dense"] == (48, 8, 8)
    assert result.shapes["incep"] == (32, 8, 8)
    assert result.shapes["cat"] == (24 + 48, 8, 8)
    assert result.module(torch.randn(2, 3, 32, 32)).shape == (2, 5)
