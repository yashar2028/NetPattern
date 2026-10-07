import io
import json

from netpattern_engine.session import handle, serve

TASK = {"type": "classification.single_label", "classes": ["a", "b", "c"]}


def _call(method, params=None):
    return handle({"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}})


def test_validate_graph_returns_shapes_params_and_flops(cnn):
    reply = _call("validate_architecture", {"architecture": cnn(), "task": TASK})["result"]

    assert reply["ok"] and reply["output_shape"] == [3]
    assert reply["nodes"]["gap"]["shape"] == [16]
    assert reply["params"] > 0 and reply["flops"] > 0


def test_validate_graph_reports_node_issues(cnn):
    raw = cnn()
    raw["nodes"][-1]["params"] = {"out_features": 7}

    reply = _call("validate_architecture", {"architecture": raw, "task": TASK})["result"]

    assert not reply["ok"]
    assert reply["issues"][0]["node_id"] == "fc"


def test_validate_pretrained_without_downloading_weights():
    architecture = {
        "kind": "pretrained",
        "base": {"source": "torchvision", "name": "resnet50", "weights": "DEFAULT"},
    }

    reply = _call("validate_architecture", {"architecture": architecture, "task": TASK})["result"]

    assert reply["ok"] and reply["output_shape"] == [3]
    assert reply["head"] == ["fc"] and reply["input"]["size"] == [224, 224]


def test_inspect_model_groups_modules():
    architecture = {
        "kind": "pretrained",
        "base": {"source": "torchvision", "name": "resnet18", "weights": None},
    }

    reply = _call("inspect_model", {"architecture": architecture, "task": TASK})["result"]

    assert reply["traceable"]
    assert any(node.get("group") == "layer4.1" for node in reply["nodes"])


def test_zoo_lists_models_for_a_task():
    reply = _call("list_zoo", {"task": "segmentation.semantic", "source": "torchvision"})["result"]

    names = {entry["name"] for entry in reply}
    assert "deeplabv3_resnet50" in names and "resnet50" not in names


def test_errors_follow_json_rpc():
    assert _call("nope")["error"]["code"] == -32601
    reply = _call("validate_architecture", {"architecture": {"kind": "graph"}, "task": TASK})
    assert reply["error"]["code"] == 4001


def test_serve_loop_answers_until_shutdown():
    stdin = io.StringIO(
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"})
        + "\n"
        + json.dumps({"jsonrpc": "2.0", "id": 2, "method": "shutdown"})
        + "\n"
    )
    stdout = io.StringIO()

    assert serve(stdin, stdout) == 0
    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert lines[0]["method"] == "ready"
    assert lines[1]["result"]["pong"] is True
    assert lines[2]["result"] == {"bye": True}
