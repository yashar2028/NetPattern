import pytest
import torch
import torchvision

from netpattern_engine.errors import SpecError
from netpattern_engine.graph.inspect import inspect_model
from netpattern_engine.models.build import build_model
from netpattern_engine.spec.architecture import PretrainedArchitecture
from netpattern_engine.spec.task import TaskSpec
from netpattern_engine.tasks import get_task


def _task(kind: str, classes: int = 4):
    task = get_task(TaskSpec(type=kind))
    if kind == "regression":
        task.setup([], [f"y{i}" for i in range(classes)], [])
    else:
        task.setup([f"c{i}" for i in range(classes)], [], [])
    return task


def _pretrained(
    source: str, name: str, weights=None, patches=None, **extra
) -> PretrainedArchitecture:
    return PretrainedArchitecture.model_validate(
        {
            "kind": "pretrained",
            "base": {"source": source, "name": name, "weights": weights, **extra},
            "patches": patches or [],
        }
    )


def _meta_forward(built, batch: int = 2):
    built.module.eval()
    with torch.no_grad():
        out = built.module(
            torch.empty((batch, built.input.channels, *built.input.size), device="meta")
        )
    return out["out"] if isinstance(out, dict) else out


@pytest.mark.parametrize(
    "name",
    [
        "alexnet",
        "vgg11",
        "resnet18",
        "densenet121",
        "mobilenet_v3_small",
        "efficientnet_b0",
        "convnext_tiny",
        "squeezenet1_0",
        "regnet_y_400mf",
        "vit_b_16",
    ],
)
def test_task_head_fits_every_torchvision_family(name):
    task = _task("classification.single_label", classes=7)

    built = build_model(
        _pretrained("torchvision", name, weights="DEFAULT"), task, 3, materialize=False
    )

    assert _meta_forward(built).shape == (2, 7)
    assert built.head


def test_golden_pretrained_matches_torchvision_reference():
    """A pretrained spec with no patches gives exactly torchvision's outputs."""
    task = _task("classification.single_label", classes=1000)
    arch = PretrainedArchitecture.model_validate(
        {
            "kind": "pretrained",
            "base": {"source": "torchvision", "name": "resnet18", "weights": "DEFAULT"},
            "replace_head": False,
        }
    )
    ours = build_model(arch, task, 3).module.eval()
    reference = torchvision.models.resnet18(weights="DEFAULT").eval()
    x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        assert torch.equal(ours(x), reference(x))


def test_golden_fx_import_matches_reference_parameter_count():
    reference = torchvision.models.resnet18()
    with torch.device("meta"):
        meta_model = torchvision.models.resnet18()

    view = inspect_model(meta_model, (3, 224, 224))

    assert view["traceable"]
    counted = sum(node.get("params", 0) for node in view["nodes"] if node["kind"] == "call_module")
    assert counted == sum(p.numel() for p in reference.parameters())
    assert any(node.get("group") == "layer1.0" for node in view["nodes"])
    assert all(node["shape"] for node in view["nodes"] if node["kind"] == "call_module")


def test_patches_freeze_replace_insert_and_remove():
    task = _task("classification.single_label", classes=3)
    patches = [
        {"op": "freeze", "targets": ["@backbone"]},
        {
            "op": "replace",
            "target": "fc",
            "with": [
                {"op": "Dropout", "params": {"p": 0.2}},
                {"op": "Linear", "params": {"out_features": "$num_outputs"}},
            ],
        },
        {
            "op": "insert_after",
            "target": "layer4",
            "ops": [{"op": "Dropout2d", "params": {"p": 0.1}}],
        },
        {"op": "remove", "target": "maxpool"},
    ]

    built = build_model(_pretrained("torchvision", "resnet18", patches=patches), task, 3)

    assert built.module(torch.randn(2, 3, 64, 64)).shape == (2, 3)
    trainable = {n for n, p in built.module.named_parameters() if p.requires_grad}
    assert trainable == {"fc.1.weight", "fc.1.bias"}
    assert isinstance(built.module.maxpool, torch.nn.Identity)
    assert isinstance(built.module.layer4[-1], torch.nn.Dropout2d)


def test_patch_on_unknown_module_is_reported():
    task = _task("classification.single_label")
    with pytest.raises(SpecError) as raised:
        build_model(
            _pretrained("torchvision", "resnet18", patches=[{"op": "remove", "target": "nope"}]),
            task,
            3,
            materialize=False,
        )
    assert raised.value.issues[0].field == "architecture.patches[0].target"


def test_timm_model_takes_single_channel_input():
    task = _task("regression", classes=2)

    built = build_model(_pretrained("timm", "resnet18"), task, 1, materialize=False)

    assert built.input.channels == 1 and len(built.input.mean) == 1
    assert _meta_forward(built).shape == (2, 2)


def test_unet_outputs_one_channel_per_class_at_input_size():
    task = _task("segmentation.semantic", classes=3)

    built = build_model(
        _pretrained("netpattern", "unet", encoder="resnet18", options={"image_size": 64}),
        task,
        3,
        materialize=False,
    )

    assert _meta_forward(built).shape == (2, 3, 64, 64)


def test_torchvision_segmentation_head_is_replaced():
    task = _task("segmentation.semantic", classes=2)

    built = build_model(
        _pretrained("torchvision", "fcn_resnet50", weights="DEFAULT"), task, 3, materialize=False
    )

    assert set(built.head) == {"classifier.4", "aux_classifier.4"}
    assert built.module.classifier[4].out_channels == 2


def test_detection_model_gets_background_plus_classes():
    task = _task("detection.bbox", classes=2)

    built = build_model(
        _pretrained("torchvision", "fasterrcnn_mobilenet_v3_large_320_fpn", weights="DEFAULT"),
        task,
        3,
        materialize=False,
    )

    assert built.module.roi_heads.box_predictor.cls_score.out_features == 3
    assert built.head == ["roi_heads.box_predictor"]


def test_wrong_model_for_task_is_rejected():
    with pytest.raises(SpecError):
        build_model(
            _pretrained("torchvision", "resnet18"), _task("detection.bbox"), 3, materialize=False
        )
