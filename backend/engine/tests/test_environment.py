"""Checks that the pinned sandbox environment provides the libraries the engine needs.

These run inside the Nix engine-dev shell; outside it they are skipped.
"""

import pytest


@pytest.mark.parametrize("module", ["torchvision", "timm", "torchmetrics", "pydicom", "nibabel"])
def test_library_is_importable(module):
    pytest.importorskip("torch")
    pytest.importorskip(module)


def test_resnet18_forward_pass_on_cpu():
    torch = pytest.importorskip("torch")
    models = pytest.importorskip("torchvision.models")

    model = models.resnet18(weights=None).eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 3, 224, 224))

    assert logits.shape == (1, 1000)
