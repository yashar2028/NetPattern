"""Composite blocks for the layer editor (ResNet, DenseNet, SE, Inception, MBConv)."""

from __future__ import annotations

import torch
from torch import nn


def conv_bn_act(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    groups: int = 1,
    act: bool = True,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            kernel_size // 2,
            groups=groups,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
    ]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channel dimension of a C x H x W feature map (ConvNeXt style)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = conv_bn_act(in_channels, out_channels, 3, stride)
        self.conv2 = conv_bn_act(out_channels, out_channels, 3, 1, act=False)
        self.shortcut = (
            conv_bn_act(in_channels, out_channels, 1, stride, act=False)
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + self.shortcut(x))


class Bottleneck(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, expansion: int = 4
    ) -> None:
        super().__init__()
        width = max(out_channels // expansion, 1)
        self.conv1 = conv_bn_act(in_channels, width, 1)
        self.conv2 = conv_bn_act(width, width, 3, stride)
        self.conv3 = conv_bn_act(width, out_channels, 1, act=False)
        self.shortcut = (
            conv_bn_act(in_channels, out_channels, 1, stride, act=False)
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv3(self.conv2(self.conv1(x))) + self.shortcut(x))


class DenseBlock(nn.Module):
    def __init__(
        self, in_channels: int, num_layers: int = 4, growth_rate: int = 32, bn_size: int = 4
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        channels = in_channels
        for _ in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, bn_size * growth_rate, 1, bias=False),
                    nn.BatchNorm2d(bn_size * growth_rate),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(bn_size * growth_rate, growth_rate, 3, padding=1, bias=False),
                )
            )
            channels += growth_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]
        for layer in self.layers:
            features.append(layer(torch.cat(features, dim=1)))
        return torch.cat(features, dim=1)


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class InceptionBlock(nn.Module):
    def __init__(self, in_channels: int, c1x1: int, c3x3: int, c5x5: int, pool_proj: int) -> None:
        super().__init__()
        self.branch1 = conv_bn_act(in_channels, c1x1, 1)
        self.branch3 = nn.Sequential(
            conv_bn_act(in_channels, max(c3x3 // 2, 1), 1), conv_bn_act(max(c3x3 // 2, 1), c3x3, 3)
        )
        self.branch5 = nn.Sequential(
            conv_bn_act(in_channels, max(c5x5 // 2, 1), 1), conv_bn_act(max(c5x5 // 2, 1), c5x5, 5)
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, 1, 1), conv_bn_act(in_channels, pool_proj, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.branch1(x), self.branch3(x), self.branch5(x), self.branch_pool(x)], dim=1
        )


class MBConv(nn.Module):
    """MobileNetV2/EfficientNet inverted residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int = 4,
        kernel_size: int = 3,
        stride: int = 1,
        se: bool = True,
    ) -> None:
        super().__init__()
        hidden = in_channels * expand_ratio
        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_channels, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.SiLU(inplace=True),
            ]
        layers += [
            nn.Conv2d(
                hidden, hidden, kernel_size, stride, kernel_size // 2, groups=hidden, bias=False
            ),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        ]
        if se:
            layers.append(SqueezeExcite(hidden, reduction=4 * expand_ratio))
        layers += [nn.Conv2d(hidden, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels)]
        self.block = nn.Sequential(*layers)
        self.residual = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return out + x if self.residual else out


class TimmBackbone(nn.Module):
    """A timm model used as one block: returns its last feature map, or a pooled vector."""

    def __init__(self, name: str, in_channels: int, pretrained: bool, pooled: bool) -> None:
        super().__init__()
        import timm

        self.pooled = pooled
        if pooled:
            self.model = timm.create_model(
                name, pretrained=pretrained, num_classes=0, in_chans=in_channels
            )
        else:
            self.model = timm.create_model(
                name,
                pretrained=pretrained,
                features_only=True,
                out_indices=[-1],
                in_chans=in_channels,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out if self.pooled else out[-1]
