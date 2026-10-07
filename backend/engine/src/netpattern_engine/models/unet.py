"""U-Net with any timm encoder (pretrained or blank) for semantic segmentation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

DEFAULT_DECODER_CHANNELS = (256, 128, 64, 32, 16)


def _double_conv(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(
        self,
        encoder: str,
        in_channels: int,
        num_classes: int,
        pretrained: bool,
        decoder_channels: tuple[int, ...] = DEFAULT_DECODER_CHANNELS,
    ) -> None:
        super().__init__()
        import timm

        self.encoder = timm.create_model(
            encoder, features_only=True, pretrained=pretrained, in_chans=in_channels
        )
        skip_channels = list(self.encoder.feature_info.channels())
        channels = skip_channels[-1]
        self.decoder = nn.ModuleList()
        for i, skip in enumerate(reversed(skip_channels[:-1])):
            out_channels = decoder_channels[min(i, len(decoder_channels) - 1)]
            self.decoder.append(_double_conv(channels + skip, out_channels))
            channels = out_channels
        self.head = nn.Conv2d(channels, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        features = self.encoder(x)
        y = features[-1]
        for block, skip in zip(self.decoder, reversed(features[:-1])):
            y = functional.interpolate(y, size=skip.shape[-2:], mode="nearest")
            y = block(torch.cat([y, skip], dim=1))
        y = self.head(y)
        return functional.interpolate(y, size=size, mode="bilinear", align_corners=False)
