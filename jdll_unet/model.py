"""JDLL-owned compact UNet variants."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .config import ArchitectureConfig, architecture_defaults


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


def _normalization(name: str, channels: int) -> nn.Module:
    name = name.lower()
    if name in {"none", "identity"}:
        return nn.Identity()
    if name == "batch":
        return nn.BatchNorm2d(channels)
    if name == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if name == "group":
        groups = min(8, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unsupported normalization: {name}")


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        convs_per_level: int = 2,
        normalization: str = "batch",
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = in_channels
        for _ in range(convs_per_level):
            layers.append(nn.Conv2d(current, out_channels, kernel_size=3, padding=1, bias=False))
            layers.append(_normalization(normalization, out_channels))
            layers.append(_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            current = out_channels
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet2D(nn.Module):
    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        channels = [config.base_channels * (2**level) for level in range(config.depth)]
        self.encoders = nn.ModuleList()
        in_channels = config.input_channels
        for out_channels in channels:
            self.encoders.append(
                ConvBlock(
                    in_channels,
                    out_channels,
                    convs_per_level=config.convs_per_level,
                    normalization=config.normalization,
                    activation=config.activation,
                    dropout=config.dropout,
                )
            )
            in_channels = out_channels
        self.pool = nn.MaxPool2d(2)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level in range(config.depth - 1, 0, -1):
            self.upconvs.append(nn.ConvTranspose2d(channels[level], channels[level - 1], kernel_size=2, stride=2))
            self.decoders.append(
                ConvBlock(
                    channels[level - 1] * 2,
                    channels[level - 1],
                    convs_per_level=config.convs_per_level,
                    normalization=config.normalization,
                    activation=config.activation,
                    dropout=config.dropout,
                )
            )
        self.out = nn.Conv2d(channels[0], config.output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for index, encoder in enumerate(self.encoders):
            x = encoder(x)
            skips.append(x)
            if index != len(self.encoders) - 1:
                x = self.pool(x)
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips[:-1])):
            x = upconv(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)
        return self.out(x)


def build_unet(
    architecture: str | ArchitectureConfig = "tiny-2d",
    input_channels: int = 1,
    output_channels: int = 1,
    normalization: str = "batch",
) -> UNet2D:
    if isinstance(architecture, ArchitectureConfig):
        config = architecture
    else:
        config = architecture_defaults(
            architecture,
            input_channels=input_channels,
            output_channels=output_channels,
            normalization=normalization,
        )
    return UNet2D(config)
