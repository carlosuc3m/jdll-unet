"""JDLL-owned compact UNet variants."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

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


def _is_3d(dimensions: str) -> bool:
    return dimensions.lower() == "3d"


def _normalization(name: str, channels: int, dimensions: str = "2d") -> nn.Module:
    name = name.lower()
    if name in {"none", "identity"}:
        return nn.Identity()
    if name == "batch":
        return nn.BatchNorm3d(channels) if _is_3d(dimensions) else nn.BatchNorm2d(channels)
    if name == "instance":
        return nn.InstanceNorm3d(channels, affine=True) if _is_3d(dimensions) else nn.InstanceNorm2d(channels, affine=True)
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
        normalization: str = "group",
        activation: str = "relu",
        dropout: float = 0.0,
        dimensions: str = "2d",
        kernel_size: int | tuple[int, ...] = 3,
    ) -> None:
        super().__init__()
        conv = nn.Conv3d if _is_3d(dimensions) else nn.Conv2d
        dropout_layer = nn.Dropout3d if _is_3d(dimensions) else nn.Dropout2d
        layers: list[nn.Module] = []
        current = in_channels
        for _ in range(convs_per_level):
            padding = tuple(value // 2 for value in kernel_size) if isinstance(kernel_size, (list, tuple)) else kernel_size // 2
            layers.append(conv(current, out_channels, kernel_size=kernel_size, padding=padding, bias=False))  # type: ignore[arg-type]
            layers.append(_normalization(normalization, out_channels, dimensions))
            layers.append(_activation(activation))
            if dropout > 0:
                layers.append(dropout_layer(dropout))
            current = out_channels
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualEncoderBlock(nn.Module):
    """Residual encoder block used by the ResEnc UNet variants."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        convs_per_level: int = 2,
        normalization: str = "group",
        activation: str = "relu",
        dropout: float = 0.0,
        dimensions: str = "2d",
        kernel_size: int | tuple[int, ...] = 3,
    ) -> None:
        super().__init__()
        if convs_per_level < 1:
            raise ValueError("convs_per_level must be at least 1")
        conv = nn.Conv3d if _is_3d(dimensions) else nn.Conv2d
        dropout_layer = nn.Dropout3d if _is_3d(dimensions) else nn.Dropout2d
        layers: list[nn.Module] = []
        current = in_channels
        for index in range(convs_per_level):
            padding = tuple(value // 2 for value in kernel_size) if isinstance(kernel_size, (list, tuple)) else kernel_size // 2
            layers.append(conv(current, out_channels, kernel_size=kernel_size, padding=padding, bias=False))  # type: ignore[arg-type]
            layers.append(_normalization(normalization, out_channels, dimensions))
            if index != convs_per_level - 1:
                layers.append(_activation(activation))
                if dropout > 0:
                    layers.append(dropout_layer(dropout))
            current = out_channels
        self.body = nn.Sequential(*layers)
        self.projection = (
            nn.Identity()
            if in_channels == out_channels
            else conv(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.activation = _activation(activation)
        self.dropout = dropout_layer(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.body(x) + self.projection(x)
        return self.dropout(self.activation(out))


class UNet2D(nn.Module):
    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        is_3d = _is_3d(config.dimensions)
        conv = nn.Conv3d if is_3d else nn.Conv2d
        conv_transpose = nn.ConvTranspose3d if is_3d else nn.ConvTranspose2d
        channels = list(config.channels) if config.channels else [config.base_channels * (2**level) for level in range(config.depth)]
        spatial_dims = 3 if is_3d else 2
        kernels = list(config.kernels) if config.kernels else [(3,) * spatial_dims] * len(channels)
        strides = list(config.strides) if config.strides else [(2,) * spatial_dims] * (len(channels) - 1)
        blocks = list(config.encoder_blocks) if config.encoder_blocks else [config.convs_per_level] * len(channels)
        self.encoders = nn.ModuleList()
        in_channels = config.input_channels
        encoder_block = ResidualEncoderBlock if config.block_type == "residual" else ConvBlock
        for level, out_channels in enumerate(channels):
            self.encoders.append(
                encoder_block(
                    in_channels,
                    out_channels,
                    convs_per_level=blocks[level],
                    normalization=config.normalization,
                    activation=config.activation,
                    dropout=config.dropout,
                    dimensions=config.dimensions,
                    kernel_size=kernels[level],
                )
            )
            in_channels = out_channels
        pool_cls = nn.MaxPool3d if is_3d else nn.MaxPool2d
        self.pools = nn.ModuleList(pool_cls(kernel_size=stride, stride=stride) for stride in strides)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.deep_supervision_heads = nn.ModuleList()
        for level in range(config.depth - 1, 0, -1):
            stride = strides[level - 1]
            self.upconvs.append(
                conv_transpose(channels[level], channels[level - 1], kernel_size=stride, stride=stride)  # type: ignore[arg-type]
            )
            self.decoders.append(
                ConvBlock(
                    channels[level - 1] * 2,
                    channels[level - 1],
                    convs_per_level=config.convs_per_level,
                    normalization=config.normalization,
                    activation=config.activation,
                    dropout=config.dropout,
                    dimensions=config.dimensions,
                    kernel_size=kernels[level - 1],
                )
            )
            if config.deep_supervision and level > 1:
                self.deep_supervision_heads.append(conv(channels[level - 1], config.output_channels, kernel_size=1))
        self.out = conv(channels[0], config.output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        skips: list[torch.Tensor] = []
        for index, encoder in enumerate(self.encoders):
            x = encoder(x)
            skips.append(x)
            if index != len(self.encoders) - 1:
                x = self.pools[index](x)
        auxiliary_outputs: list[torch.Tensor] = []
        head_index = 0
        for decoder_index, (upconv, decoder, skip) in enumerate(
            zip(self.upconvs, self.decoders, reversed(skips[:-1]), strict=True)
        ):
            x = upconv(x)
            if x.shape[2:] != skip.shape[2:]:
                mode = "trilinear" if _is_3d(self.config.dimensions) else "bilinear"
                x = F.interpolate(x, size=skip.shape[2:], mode=mode, align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)
            if self.config.deep_supervision and decoder_index < len(self.decoders) - 1:
                auxiliary_outputs.append(self.deep_supervision_heads[head_index](x))
                head_index += 1
        primary = self.out(x)
        if not self.config.deep_supervision:
            return primary
        return [primary, *reversed(auxiliary_outputs)]


def build_unet(
    architecture: str | ArchitectureConfig = "resenc-tiny-2d",
    input_channels: int = 1,
    output_channels: int = 1,
    normalization: str = "group",
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
