"""Precision contexts and explicit, differentiable MPS compatibility operations."""

from __future__ import annotations

import warnings
from contextlib import AbstractContextManager, nullcontext
from functools import cache
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .errors import ConfigError

MPS_3D_NOTICE = (
    "MPS 3D compatibility mode: transposed convolutions, max pooling and volumetric "
    "resizing run on CPU; convolution blocks remain on MPS. Transfers preserve gradients "
    "and checkpoint weights, but this is hybrid execution and may be slower than CPU-only training."
)


def validate_device_dimensions(device: torch.device, dimensions: str) -> None:
    if device.type == "mps" and dimensions == "3d" and not torch.backends.mps.is_macos_or_newer(13, 2):
        raise ConfigError(
            "3D MPS execution requires macOS 13.2 or newer for Conv3d. "
            "Use device='cpu' on older macOS versions."
        )


def autocast_context(device: torch.device, dtype: torch.dtype = torch.float32) -> AbstractContextManager[Any]:
    # Torch 2.4 rejects MPS autocast even with enabled=False.
    if device.type not in {"cpu", "cuda"}:
        return nullcontext()
    return torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32)


def _on_mps(tensor: torch.Tensor) -> bool:
    return tensor.device.type == "mps"


@cache
def _warn_cpu_operation(name: str) -> None:
    warnings.warn(f"MPS compatibility: {name} runs on CPU with differentiable device transfers.",
                  RuntimeWarning, stacklevel=3)


class CompatibleConvTranspose3d(nn.ConvTranspose3d):
    """Keep parameter names/layout and gradients identical to ConvTranspose3d."""

    def forward(self, input: torch.Tensor, output_size: list[int] | None = None) -> torch.Tensor:
        if not _on_mps(input):
            return super().forward(input, output_size)
        _warn_cpu_operation("ConvTranspose3d")
        assert isinstance(self.padding, tuple)
        output_padding = self._output_padding(input, output_size, list(self.stride), list(self.padding),
                                              list(self.kernel_size), 3, list(self.dilation))
        # Do not detach or move the module itself: the optimizer owns MPS parameters.
        return F.conv_transpose3d(input.cpu(), self.weight.cpu(), self.bias.cpu() if self.bias is not None else None,
                                  self.stride, self.padding, output_padding, self.groups, self.dilation).to(input.device)


class CompatibleMaxPool3d(nn.MaxPool3d):
    def forward(self, input: torch.Tensor) -> Any:
        if not _on_mps(input):
            return super().forward(input)
        _warn_cpu_operation("MaxPool3d")
        result = super().forward(input.cpu())
        if isinstance(result, tuple):
            return tuple(value.to(input.device) for value in result)
        return result.to(input.device)


def interpolate(input: torch.Tensor, size: tuple[int, ...] | torch.Size, *,
                mode: str, align_corners: bool | None = None) -> torch.Tensor:
    if input.ndim == 5 and _on_mps(input):
        _warn_cpu_operation(f"interpolate({mode}, 3d)")
        return F.interpolate(input.cpu(), size=size, mode=mode, align_corners=align_corners).to(input.device)
    return F.interpolate(input, size=size, mode=mode, align_corners=align_corners)


def adaptive_avg_pool(input: torch.Tensor, size: tuple[int, ...]) -> torch.Tensor:
    pool: Any = F.adaptive_avg_pool3d if input.ndim == 5 else F.adaptive_avg_pool2d
    # MPS also restricts some non-divisible 2D adaptive-pooling geometries.
    if _on_mps(input):
        _warn_cpu_operation(f"adaptive_avg_pool{input.ndim - 2}d")
        return pool(input.cpu(), size).to(input.device)
    return pool(input, size)
