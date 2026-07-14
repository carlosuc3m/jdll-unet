"""Dataset fingerprinting and physically aware 2.5D/3D planning."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import tifffile
from scipy import ndimage as ndi

from .errors import ConfigError, DataFormatError
from .io import ImageMaskPair


@dataclass(frozen=True, slots=True)
class CaseSpacing:
    case: str
    spacing: tuple[float, float, float]
    source: str
    original_spacing: tuple[float, float, float] | None


@dataclass(frozen=True, slots=True)
class DatasetPlan:
    dimensions: str
    cases: tuple[CaseSpacing, ...]
    known_fraction: float
    default_spacing: tuple[float, float, float]
    target_spacing: tuple[float, float, float] | None
    anisotropy_threshold: float
    anisotropic_axis: int | None
    context_spacing: float | None
    context_spacing_reliable: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RuntimeMemoryPlan:
    preferred_patch: tuple[int, ...]
    resolved_patch: tuple[int, ...]
    microbatch_cap: int
    resolved_microbatch: int
    reference_memory_gb: int
    available_memory_gb: float | None
    planning_budget_gb: float
    reductions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _positive_spacing(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        parsed = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return cast(tuple[float, float, float], parsed) if all(np.isfinite(item) and item > 0 for item in parsed) else None


def _sidecar_spacing(path: Path) -> tuple[float, float, float] | None:
    candidates = (path.with_suffix(".json"), Path(f"{path}.json"))
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise DataFormatError(f"Invalid spacing sidecar {candidate}: {exc}") from exc
        spacing = _positive_spacing(payload.get("spacing") if isinstance(payload, dict) else None)
        if spacing is None:
            raise DataFormatError(f"Spacing sidecar {candidate} must contain three positive Z,Y,X values")
        return spacing
    return None


def _ome_spacing(path: Path) -> tuple[float, float, float] | None:
    if path.suffix.lower() not in {".tif", ".tiff"}:
        return None
    try:
        with tifffile.TiffFile(path) as tif:
            metadata = tif.ome_metadata
            imagej = tif.imagej_metadata or {}
            if metadata:
                root = ET.fromstring(metadata)
                pixels = next((element for element in root.iter() if element.tag.endswith("Pixels")), None)
                if pixels is not None:
                    x_text = pixels.attrib.get("PhysicalSizeX")
                    y_text = pixels.attrib.get("PhysicalSizeY")
                    z_text = pixels.attrib.get("PhysicalSizeZ")
                    spacing = _positive_spacing((z_text, y_text, x_text)) if x_text and y_text and z_text else None
                    if spacing is not None:
                        return spacing
            if "spacing" in imagej:
                z = float(imagej["spacing"])
                page = tif.pages[0]
                xres = page.tags.get("XResolution")  # type: ignore[union-attr]
                yres = page.tags.get("YResolution")  # type: ignore[union-attr]
                if xres and yres:
                    xv = xres.value
                    yv = yres.value
                    x_size = float(xv[1]) / float(xv[0])
                    y_size = float(yv[1]) / float(yv[0])
                    return _positive_spacing((z, y_size, x_size))
    except (OSError, ValueError, ET.ParseError, tifffile.TiffFileError):
        return None
    return None


def read_spacing(path: Path) -> tuple[tuple[float, float, float] | None, str]:
    """Read Z,Y,X spacing with explicit sidecars taking precedence over TIFF metadata."""

    sidecar = _sidecar_spacing(path)
    if sidecar is not None:
        return sidecar, "sidecar"
    embedded = _ome_spacing(path)
    return (embedded, "embedded_metadata") if embedded is not None else (None, "missing")


def build_dataset_plan(
    pairs: list[ImageMaskPair],
    dimensions: str,
    *,
    default_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    known_fraction_threshold: float = 0.5,
    target_spacing: str | tuple[float, float, float] = "auto",
    anisotropy_threshold: float = 3.0,
    max_upsampling: float = 3.0,
) -> DatasetPlan:
    if dimensions == "2d":
        return DatasetPlan(dimensions, (), 0.0, default_spacing, None, anisotropy_threshold, None, None, False)
    measured = [read_spacing(pair.image) for pair in pairs]
    known = [spacing for spacing, _source in measured if spacing is not None]
    known_fraction = len(known) / max(1, len(pairs))
    if known_fraction >= known_fraction_threshold and known:
        fill = cast(tuple[float, float, float], tuple(float(v) for v in np.median(np.asarray(known), axis=0)))
        missing_source = "imputed_per_axis_median"
    else:
        fill = default_spacing
        missing_source = "imputed_default"
    cases = tuple(
        CaseSpacing(pair.stem, spacing or fill, source if spacing is not None else missing_source, spacing)
        for pair, (spacing, source) in zip(pairs, measured, strict=True)
    )
    resolved = np.asarray([case.spacing for case in cases], dtype=np.float64)
    if target_spacing == "auto":
        target = np.median(resolved, axis=0)
        ratio = float(target.max() / target.min())
        anisotropic_axis = int(np.argmax(target)) if ratio >= anisotropy_threshold else None
        if anisotropic_axis is not None:
            axis_values = resolved[:, anisotropic_axis]
            robust_fine = float(np.percentile(axis_values, 10))
            upsampling_floor = float(np.percentile(axis_values, 95)) / max_upsampling
            target[anisotropic_axis] = max(robust_fine, upsampling_floor)
    else:
        parsed = _positive_spacing(target_spacing)
        if parsed is None:
            raise ConfigError("target_spacing must be 'auto' or three positive Z,Y,X values")
        target = np.asarray(parsed)
        anisotropic_axis = int(np.argmax(target)) if float(target.max() / target.min()) >= anisotropy_threshold else None
    context_reliable = known_fraction >= known_fraction_threshold
    context_spacing = float(np.median(resolved[:, 0])) if context_reliable else None
    return DatasetPlan(
        dimensions,
        cases,
        known_fraction,
        default_spacing,
        cast(tuple[float, float, float], tuple(float(v) for v in target)),
        anisotropy_threshold,
        anisotropic_axis,
        context_spacing,
        context_reliable,
    )


def resolve_context_stride(policy: str, *, fixed_stride: int, target_spacing: float | None, z_spacing: float) -> int:
    if policy == "adjacent":
        return 1
    if policy == "fixed_stride":
        if fixed_stride < 1:
            raise ConfigError("context_stride must be at least 1 for fixed_stride")
        return fixed_stride
    if policy == "nearest_physical":
        if target_spacing is None:
            return 1
        return max(1, int(math.floor(target_spacing / z_spacing + 0.5)))
    raise ConfigError(f"Unsupported context_stride_policy: {policy}")


def derive_stage_geometry(
    patch_size: tuple[int, ...],
    spacing: tuple[float, ...],
    stages: int,
    *,
    anisotropy_threshold: float = 2.0,
    minimum_feature_map: int = 4,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Derive per-stage kernels and downsampling strides from physical resolution."""

    shape = np.asarray(patch_size, dtype=np.int64)
    current_spacing = np.asarray(spacing, dtype=np.float64)
    kernels: list[tuple[int, ...]] = []
    strides: list[tuple[int, ...]] = []
    for stage in range(stages):
        finest = float(current_spacing.min())
        kernels.append(tuple(1 if value / finest > anisotropy_threshold else 3 for value in current_spacing))
        if stage == stages - 1:
            break
        stride = tuple(
            2 if shape[axis] // 2 >= minimum_feature_map and current_spacing[axis] / finest <= anisotropy_threshold else 1
            for axis in range(len(shape))
        )
        if all(value == 1 for value in stride):
            eligible = [axis for axis in range(len(shape)) if shape[axis] // 2 >= minimum_feature_map]
            if eligible:
                stride = tuple(2 if axis in eligible else 1 for axis in range(len(shape)))
        strides.append(stride)
        shape //= np.asarray(stride)
        current_spacing *= np.asarray(stride)
    return tuple(kernels), tuple(strides)


def resample_image_mask(
    image: np.ndarray,
    mask: np.ndarray,
    source_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    factors = np.asarray(source_spacing) / np.asarray(target_spacing)
    if np.allclose(factors, 1.0):
        return image, mask
    resampled_image = np.stack([ndi.zoom(channel, factors, order=1, mode="nearest", prefilter=False) for channel in image])
    resampled_mask = ndi.zoom(mask, factors, order=0, mode="nearest", prefilter=False)
    return np.ascontiguousarray(resampled_image.astype(np.float32)), np.ascontiguousarray(resampled_mask.astype(mask.dtype))


def restore_continuous_maps(
    maps: np.ndarray,
    native_shape: tuple[int, ...],
) -> np.ndarray:
    factors = np.asarray(native_shape, dtype=np.float64) / np.asarray(maps.shape[1:], dtype=np.float64)
    restored = np.stack([ndi.zoom(channel, factors, order=1, mode="nearest", prefilter=False) for channel in maps])
    crop = tuple(slice(0, size) for size in native_shape)
    if restored.shape[1:] != native_shape:
        pads = tuple((0, max(0, size - current)) for size, current in zip(native_shape, restored.shape[1:], strict=True))
        restored = np.pad(restored, ((0, 0), *pads), mode="edge")
    return np.ascontiguousarray(restored[(slice(None), *crop)])


def estimate_training_memory_bytes(
    patch_size: tuple[int, ...],
    channels: tuple[int, ...],
    blocks: tuple[int, ...],
    *,
    effective_microbatch: int = 1,
    input_channels: int = 1,
    deep_supervision: bool = False,
) -> int:
    """Conservative architecture-relative memory estimate for patch planning."""

    shape = np.asarray(patch_size, dtype=np.int64)
    activation_elements = 0
    parameter_elements = 0
    previous = input_channels
    for level, (width, count) in enumerate(zip(channels, blocks, strict=True)):
        activation_elements += int(np.prod(shape)) * width * max(2, count)
        parameter_elements += previous * width * (3 ** len(shape)) * count
        previous = width
        if level < len(channels) - 1:
            shape = np.maximum(1, shape // 2)
    supervision_factor = 1.12 if deep_supervision else 1.0
    return int(effective_microbatch * activation_elements * 16 * supervision_factor + parameter_elements * 20)


def plan_patch_and_microbatch(
    preferred: tuple[int, ...],
    image_shape: tuple[int, ...],
    channels: tuple[int, ...],
    blocks: tuple[int, ...],
    reference_memory_gb: int,
    microbatch_cap: int,
    *,
    effective_batch_size: int = 4,
    available_memory_bytes: int | None = None,
    memory_fraction: float = 0.8,
    minimum_size: int = 8,
    input_channels: int = 1,
    deep_supervision: bool = False,
) -> RuntimeMemoryPlan:
    """Resolve microbatch first, then shrink a preferred patch to the usable budget."""

    if microbatch_cap < 1 or effective_batch_size < 1:
        raise ConfigError("microbatch_cap and effective_batch_size must be at least one")
    patch = np.minimum(np.asarray(preferred, dtype=np.int64), np.asarray(image_shape, dtype=np.int64))
    reference_bytes = reference_memory_gb * (1024**3)
    usable_source = min(reference_bytes, available_memory_bytes) if available_memory_bytes else reference_bytes
    budget = int(usable_source * memory_fraction)
    eligible_microbatches = [
        value
        for value in range(1, min(microbatch_cap, effective_batch_size) + 1)
        if effective_batch_size % value == 0
    ]
    microbatch = max(eligible_microbatches)
    reductions: list[str] = []

    def estimate() -> int:
        return estimate_training_memory_bytes(
            tuple(int(value) for value in patch),
            channels,
            blocks,
            effective_microbatch=microbatch,
            input_channels=input_channels,
            deep_supervision=deep_supervision,
        )

    while microbatch > 1 and estimate() > budget:
        microbatch = max(value for value in eligible_microbatches if value < microbatch)
        reductions.append("microbatch_reduced_for_memory")
    while estimate() > budget:
        candidates = [axis for axis, value in enumerate(patch) if value > minimum_size]
        if not candidates:
            raise ConfigError("Preset cannot fit its minimum patch with microbatch one in the planning budget")
        axis = max(candidates, key=lambda item: patch[item])
        reduced = max(minimum_size, int(math.floor(patch[axis] * 0.9 / minimum_size) * minimum_size))
        patch[axis] = reduced if reduced < patch[axis] else patch[axis] - 1
        reductions.append("patch_reduced_for_memory")
    return RuntimeMemoryPlan(
        preferred,
        tuple(int(value) for value in patch),
        microbatch_cap,
        microbatch,
        reference_memory_gb,
        available_memory_bytes / (1024**3) if available_memory_bytes is not None else None,
        budget / (1024**3),
        tuple(dict.fromkeys(reductions)),
    )


def fit_patch_to_reference_budget(
    preferred: tuple[int, ...],
    image_shape: tuple[int, ...],
    channels: tuple[int, ...],
    blocks: tuple[int, ...],
    reference_memory_gb: int,
    *,
    memory_fraction: float = 0.8,
    minimum_size: int = 8,
) -> tuple[int, ...]:
    """Shrink a fixed preset patch to its universal reference budget; never enlarge it."""

    return plan_patch_and_microbatch(
        preferred,
        image_shape,
        channels,
        blocks,
        reference_memory_gb,
        1,
        memory_fraction=memory_fraction,
        minimum_size=minimum_size,
    ).resolved_patch
