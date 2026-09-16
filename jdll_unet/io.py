"""Image, mask, and dataset layout I/O."""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import DataFormatError, DatasetError
from .image_reading import ImageReadSession, current_read_session, decode_pixels

IMAGE_ALIASES = ("images", "image", "imgs", "img", "data")
MASK_ALIASES = ("masks", "mask", "labels", "label", "gt")
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg"}
MASK_EXTENSIONS = {".tif", ".tiff", ".png", ".bmp"}
IMAGE_SUFFIXES = ("_image", "-image", "_img", "-img", "_raw", "-raw")
MASK_SUFFIXES = ("_mask", "-mask", "_label", "-label", "_labels", "-labels", "_gt", "-gt")


@dataclass(frozen=True, slots=True)
class ImageMaskPair:
    image: Path
    mask: Path
    stem: str
    image_axes: str | None = None
    mask_axes: str | None = None
    spatial_shape: tuple[int, ...] = ()
    region: tuple[tuple[int, int], ...] = ()
    source_id: str = ""
    split_origin: str = "provided"
    image_channels: int = 0
    plane_positive_counts: tuple[int, ...] = ()
    label_values: tuple[int, ...] = ()
    source_kind: str = ""
    eligible_centers: tuple[int, ...] | None = None

    @property
    def domain_shape(self) -> tuple[int, ...]:
        return tuple(end - start for start, end in self.region) if self.region else self.spatial_shape


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    train: list[ImageMaskPair]
    val: list[ImageMaskPair]
    explicit_val: bool


def _iter_files(folder: Path, extensions: set[str]) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in extensions and not path.name.startswith(".")
    )


def _canonical_stem(path: Path, suffixes: Sequence[str]) -> str:
    stem = path.stem
    lower = stem.lower()
    for suffix in suffixes:
        if lower.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _find_alias_folder(root: Path, aliases: Iterable[str]) -> Path | None:
    if not root.exists():
        return None
    candidates = {child.name.lower(): child for child in root.iterdir() if child.is_dir()}
    for alias in aliases:
        if alias in candidates:
            return candidates[alias]
    return None


def pair_images_and_masks(image_dir: Path, mask_dir: Path) -> list[ImageMaskPair]:
    """Pair images and masks by stem, accepting common suffix variants."""

    images = _iter_files(image_dir, IMAGE_EXTENSIONS)
    masks = _iter_files(mask_dir, MASK_EXTENSIONS)
    mask_lookup: dict[str, list[Path]] = {}
    for mask in masks:
        keys = {mask.stem, _canonical_stem(mask, MASK_SUFFIXES)}
        for key in keys:
            mask_lookup.setdefault(key.lower(), []).append(mask)

    pairs: list[ImageMaskPair] = []
    missing: list[str] = []
    for image in images:
        image_keys = [image.stem, _canonical_stem(image, IMAGE_SUFFIXES)]
        candidates: list[Path] = []
        for key in image_keys:
            candidates.extend(mask_lookup.get(key.lower(), []))
        unique = sorted(set(candidates))
        if not unique:
            missing.append(image.name)
            continue
        if len(unique) > 1:
            names = ", ".join(path.name for path in unique)
            raise DatasetError(f"Ambiguous mask match for {image.name}: {names}")
        pairs.append(ImageMaskPair(image=image, mask=unique[0], stem=_canonical_stem(image, IMAGE_SUFFIXES)))

    if missing:
        sample = ", ".join(missing[:5])
        raise DatasetError(f"Missing masks for {len(missing)} image(s): {sample}")
    if not pairs:
        raise DatasetError(f"No image/mask pairs found in {image_dir} and {mask_dir}")
    return pairs


def discover_dataset(dataset_path: Path | str) -> DatasetSplits:
    """Discover supported dataset layouts and return paired train/val splits."""

    root = Path(dataset_path)
    if not root.exists():
        raise DatasetError(f"Dataset path does not exist: {root}")
    if not root.is_dir():
        raise DatasetError(f"Dataset path must be a directory: {root}")

    train_root = root / "train"
    val_root = root / "val"
    train_image_dir = _find_alias_folder(train_root, IMAGE_ALIASES)
    train_mask_dir = _find_alias_folder(train_root, MASK_ALIASES)
    val_image_dir = _find_alias_folder(val_root, IMAGE_ALIASES)
    val_mask_dir = _find_alias_folder(val_root, MASK_ALIASES)

    if train_image_dir and train_mask_dir:
        train_pairs = pair_images_and_masks(train_image_dir, train_mask_dir)
        val_pairs = pair_images_and_masks(val_image_dir, val_mask_dir) if val_image_dir and val_mask_dir else []
        return DatasetSplits(train=train_pairs, val=val_pairs, explicit_val=bool(val_pairs))

    image_dir = _find_alias_folder(root, IMAGE_ALIASES)
    mask_dir = _find_alias_folder(root, MASK_ALIASES)
    if image_dir and mask_dir:
        return DatasetSplits(train=pair_images_and_masks(image_dir, mask_dir), val=[], explicit_val=False)

    raise DatasetError(
        "Unsupported dataset layout. Expected images/masks or train/images, train/masks, val/images, val/masks."
    )


def load_array(path: Path | str, *, role: str = "image", session: ImageReadSession | None = None) -> np.ndarray:
    return (session or current_read_session()).pixels(path, role=role)


def _load_with_axes(path: Path, role: str, session: ImageReadSession | None) -> tuple[np.ndarray, str | None, str]:
    from .geometry import _normalized_shape, _reader_array_info

    def read(reader: Any, fmt: str) -> tuple[np.ndarray, str | None, str]:
        _shape, candidates, provenance = _reader_array_info(reader, fmt, role)
        axes = candidates[0] if provenance in {"explicit_tiff_axes", "raster_format"} else None
        if axes is not None:
            _normalized_shape(_shape, axes)
        return decode_pixels(reader, fmt, role), axes, fmt

    return (session or current_read_session()).read(path, read, role=role)


def _looks_like_rgb_last_axis(arr: np.ndarray) -> bool:
    return arr.ndim == 3 and arr.shape[-1] in {1, 3, 4} and arr.shape[0] not in {1, 3, 4}


def load_image(path: Path | str, dimensions: str = "2d", *, session: ImageReadSession | None = None) -> np.ndarray:
    """Load an image as float32 channels-first C,Y,X or C,Z,Y,X without normalizing."""

    path = Path(path)
    arr, axes, file_format = _load_with_axes(path, "image", session)
    if file_format == "BMP" and dimensions.lower() in {"3d", "2.5d"}:
        raise DataFormatError("BMP is supported only for 2D images; use a TIFF stack for volumetric data")
    dimensions = dimensions.lower()
    if axes is not None:
        from .geometry import _normalize_array, _normalized_shape

        _channels, spatial, _singleton = _normalized_shape(arr.shape, axes)
        if len(spatial) != (3 if dimensions in {"3d", "2.5d"} else 2):
            raise DataFormatError(f"Image {path} axes {axes} and shape {arr.shape} are incompatible with {dimensions}")
        out = _normalize_array(arr, axes, mask=False)
    elif dimensions in {"3d", "2.5d"}:
        if arr.ndim == 3:
            if _looks_like_rgb_last_axis(arr):
                raise DataFormatError(
                    f"Image {path} looks like a 2D RGB image, not a 3D volume. "
                    "Use a TIFF stack with shape Z,Y,X or a multichannel volume with shape C,Z,Y,X or Z,Y,X,C."
                )
            if arr.shape[0] <= 1:
                raise DataFormatError(f"3D image {path} must have a real Z dimension, got shape {arr.shape}")
            out = arr[None, ...]
        elif arr.ndim == 4:
            first_is_channel = arr.shape[0] in {1, 2, 3, 4}
            last_is_channel = arr.shape[-1] in {1, 2, 3, 4}
            if first_is_channel and not last_is_channel:
                out = arr
            elif last_is_channel and not first_is_channel:
                out = np.moveaxis(arr, -1, 0)
            else:
                raise DataFormatError(
                    f"Ambiguous 3D multichannel image shape {arr.shape} for {path}; expected C,Z,Y,X or Z,Y,X,C."
                )
        else:
            raise DataFormatError(f"Unsupported 3D image rank {arr.ndim} for {path}")
    elif arr.ndim == 2:
        out = arr[None, ...]
    elif arr.ndim == 3:
        out = np.moveaxis(arr[..., :3], -1, 0) if arr.shape[-1] in {1, 3, 4} and arr.shape[0] not in {1, 3, 4} else arr
    else:
        raise DataFormatError(f"Unsupported image rank {arr.ndim} for {path}")
    if not np.all(np.isfinite(out)):
        raise DataFormatError(f"Image {path} contains non-finite values")
    return np.ascontiguousarray(out.astype(np.float32, copy=False))


def _warn_discarded_mask_channels(path: Path, count: int) -> None:
    if count > 1:
        warnings.warn(
            f"Mask {path} has {count} channels; only the first channel will be used",
            RuntimeWarning,
            stacklevel=3,
        )


def _collapse_mask_channels(arr: np.ndarray, path: Path, dimensions: str | None) -> np.ndarray:
    if dimensions == "2d":
        if arr.ndim == 3 and arr.shape[-1] > 1:
            _warn_discarded_mask_channels(path, int(arr.shape[-1]))
            return arr[..., 0]
        if arr.ndim == 3 and arr.shape[-1] == 1:
            return arr[..., 0]
        return arr
    if dimensions in {"3d", "2.5d"}:
        if arr.ndim != 4:
            return arr
        first_is_channel = arr.shape[0] <= 4 and arr.shape[-1] > 4
        last_is_channel = arr.shape[-1] <= 4 and arr.shape[0] > 4
        if first_is_channel:
            _warn_discarded_mask_channels(path, int(arr.shape[0]))
            return arr[0]
        if last_is_channel:
            _warn_discarded_mask_channels(path, int(arr.shape[-1]))
            return arr[..., 0]
        raise DataFormatError(
            f"Ambiguous 3D multichannel mask shape {arr.shape} for {path}; expected C,Z,Y,X or Z,Y,X,C"
        )
    if arr.ndim == 3 and arr.shape[-1] in {1, 3, 4}:
        if np.all(arr == arr[..., :1]):
            _warn_discarded_mask_channels(path, int(arr.shape[-1]))
            return arr[..., 0]
        raise DataFormatError(
            f"Ambiguous mask shape {arr.shape} for {path}; pass dimensions='2d' for Y,X,C or '3d' for Z,Y,X"
        )
    return arr


def validate_mask_labels(arr: np.ndarray, path: Path) -> None:
    if not np.all(np.isfinite(arr)):
        raise DataFormatError(f"Mask {path} contains non-finite values")
    if np.issubdtype(arr.dtype, np.floating) and not np.all(arr == np.round(arr)):
        raise DataFormatError(f"Mask {path} contains non-integer floating values")
    if np.any(arr < 0) or np.any(arr > np.iinfo(np.int64).max):
        raise DataFormatError(f"Mask {path} requires nonnegative integer labels within int64 range")


def load_mask(
    path: Path | str, dimensions: str | None = None, *, session: ImageReadSession | None = None
) -> np.ndarray:
    """Load a 2D or 3D integer mask while preserving label values."""

    path = Path(path)
    arr, axes, file_format = _load_with_axes(path, "mask", session)
    if file_format == "BMP" and dimensions in {"3d", "2.5d"}:
        raise DataFormatError("BMP is supported only for 2D masks; use a TIFF stack for volumetric labels")
    if axes is not None:
        from .geometry import _normalize_array, _normalized_shape

        channels, _spatial, _singleton = _normalized_shape(arr.shape, axes)
        _warn_discarded_mask_channels(path, channels)
        arr = _normalize_array(arr, axes, mask=True)
    else:
        arr = _collapse_mask_channels(arr, path, dimensions)
    if dimensions == "2d" and arr.ndim != 2:
        raise DataFormatError(f"2D masks must have shape Y,X, got shape {arr.shape}")
    if dimensions in {"3d", "2.5d"}:
        if arr.ndim != 3:
            raise DataFormatError(f"3D masks must have shape Z,Y,X, got shape {arr.shape}")
    elif arr.ndim not in {2, 3}:
        raise DataFormatError(f"Masks must be 2D or 3D integer label arrays, got shape {arr.shape}")
    validate_mask_labels(arr, path)
    return np.ascontiguousarray(arr.astype(np.int64, copy=False))


def fit_normalization(image: np.ndarray, normalization: dict | object | None = None) -> dict:
    """Fit per-channel statistics on real source/domain pixels only."""

    def setting(name: str, default: Any) -> Any:
        return (
            normalization.get(name, default)
            if isinstance(normalization, dict)
            else getattr(normalization, name, default)
        )

    kind = str(setting("type", "percentile"))
    low, high, eps = float(setting("low", 1.0)), float(setting("high", 99.8)), float(setting("eps", 1e-6))
    parameters = []
    for source_channel in image:
        channel = source_channel.astype(np.float32, copy=False)
        if kind == "none":
            offset, scale = 0.0, 1.0
        elif kind == "percentile":
            offset, maximum = np.percentile(channel, [low, high])
            scale = max(float(maximum - offset), eps)
        elif kind == "minmax":
            offset = channel.min()
            scale = max(float(channel.max() - offset), eps)
        elif kind == "zscore":
            offset = channel.mean()
            scale = max(float(channel.std()), eps)
        else:
            raise DataFormatError(f"Unsupported normalization type: {kind}")
        parameters.append((offset, scale))
    return {"type": kind, "channels": parameters}


def normalize_image(
    image: np.ndarray, normalization: dict | object | None = None, *, statistics: dict | None = None
) -> np.ndarray:
    """Normalize channels with supplied domain statistics or image-local statistics."""
    stats = statistics if statistics is not None else fit_normalization(image, normalization)
    img = image.astype(np.float32, copy=True)
    if stats["type"] == "none":
        return img
    for channel, (offset, scale) in enumerate(stats["channels"]):
        normalized = (img[channel] - offset) / scale
        img[channel] = np.clip(normalized, 0.0, 1.0) if stats["type"] == "percentile" else normalized
    return img


def read_class_names(dataset_path: Path | str) -> list[str] | None:
    root = Path(dataset_path)
    for name in ("classes.json", "labels.json"):
        path = root / name
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return [str(item) for item in data]
        if isinstance(data, dict):
            values = data.get("classes", data.get("labels", data))
            if isinstance(values, list):
                return [str(item) for item in values]
            if isinstance(values, dict):
                return [str(values[key]) for key in sorted(values)]
    return None


def read_class_labels(dataset_path: Path | str) -> list[int] | None:
    """Read declared class identities without learning heads from validation masks."""
    for filename in ("classes.json", "labels.json"):
        path = Path(dataset_path) / filename
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        values = payload.get("classes", payload.get("labels", payload)) if isinstance(payload, dict) else payload
        try:
            if isinstance(values, dict):
                labels = [int(value) for value in values]
            elif isinstance(values, list) and all(isinstance(value, dict) and "id" in value for value in values):
                labels = [int(value["id"]) for value in values]
            elif isinstance(values, list) and all(
                isinstance(value, int) and not isinstance(value, bool) for value in values
            ):
                labels = values
            elif isinstance(values, list) and all(isinstance(value, str) for value in values):
                first = 0 if values and values[0].lower() in {"background", "bg"} else 1
                labels = list(range(first, len(values) + first))
            else:
                raise ValueError("unsupported class identities")
        except (ValueError, TypeError) as exc:
            raise DatasetError(
                f"Class metadata {path} must declare label IDs or an ordered list of class names"
            ) from exc
        if any(value < 0 for value in labels) or len(labels) != len(set(labels)):
            raise DatasetError(f"Class metadata {path} must have unique nonnegative IDs")
        return sorted(value for value in labels if value > 0)
    return None
