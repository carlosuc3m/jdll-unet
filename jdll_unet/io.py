"""Image, mask, and dataset layout I/O."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as imageio
import numpy as np
import tifffile

from .errors import DataFormatError, DatasetError

IMAGE_ALIASES = ("images", "image", "imgs", "img", "data")
MASK_ALIASES = ("masks", "mask", "labels", "label", "gt")
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
MASK_EXTENSIONS = {".tif", ".tiff", ".png"}
IMAGE_SUFFIXES = ("_image", "-image", "_img", "-img", "_raw", "-raw")
MASK_SUFFIXES = ("_mask", "-mask", "_label", "-label", "_labels", "-labels", "_gt", "-gt")


@dataclass(frozen=True, slots=True)
class ImageMaskPair:
    image: Path
    mask: Path
    stem: str


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


def load_array(path: Path | str) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise DataFormatError(f"Image file does not exist: {path}")
    if not path.is_file():
        raise DataFormatError(f"Image path is not a file: {path}")
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        return tifffile.imread(path)
    if suffix in IMAGE_EXTENSIONS:
        return imageio.imread(path)
    raise DataFormatError(f"Unsupported image format: {path.suffix}")


def _looks_like_rgb_last_axis(arr: np.ndarray) -> bool:
    return arr.ndim == 3 and arr.shape[-1] in {1, 3, 4} and arr.shape[0] not in {1, 3, 4}


def load_image(path: Path | str, dimensions: str = "2d") -> np.ndarray:
    """Load an image as float32 channels-first C,Y,X or C,Z,Y,X without normalizing."""

    arr = np.asarray(load_array(path))
    dimensions = dimensions.lower()
    if dimensions in {"3d", "2.5d"}:
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
                    f"Ambiguous 3D multichannel image shape {arr.shape} for {path}; "
                    "expected C,Z,Y,X or Z,Y,X,C."
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


def _collapse_mask_channels(arr: np.ndarray, path: Path) -> np.ndarray:
    if arr.ndim != 3 or arr.shape[-1] not in {3, 4}:
        return arr
    channels = arr[..., :3]
    if np.all(channels == channels[..., :1]):
        return channels[..., 0]
    raise DataFormatError(
        f"Mask {path} has RGB channels that are not a duplicated label plane; "
        "store masks as single-channel integer label images."
    )


def load_mask(path: Path | str, dimensions: str | None = None) -> np.ndarray:
    """Load a 2D or 3D integer mask while preserving label values."""

    path = Path(path)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        raise DataFormatError("JPEG masks are not supported because compression changes labels")
    arr = np.asarray(load_array(path))
    arr = _collapse_mask_channels(arr, path)
    if dimensions == "2d" and arr.ndim != 2:
        raise DataFormatError(f"2D masks must have shape Y,X, got shape {arr.shape}")
    if dimensions in {"3d", "2.5d"}:
        if arr.ndim != 3:
            raise DataFormatError(f"3D masks must have shape Z,Y,X, got shape {arr.shape}")
        if arr.shape[-1] in {3, 4} and arr.shape[0] not in {1, 3, 4}:
            raise DataFormatError(f"Mask {path} looks like a 2D RGB image, not a 3D label volume")
    elif arr.ndim not in {2, 3}:
        raise DataFormatError(f"Masks must be 2D or 3D integer label arrays, got shape {arr.shape}")
    if np.issubdtype(arr.dtype, np.floating):
        if not np.all(np.isfinite(arr)):
            raise DataFormatError(f"Mask {path} contains non-finite values")
        if not np.allclose(arr, np.round(arr)):
            raise DataFormatError(f"Mask {path} contains non-integer floating values")
    return np.ascontiguousarray(arr.astype(np.int64, copy=False))


def normalize_image(image: np.ndarray, normalization: dict | object | None = None) -> np.ndarray:
    """Normalize channels independently with configurable conservative defaults."""

    if normalization is None:
        norm_type, low, high, eps = "percentile", 1.0, 99.8, 1e-6
    elif isinstance(normalization, dict):
        norm_type = normalization.get("type", "percentile")
        low = float(normalization.get("low", 1.0))
        high = float(normalization.get("high", 99.8))
        eps = float(normalization.get("eps", 1e-6))
    else:
        norm_type = getattr(normalization, "type", "percentile")
        low = float(getattr(normalization, "low", 1.0))
        high = float(getattr(normalization, "high", 99.8))
        eps = float(getattr(normalization, "eps", 1e-6))

    img = image.astype(np.float32, copy=True)
    if norm_type == "none":
        return img
    if norm_type == "minmax":
        mins = img.reshape(img.shape[0], -1).min(axis=1)
        maxs = img.reshape(img.shape[0], -1).max(axis=1)
        for channel, mn, mx in zip(range(img.shape[0]), mins, maxs, strict=True):
            img[channel] = (img[channel] - mn) / max(float(mx - mn), eps)
        return img
    if norm_type == "zscore":
        means = img.reshape(img.shape[0], -1).mean(axis=1)
        stds = img.reshape(img.shape[0], -1).std(axis=1)
        for channel, mean, std in zip(range(img.shape[0]), means, stds, strict=True):
            img[channel] = (img[channel] - mean) / max(float(std), eps)
        return img
    if norm_type != "percentile":
        raise DataFormatError(f"Unsupported normalization type: {norm_type}")

    for channel in range(img.shape[0]):
        lo, hi = np.percentile(img[channel], [low, high])
        img[channel] = np.clip((img[channel] - lo) / max(float(hi - lo), eps), 0.0, 1.0)
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
