"""Source geometry, bounded domain reads, and reproducible spatial holdouts."""

from __future__ import annotations

import hashlib
import math
import warnings
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ConfigError, DataFormatError, DatasetError
from .image_reading import ImageReadSession, current_read_session
from .io import ImageMaskPair, load_image, load_mask, validate_mask_labels


def source_identity(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_dev}:{stat.st_ino}"


def stable_seed(seed: int, epoch: int, identity: str) -> int:
    digest = hashlib.sha256(f"{seed}:{epoch}:{identity}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _reader_array_info(reader: Any, file_format: str, role: str) -> tuple[tuple[int, ...], list[str], str]:
    """Inspect axes without decoding pixels; reader selection is shared with I/O."""
    if file_format != "TIFF":
        channels = 3 if reader.mode == "P" and role == "image" else len(reader.getbands())
        shape = (reader.height, reader.width) + ((channels,) if channels > 1 else ())
        return shape, ["YX" if len(shape) == 2 else "YXC"], "raster_format"
    series = reader.series[0]
    shape = tuple(series.shape)
    axes = series.axes.upper().replace("S", "C")
    explicit = bool(reader.is_ome or reader.is_imagej)
    shaped = reader.shaped_metadata
    if shaped and isinstance(shaped[0], dict) and "axes" in shaped[0]:
        axes = str(shaped[0]["axes"]).upper().replace("S", "C")
        explicit = True
    if explicit:
        return shape, [axes], "explicit_tiff_axes"
    page = reader.pages[0].keyframe
    if page is None:
        raise DataFormatError("Missing TIFF page metadata")
    if int(page.samplesperpixel) > 1:
        # Samples are a real TIFF channel axis; Q/I page axes remain ambiguous.
        if all(axis in "CZYX" for axis in axes):
            return shape, [axes], "tiff_samples"
        if len(shape) == 4:
            return shape, [axes.replace("Q", "Z").replace("I", "Z")], "tiff_samples_stack"
    if len(shape) == 2:
        return shape, ["YX"], "spatial_pair"
    if len(shape) == 3:
        return shape, ["ZYX", "YXC"], "paired_shape_candidates"
    if len(shape) == 4:
        return shape, ["CZYX", "ZYXC", "ZCYX"], "paired_shape_candidates"
    raise DataFormatError(f"Unsupported TIFF shape {shape}; export with explicit axes")


def _array_info(
    path: Path, *, role: str = "image", session: ImageReadSession | None = None
) -> tuple[tuple[int, ...], list[str], str]:
    return (session or current_read_session()).read(
        path, lambda reader, fmt: _reader_array_info(reader, fmt, role), role=role, pixels=False
    )


def _normalized_shape(shape: tuple[int, ...], axes: str) -> tuple[int, tuple[int, ...], bool]:
    if len(shape) != len(axes) or len(set(axes)) != len(axes):
        raise DataFormatError(f"Invalid axes {axes!r} for shape {shape}")
    for length, axis in zip(shape, axes, strict=True):
        if axis not in "CZYX" and length != 1:
            raise DataFormatError(f"Unsupported axis {axis!r} in {axes}; export a single time point with explicit axes")
    sizes = dict(zip(axes, shape, strict=True))
    if "Y" not in sizes or "X" not in sizes or any(size < 1 for size in shape):
        raise DataFormatError(f"Missing/empty spatial axes in {axes}: {shape}")
    spatial: tuple[int, ...] = (sizes["Y"], sizes["X"])
    if sizes.get("Z", 1) > 1:
        spatial = (sizes["Z"], *spatial)
    return sizes.get("C", 1), spatial, "Z" in sizes and sizes["Z"] == 1


def _normalize_array(array: np.ndarray, axes: str, *, mask: bool) -> np.ndarray:
    for index in reversed(range(len(axes))):
        if axes[index] not in "CZYX" or (axes[index] == "Z" and array.shape[index] == 1):
            array = np.take(array, 0, axis=index)
            axes = axes[:index] + axes[index + 1 :]
    if mask and "C" in axes:
        array = np.take(array, 0, axis=axes.index("C"))
        axes = axes.replace("C", "")
    if not mask and "C" not in axes:
        array = array[None]
        axes = "C" + axes
    ordered = ("" if mask else "C") + ("Z" if "Z" in axes else "") + "YX"
    return array.transpose(tuple(axes.index(axis) for axis in ordered))


class DomainReader:
    """Per-consumer LRU of raw arrays, with a strict byte bound (large arrays bypass it)."""

    def __init__(self, max_bytes: int = 64 * 1024**2, *, session: ImageReadSession | None = None):
        from collections import OrderedDict

        self.max_bytes = max_bytes
        self.session = session or current_read_session()
        self.cache: OrderedDict[tuple[str, int, int, str], np.ndarray] = OrderedDict()
        self.bytes = 0

    def read(self, path: Path, *, role: str = "image") -> np.ndarray:
        stat = path.stat()
        key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size, role)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        array = self.session.pixels(path, role=role, memmap=True)
        if array.nbytes <= self.max_bytes:
            while self.cache and self.bytes + array.nbytes > self.max_bytes:
                _, old = self.cache.popitem(last=False)
                self.bytes -= old.nbytes
            array.flags.writeable = False
            self.cache[key] = array
            self.bytes += array.nbytes
        return array


def _planning_reader() -> DomainReader:
    session = current_read_session()
    if session.domain_reader is None:
        session.domain_reader = DomainReader(session=session)
    return session.domain_reader


def load_domain_mask(
    pair: ImageMaskPair, dimensions: str | None = "2d", reader: DomainReader | None = None, *, raw: bool = False
) -> np.ndarray:
    reader = reader or _planning_reader()
    if pair.mask_axes is None:
        return load_mask(pair.mask, dimensions=dimensions, session=reader.session)
    array = _normalize_array(reader.read(pair.mask, role="mask"), pair.mask_axes, mask=True)
    if pair.region:
        array = array[tuple(slice(start, end) for start, end in pair.region)]
    if not raw:
        validate_mask_labels(array, pair.mask)
    return array if raw else np.ascontiguousarray(array, dtype=np.int64)


def load_domain_image(
    pair: ImageMaskPair, dimensions: str = "2d", reader: DomainReader | None = None, *, raw: bool = False
) -> np.ndarray:
    reader = reader or _planning_reader()
    if pair.image_axes is None:
        return load_image(pair.image, dimensions=dimensions, session=reader.session)
    array = _normalize_array(reader.read(pair.image), pair.image_axes, mask=False)
    if pair.region:
        array = array[(slice(None), *(slice(start, end) for start, end in pair.region))]
    return array if raw else np.ascontiguousarray(array, dtype=np.float32)


def with_region(pair: ImageMaskPair, bounds: tuple[tuple[int, int], ...], origin: str) -> ImageMaskPair:
    current = replace(pair, region=bounds, split_origin=origin, eligible_centers=None)
    mask = load_domain_mask(current)
    counts = tuple(int(value) for value in np.count_nonzero(mask > 0, axis=(-2, -1)).reshape(-1))
    return replace(current, plane_positive_counts=counts, label_values=tuple(int(v) for v in np.unique(mask) if v > 0))


def inspect_pair(pair: ImageMaskPair, *, reader: DomainReader | None = None) -> tuple[ImageMaskPair, dict[str, Any]]:
    reader = reader or _planning_reader()
    image_shape, image_candidates, image_provenance = _array_info(pair.image, session=reader.session)
    mask_shape, mask_candidates, mask_provenance = _array_info(pair.mask, role="mask", session=reader.session)
    matches = []
    for image_axes in image_candidates:
        channels, spatial, singleton = _normalized_shape(image_shape, image_axes)
        for mask_axes in mask_candidates:
            mask_channels, mask_spatial, mask_singleton = _normalized_shape(mask_shape, mask_axes)
            if spatial == mask_spatial:
                matches.append((image_axes, mask_axes, channels, spatial, singleton or mask_singleton, mask_channels))
    # Scalar stacks with identical full spatial geometry are volumes. A generic
    # page axis alone cannot turn an unmatched stack into channels or broadcast labels.
    if len(image_shape) == len(mask_shape) == 3 and image_shape == mask_shape:
        volume_matches = [item for item in matches if item[0] == item[1] == "ZYX"]
        if volume_matches and image_provenance == mask_provenance == "paired_shape_candidates":
            matches = volume_matches
    if (
        len(matches) != 1
        and matches
        and len({(item[2], item[3]) for item in matches}) == 1
        and all(size == 1 for size in image_shape[:-2] + mask_shape[:-2])
    ):
        matches = [matches[0]]
    if len(matches) != 1:
        detail = "Ambiguous axes" if matches else "Incompatible image/mask spatial geometry"
        raise DataFormatError(
            f"{detail}: image={pair.image} shape={image_shape}, mask={pair.mask} shape={mask_shape}; "
            "export matching image/mask data with explicit C,Z,Y,X axes"
        )
    image_axes, mask_axes, channels, spatial, singleton, mask_channels = matches[0]
    if mask_channels > 1:
        warnings.warn(
            f"Mask {pair.mask} has {mask_channels} channels; only the first channel will be used",
            RuntimeWarning,
            stacklevel=2,
        )
    kind = "volume" if len(spatial) == 3 else "singleton_stack" if singleton else "image_2d"
    resolved = replace(
        pair,
        image_axes=image_axes,
        mask_axes=mask_axes,
        spatial_shape=spatial,
        source_id=source_identity(pair.image),
        image_channels=channels,
        source_kind=kind,
    )
    image = load_domain_image(resolved, reader=reader)
    raw_mask = _normalize_array(reader.read(pair.mask, role="mask"), mask_axes, mask=True)
    if not np.all(np.isfinite(image)) or not np.all(np.isfinite(raw_mask)):
        raise DataFormatError(f"Non-finite image or label values: {pair.image}, {pair.mask}")
    if np.any(raw_mask < 0) or np.any(raw_mask > np.iinfo(np.int64).max) or not np.all(raw_mask == np.round(raw_mask)):
        raise DataFormatError(f"Masks require nonnegative integer labels: {pair.mask}")
    counts = tuple(int(value) for value in np.count_nonzero(raw_mask > 0, axis=(-2, -1)).reshape(-1))
    resolved = replace(
        resolved, plane_positive_counts=counts, label_values=tuple(int(v) for v in np.unique(raw_mask) if v > 0)
    )
    details = {
        **case_record(resolved),
        "image_shape": image_shape,
        "mask_shape": mask_shape,
        "image_axes_provenance": image_provenance,
        "mask_axes_provenance": mask_provenance,
        "mask_channels": mask_channels,
    }
    return resolved, details


def case_record(pair: ImageMaskPair) -> dict[str, Any]:
    return {
        **asdict(pair),
        "image": str(pair.image),
        "mask": str(pair.mask),
        "original_image": str(pair.image.resolve()),
        "original_mask": str(pair.mask.resolve()),
        "region": pair.region or tuple((0, size) for size in pair.spatial_shape),
    }


def inspect_sources(
    pairs: list[ImageMaskPair], dimensions: str, emit: Callable[..., Any], check_cancel: Callable[[], None]
) -> tuple[list[ImageMaskPair], list[dict[str, Any]]]:
    accepted, records = [], []
    reader = _planning_reader()
    for pair in pairs:
        check_cancel()
        record: dict[str, Any] = {"image": str(pair.image), "mask": str(pair.mask)}
        try:
            case, record = inspect_pair(pair, reader=reader)
            if dimensions != "2d" and case.source_kind != "volume":
                raise DataFormatError(f"{dimensions} training requires volumes with Z>1; skipped {pair.image}")
        except (DataFormatError, OSError, ValueError) as exc:
            reason = (
                "ambiguous_axes"
                if "Ambiguous" in str(exc)
                else "incompatible_dimension"
                if "requires volumes" in str(exc)
                else "invalid_geometry"
            )
            record.update(status="skipped", reason=reason, message=str(exc))
            for role, path in (("image", pair.image), ("mask", pair.mask)):
                if f"{role}_shape" not in record:
                    try:
                        record[f"{role}_shape"] = _array_info(path, role=role, session=reader.session)[0]
                    except (DataFormatError, OSError, ValueError):
                        record[f"{role}_shape"] = None
            emit("warning", **record)
        else:
            record["status"] = "accepted"
            accepted.append(case)
            if record.get("mask_channels", 1) > 1:
                emit(
                    "warning",
                    message=f"Mask {pair.mask.name}: using the first of {record['mask_channels']} channels.",
                    reason="first_mask_channel",
                    image=str(pair.image),
                    mask=str(pair.mask),
                    mask_channels=record["mask_channels"],
                )
        records.append(record)
    return accepted, records


def padding_extents(shape: tuple[int, ...], patch: tuple[int, ...], ratio: float = 1.0) -> tuple[tuple[int, int], ...]:
    if not math.isfinite(ratio) or ratio < 0 or len(shape) != len(patch):
        raise ConfigError("Padding ratio must be finite and nonnegative; patch and domain ranks must agree")
    pads = tuple(
        (max(0, p - length) // 2, (max(0, p - length) + 1) // 2) for length, p in zip(shape, patch, strict=True)
    )
    for axis, (length, pad) in enumerate(zip(shape, pads, strict=True)):
        if length < 1 or max(pad) > ratio * length:
            raise DatasetError(
                f"Axis {axis}: real length {length} needs padding {pad[0]}+{pad[1]} for patch {patch[axis]}; maximum is {ratio * length:g} per side"
            )
    return pads


def eligible_centers(depth: int, context: int, stride: int, ratio: float = 1.0) -> tuple[int, ...]:
    if depth <= 1:
        return ()
    radius = (context // 2) * stride
    return tuple(
        z
        for z in range(depth)
        if max(0, radius - z) <= ratio * depth and max(0, z + radius - (depth - 1)) <= ratio * depth
    )


def split_sources(
    pairs: list[ImageMaskPair], fraction: float, seed: int
) -> tuple[list[ImageMaskPair], list[ImageMaskPair]]:
    if fraction <= 0:
        raise DatasetError(
            "Training requires validation; validation_fraction must be positive or supply an explicit validation split"
        )
    unique: dict[str, ImageMaskPair] = {}
    for pair in pairs:
        unique.setdefault(pair.source_id or source_identity(pair.image), pair)
    cases = list(unique.values())
    if len(cases) < 2:
        raise DatasetError("A single source requires a feasible spatial validation holdout")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cases))
    count = min(len(cases) - 1, max(1, round(len(cases) * fraction)))
    return (
        [replace(cases[i], split_origin="generated") for i in order[count:]],
        [replace(cases[i], split_origin="generated") for i in order[:count]],
    )


def spatial_holdout(
    pair: ImageMaskPair,
    fraction: float,
    seed: int,
    feasible: Callable[[ImageMaskPair, bool], bool],
    *,
    excluded_regions: set[tuple] | None = None,
    check_cancel: Callable[[], None] | None = None,
) -> tuple[list[ImageMaskPair], list[ImageMaskPair]]:
    if fraction <= 0:
        raise DatasetError("A single-source holdout requires a positive validation_fraction")
    shape = pair.domain_shape
    original = pair.region or tuple((0, length) for length in shape)
    rng = np.random.default_rng(seed)
    axes = rng.permutation(len(shape))
    options = []
    for order, axis in enumerate(axes):
        for count in range(1, shape[axis]):
            options.append((abs(count / shape[axis] - fraction), order, int(axis), count))
    end_first = bool(rng.integers(2))
    for _, _, axis, count in sorted(options):
        if check_cancel is not None:
            check_cancel()
        for val_at_end in (end_first, not end_first):
            start, end = original[axis]
            boundary = end - count if val_at_end else start + count
            train_bounds, val_bounds = list(original), list(original)
            train_bounds[axis] = (start, boundary) if val_at_end else (boundary, end)
            val_bounds[axis] = (boundary, end) if val_at_end else (start, boundary)
            if excluded_regions and tuple(train_bounds) in excluded_regions:
                continue
            train = with_region(pair, tuple(train_bounds), "spatial_holdout")
            val = with_region(pair, tuple(val_bounds), "spatial_holdout")
            if any(train.plane_positive_counts) and feasible(train, True) and feasible(val, False):
                return [train], [val]
    raise DatasetError(
        "No feasible disjoint spatial holdout exists; provide another source or choose a compatible patch/preset"
    )


def assert_disjoint(train: list[ImageMaskPair], val: list[ImageMaskPair]) -> None:
    for left in train:
        for right in val:
            same_image = (left.source_id or source_identity(left.image)) == (
                right.source_id or source_identity(right.image)
            )
            same_mask = source_identity(left.mask) == source_identity(right.mask)
            if (same_image or same_mask) and (
                not left.region
                or not right.region
                or not any(a[1] <= b[0] or b[1] <= a[0] for a, b in zip(left.region, right.region, strict=True))
            ):
                raise DatasetError(
                    f"Explicit split conflict: {left.image} and {right.image} reference overlapping real source data"
                )
