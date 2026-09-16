"""PyTorch dataset wrappers for paired JDLL UNet data."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import AugmentationConfig, EmptyPatchError, apply_augmentation, make_augmentation_config
from .errors import DatasetError
from .geometry import DomainReader, eligible_centers, load_domain_image, load_domain_mask, split_sources, stable_seed
from .io import ImageMaskPair, fit_normalization, load_mask, normalize_image
from .planning import resample_image_mask, resolve_context_stride
from .targets import prepare_target


@dataclass(slots=True)
class DatasetInfo:
    input_channels: int
    image_shape: tuple[int, ...]
    label_values: list[int]
    empty_mask_count: int


def split_pairs(
    pairs: list[ImageMaskPair],
    validation_fraction: float,
    seed: int,
) -> tuple[list[ImageMaskPair], list[ImageMaskPair]]:
    return split_sources(pairs, validation_fraction, seed)


def inspect_dataset(pairs: list[ImageMaskPair], dimensions: str = "2d") -> DatasetInfo:
    if not pairs:
        raise ValueError("Cannot inspect an empty dataset")
    if all(pair.image_axes is not None for pair in pairs):
        channels = {pair.image_channels for pair in pairs}
        if len(channels) != 1:
            raise DatasetError("All training sources must have the same number of image modalities")
        return DatasetInfo(
            pairs[0].image_channels,
            pairs[0].domain_shape,
            sorted(set().union(*(set(pair.label_values) for pair in pairs))),
            sum(not any(pair.plane_positive_counts) for pair in pairs),
        )
    image = load_domain_image(pairs[0], dimensions=dimensions)
    image_shape = tuple(int(item) for item in image.shape[1:])
    labels: set[int] = set()
    empty_mask_count = 0
    for pair in pairs:
        current_image = load_domain_image(pair, dimensions=dimensions)
        mask = load_domain_mask(pair, dimensions=dimensions)
        if tuple(current_image.shape[1:]) != tuple(mask.shape):
            raise ValueError(
                f"Image/mask spatial shape mismatch for {pair.image.name}: "
                f"image={tuple(current_image.shape[1:])} mask={tuple(mask.shape)}"
            )
        labels.update(int(v) for v in np.unique(mask) if int(v) != 0)
        if current_image.shape[0] != image.shape[0]:
            raise DatasetError("All training sources must have the same number of image modalities")
        empty_mask_count += int(not np.any(mask != 0))
    return DatasetInfo(
        input_channels=int(image.shape[0]),
        image_shape=image_shape,
        label_values=sorted(labels),
        empty_mask_count=empty_mask_count,
    )


def partition_empty_pairs(
    pairs: list[ImageMaskPair], dimensions: str = "2d"
) -> tuple[list[ImageMaskPair], list[ImageMaskPair]]:
    nonempty: list[ImageMaskPair] = []
    empty: list[ImageMaskPair] = []
    mask_dimensions = dimensions if dimensions in {"3d", "2.5d"} else "2d"
    for pair in pairs:
        positive = (
            any(pair.plane_positive_counts)
            if pair.mask_axes is not None
            else np.any(load_mask(pair.mask, dimensions=mask_dimensions) != 0)
        )
        target = nonempty if positive else empty
        target.append(pair)
    return nonempty, empty


class JdllSegmentationDataset(Dataset):
    def __init__(
        self,
        pairs: list[ImageMaskPair],
        task: str,
        label_values: list[int] | None,
        normalization: object | dict | None,
        augmentation: AugmentationConfig,
        training: bool,
        dimensions: str = "2d",
        seed: int = 0,
        instance_sizes: dict[str, float] | None = None,
        fallback_instance_size: float | None = None,
        context_slices: int = 3,
        context_stride_policy: str = "adjacent",
        context_stride: int = 1,
        context_target_spacing: float | None = None,
        case_spacings: dict[str, tuple[float, float, float]] | None = None,
        target_spacing: tuple[float, float, float] | None = None,
        sample_count: int | None = None,
        max_empty_plane_fraction: float = 0.20,
    ) -> None:
        self.pairs = pairs
        self.task = task
        self.label_values = label_values
        self.normalization = normalization
        self.augmentation = augmentation
        self.training = training
        self.dimensions = dimensions
        self.seed = seed
        self.instance_sizes = instance_sizes or {}
        self.fallback_instance_size = fallback_instance_size
        self.context_slices = context_slices
        self.context_stride_policy = context_stride_policy
        self.context_stride = context_stride
        self.context_target_spacing = context_target_spacing
        self.case_spacings = case_spacings or {}
        self.target_spacing = target_spacing
        self.sample_count = sample_count
        self.max_empty_plane_fraction = max_empty_plane_fraction
        self.reader = DomainReader()
        self._normalization_statistics: dict[tuple, dict] = {}
        self.epoch = 0
        self.epoch_indices: np.ndarray | None = None
        self.sampling_summary: list[dict] = []
        self._source_items: dict[int, list[int]] = {}
        self.items: list[tuple[int, int | None]] = []
        for pair_index, pair in enumerate(pairs):
            is_volume = len(pair.domain_shape) == 3 or (pair.image_axes is None and dimensions == "2.5d")
            if dimensions in {"2d", "2.5d"} and is_volume:
                depth = pair.domain_shape[0] if pair.domain_shape else load_mask(pair.mask, dimensions="2.5d").shape[0]
                stride = resolve_context_stride(
                    context_stride_policy,
                    fixed_stride=context_stride,
                    target_spacing=context_target_spacing,
                    z_spacing=self.case_spacings.get(pair.stem, (1, 1, 1))[0],
                )
                centers = (
                    pair.eligible_centers
                    if pair.eligible_centers is not None
                    else eligible_centers(depth, context_slices, stride, augmentation.max_padding_ratio)
                    if dimensions == "2.5d"
                    else tuple(range(depth))
                )
                self.items.extend((pair_index, z) for z in centers)
            else:
                self.items.append((pair_index, None))
        for index, (pair_index, _z) in enumerate(self.items):
            self._source_items.setdefault(pair_index, []).append(index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        rng = np.random.default_rng(stable_seed(self.seed, epoch, "epoch_samples"))
        pool = []
        groups = {}
        self.sampling_summary = []
        for pair_index, pair in enumerate(self.pairs):
            indices = self._source_items.get(pair_index, [])
            counts = pair.plane_positive_counts
            if not counts:
                mask = load_domain_mask(pair, self.dimensions, self.reader)
                counts = tuple(int(v) for v in np.count_nonzero(mask, axis=(-2, -1)).reshape(-1))
            positive: list[int] = []
            empty: list[int] = []
            for i in indices:
                z = self.items[i][1]
                (positive if (counts[z] if z is not None else sum(counts)) > 0 else empty).append(i)
            plane_volume = bool(indices and self.items[indices[0]][1] is not None)
            quota = (
                min(
                    len(empty),
                    math.floor(len(positive) * self.max_empty_plane_fraction / (1 - self.max_empty_plane_fraction)),
                )
                if plane_volume
                else len(empty)
            )
            if self.augmentation.skip_empty_patches:
                quota = 0
            source_rng = np.random.default_rng(stable_seed(self.seed, epoch, str(pair.image.resolve())))
            selected = source_rng.permutation(empty)[:quota].tolist()
            pool.extend(positive + selected)
            groups[pair_index] = (positive, set(selected), plane_volume)
            self.sampling_summary.append(
                {
                    "source": pair.stem,
                    "nonempty_planes": len(positive),
                    "empty_planes": len(empty),
                    "retained_empty_quota": quota,
                    "selected_empty_planes": [self.items[i][1] for i in selected],
                }
            )
        if not pool:
            raise DatasetError("No eligible training samples remain under the foreground/empty-plane policy")
        self.pool_size = len(pool)
        self.epoch_indices = rng.choice(pool, size=len(self), replace=True)
        source_positions: dict[int, list[int]] = {}
        for position, item in enumerate(self.epoch_indices):
            source_positions.setdefault(self.items[int(item)][0], []).append(position)
        for pair_index, (positive, selected, plane_volume) in groups.items():
            positions = source_positions.get(pair_index, [])
            negatives = [n for n in positions if int(self.epoch_indices[n]) in selected]
            allowed = math.floor(len(positions) * self.max_empty_plane_fraction) if plane_volume else len(negatives)
            if len(negatives) > allowed:
                for position in rng.permutation(negatives)[allowed:]:
                    self.epoch_indices[position] = rng.choice(positive)
            self.sampling_summary[pair_index].update(
                epoch_draws=len(positions), empty_plane_draws=min(len(negatives), allowed)
            )

    def __len__(self) -> int:
        return self.sample_count if self.training and self.sample_count is not None else len(self.items)

    def _load_item(self, item_index: int) -> tuple[ImageMaskPair, np.ndarray, np.ndarray]:
        pair_index, center_z = self.items[item_index % len(self.items)]
        pair = self.pairs[pair_index]
        image = load_domain_image(pair, dimensions=self.dimensions, reader=self.reader, raw=True)
        mask = load_domain_mask(pair, dimensions=self.dimensions, reader=self.reader, raw=True)
        if self.dimensions == "2d" and center_z is not None:
            image, mask = image[:, center_z], mask[center_z]
        if self.dimensions != "2.5d":
            image = normalize_image(image, self.normalization)
        spacing = self.case_spacings.get(pair.stem)
        if self.dimensions == "3d" and spacing is not None and self.target_spacing is not None:
            image, mask = resample_image_mask(image, mask, spacing, self.target_spacing)
        if self.dimensions != "2.5d":
            return pair, image, mask
        assert center_z is not None
        key = (pair.image, pair.region, pair.image.stat().st_mtime_ns, str(self.normalization))
        if key not in self._normalization_statistics:
            if len(self._normalization_statistics) >= 128:
                self._normalization_statistics.clear()
            self._normalization_statistics[key] = fit_normalization(image, self.normalization)
        statistics = self._normalization_statistics[key]
        radius = self.context_slices // 2
        stride = resolve_context_stride(
            self.context_stride_policy,
            fixed_stride=self.context_stride,
            target_spacing=self.context_target_spacing,
            z_spacing=(spacing or (1.0, 1.0, 1.0))[0],
        )
        channels: list[np.ndarray] = []
        for modality in range(image.shape[0]):
            for z in range(center_z - radius * stride, center_z + radius * stride + 1, stride):
                if 0 <= z < image.shape[1]:
                    channel_stats = {"type": statistics["type"], "channels": [statistics["channels"][modality]]}
                    channels.append(normalize_image(image[modality, z][None], statistics=channel_stats)[0])
                else:
                    channels.append(np.zeros(image.shape[2:], dtype=np.float32))
        return pair, np.ascontiguousarray(np.stack(channels)), np.ascontiguousarray(mask[center_z], dtype=np.int64)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor | dict[str, torch.Tensor]]:
        rng = np.random.default_rng(stable_seed(self.seed, self.epoch, str(index)))
        if self.training:
            if self.epoch_indices is None:
                self.set_epoch(self.epoch)
            assert self.epoch_indices is not None
            index = int(self.epoch_indices[index])
        source_items = self._source_items[self.items[index][0]]
        candidates = (
            [index, *rng.permutation(source_items)[: self.augmentation.empty_patch_max_retries].tolist()]
            if self.training
            else [index]
        )
        for item_index in candidates:
            pair, image, mask = self._load_item(item_index)
            if self.training and item_index != index and not np.any(mask > 0):
                continue
            object_diameter = self.instance_sizes.get(pair.stem, self.fallback_instance_size)
            try:
                image, mask, validity = apply_augmentation(
                    image,
                    mask,
                    self.augmentation,
                    rng=rng,
                    training=self.training,
                    object_diameter_px=object_diameter,
                    spacing=(
                        self.target_spacing
                        if self.dimensions == "3d"
                        else (
                            self.case_spacings.get(pair.stem, (1.0, 1.0, 1.0))[-2:]
                            if self.dimensions == "2.5d"
                            else None
                        )
                    ),
                    return_validity=True,
                )
                break
            except EmptyPatchError:
                continue
        else:
            pair, image, mask = self._load_item(index)
            fallback = replace(
                self.augmentation,
                foreground_oversampling=True,
                foreground_probability=1.0,
                affine_probability=0,
                elastic_probability=0,
            )
            try:
                image, mask, validity = apply_augmentation(
                    image,
                    mask,
                    fallback,
                    rng=rng,
                    training=self.training,
                    object_diameter_px=self.instance_sizes.get(pair.stem, self.fallback_instance_size),
                    return_validity=True,
                )
            except EmptyPatchError as exc:
                raise DatasetError("No foreground patch could be sampled in the assigned source domain") from exc
        spacing = self.target_spacing if self.dimensions == "3d" else None
        target = prepare_target(self.task, mask, label_values=self.label_values, spacing=spacing, validity=validity)
        image_t = torch.from_numpy(image)
        if isinstance(target, dict):
            return image_t, {key: torch.from_numpy(value) for key, value in target.items()}
        return image_t, torch.from_numpy(target)

    def provenance(self, index: int) -> dict:
        pair_index, z = self.items[index % len(self.items)]
        pair = self.pairs[pair_index]
        return {
            "source_image": str(pair.image.resolve()),
            "source_mask": str(pair.mask.resolve()),
            "region": pair.region or tuple((0, size) for size in pair.spatial_shape),
            "original_z_index": (z + (pair.region[0][0] if pair.region else 0)) if z is not None else None,
            "split_origin": pair.split_origin,
        }


def make_dataset(
    pairs: list[ImageMaskPair],
    task: str,
    label_values: list[int] | None,
    normalization: object | dict | None,
    profile: str,
    patch_size: tuple[int, ...],
    foreground_oversampling: bool,
    foreground_probability: float,
    augmentation_overrides: dict | None,
    training: bool,
    dimensions: str,
    seed: int,
    instance_sizes: dict[str, float] | None = None,
    fallback_instance_size: float | None = None,
    context_slices: int = 3,
    context_stride_policy: str = "adjacent",
    context_stride: int = 1,
    context_target_spacing: float | None = None,
    case_spacings: dict[str, tuple[float, float, float]] | None = None,
    target_spacing: tuple[float, float, float] | None = None,
    sample_count: int | None = None,
    max_empty_plane_fraction: float = 0.20,
) -> JdllSegmentationDataset:
    aug = make_augmentation_config(
        profile=profile,
        patch_size=patch_size,
        foreground_oversampling=foreground_oversampling,
        foreground_probability=foreground_probability,
        overrides=augmentation_overrides,
    )
    return JdllSegmentationDataset(
        pairs=pairs,
        task=task,
        label_values=label_values,
        normalization=normalization,
        augmentation=aug,
        training=training,
        dimensions=dimensions,
        seed=seed,
        instance_sizes=instance_sizes,
        fallback_instance_size=fallback_instance_size,
        context_slices=context_slices,
        context_stride_policy=context_stride_policy,
        context_stride=context_stride,
        context_target_spacing=context_target_spacing,
        case_spacings=case_spacings,
        target_spacing=target_spacing,
        sample_count=sample_count,
        max_empty_plane_fraction=max_empty_plane_fraction,
    )
