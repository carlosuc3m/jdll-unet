"""PyTorch dataset wrappers for paired JDLL UNet data."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import torch
from torch.utils.data import Dataset

from .annotations import AnnotationPreparation
from .augment import AugmentationConfig, EmptyPatchError, apply_augmentation, make_augmentation_config
from .crop_reading import CropArray
from .crop_sampling import CropSampler
from .errors import DatasetError
from .geometry import domain_reader, eligible_centers, load_domain_image, load_domain_mask, split_sources, stable_seed
from .io import ImageMaskPair, fit_normalization, load_mask, normalize_image
from .label_statistics import MaskAnalysis, analyze_mask
from .planning import resample_image_mask, resample_mask, resolve_context_stride
from .spatial_augment import SpatialImagePlan, SpatialImageSample
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
        defer_photometric: bool = False,
        defer_spatial: bool = False,
        empty_patch_fraction: float = 0.0,
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
        self.defer_spatial = training and defer_spatial
        self.defer_photometric = defer_photometric or self.defer_spatial
        if not math.isfinite(empty_patch_fraction) or not 0 <= empty_patch_fraction < 1:
            raise ValueError("empty_patch_fraction must be finite and in [0, 1)")
        self.empty_patch_fraction = empty_patch_fraction if training else 0.0
        self.epoch_empty_flags: np.ndarray | None = None
        self._samplers: dict[int, CropSampler] = {}
        self._mask_analyses: dict[int, MaskAnalysis] = {}
        self.reader = domain_reader()
        if task == "instance_friendly":
            if self.reader.session.annotations is None:
                self.reader.session.annotations = AnnotationPreparation(emit=self.reader.session.emit)
            self.reader.session.annotations.prepare(pairs, dimensions, lambda: None, self.reader)
        self._normalization_statistics: dict[tuple[int, int | None], dict] = {}
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
                mask = load_domain_mask(pair, self.dimensions, self.reader, original=self.task != "instance_friendly")
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
        self._eligible_items = tuple(pool)
        self.epoch_indices = rng.choice(pool, size=len(self), replace=True)
        self.epoch_empty_flags = np.zeros(len(self), dtype=bool)
        quota_rng = np.random.default_rng(stable_seed(self.seed, epoch, "empty_patch_quota"))
        self.epoch_empty_flags[quota_rng.permutation(len(self))[:math.ceil(len(self) * self.empty_patch_fraction)]] = True
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
        return self.sample_count if self.sample_count is not None else len(self.items)

    def _normalization_stats(
        self,
        pair_index: int,
        center_z: int | None,
        image: np.ndarray,
    ) -> dict:
        key = (pair_index, center_z if self.dimensions == "2d" else None)
        statistics = self._normalization_statistics.get(key)
        if statistics is None:
            statistics = fit_normalization(image, self.normalization)
            self._normalization_statistics[key] = statistics
        return statistics

    def _load_item(
        self,
        item_index: int,
    ) -> tuple[ImageMaskPair, np.ndarray | CropArray, np.ndarray | CropArray, dict | None, bool]:
        pair_index, center_z = self.items[item_index % len(self.items)]
        pair = self.pairs[pair_index]
        if pair.image_axes is not None:
            key = (pair_index, center_z if self.dimensions == "2d" else None)
            if key not in self._normalization_statistics:
                image = load_domain_image(pair, dimensions=self.dimensions, reader=self.reader, raw=True)
                if self.dimensions == "2d" and center_z is not None:
                    image = image[:, center_z]
                self._normalization_stats(pair_index, center_z, image)
                del image
            statistics = self._normalization_statistics[key]
            shape = pair.domain_shape
            if self.dimensions == "3d" and pair.stem in self.case_spacings and self.target_spacing is not None:
                shape = tuple(np.maximum(1, np.rint(np.array(shape) * self.case_spacings[pair.stem] / np.array(self.target_spacing))).astype(int))
            stride = resolve_context_stride(
                self.context_stride_policy, fixed_stride=self.context_stride,
                target_spacing=self.context_target_spacing,
                z_spacing=self.case_spacings.get(pair.stem, (1, 1, 1))[0],
            ) if self.dimensions == "2.5d" else 1
            crop_image = CropArray(pair, self.reader, statistics=statistics, center_z=center_z,
                              context_slices=self.context_slices if self.dimensions == "2.5d" else 1,
                              context_stride=stride, resampled_shape=shape)
            crop_mask = CropArray(pair, self.reader, mask=True, original_mask=self.task != "instance_friendly",
                             center_z=center_z, resampled_shape=shape)
            return pair, crop_image, crop_mask, None, self.task == "instance_friendly"
        image = load_domain_image(pair, dimensions=self.dimensions, reader=self.reader, raw=True)
        mask = load_domain_mask(
            pair, dimensions=self.dimensions, reader=self.reader, raw=True, original=self.task != "instance_friendly"
        )
        source_has_instance_ids = self.task == "instance_friendly"
        spacing = self.case_spacings.get(pair.stem)

        if self.dimensions == "2d":
            if center_z is not None:
                image, mask = image[:, center_z], mask[center_z]
            statistics = self._normalization_stats(pair_index, center_z, image)
            return pair, image, mask, statistics, source_has_instance_ids

        statistics = self._normalization_stats(pair_index, None, image)

        if self.dimensions == "3d":
            # Legacy, uninspected array callers cannot map native TIFF regions.
            if spacing is not None and self.target_spacing is not None and not np.allclose(spacing, self.target_spacing):
                image = normalize_image(image, statistics=statistics)
                image, mask = resample_image_mask(image, mask, spacing, self.target_spacing)
                return pair, image, mask, None, source_has_instance_ids
            return pair, image, mask, statistics, source_has_instance_ids

        if self.dimensions != "2.5d":
            raise ValueError(f"Unsupported dataset dimensionality: {self.dimensions}")

        assert center_z is not None
        radius = self.context_slices // 2
        stride = resolve_context_stride(
            self.context_stride_policy,
            fixed_stride=self.context_stride,
            target_spacing=self.context_target_spacing,
            z_spacing=(spacing or (1.0, 1.0, 1.0))[0],
        )
        channels: list[np.ndarray] = []
        expanded_stats: list[tuple[float, float]] = []
        for modality in range(image.shape[0]):
            channel_stats = statistics["channels"][modality]
            offset = float(channel_stats[0])
            for z in range(center_z - radius * stride, center_z + radius * stride + 1, stride):
                if 0 <= z < image.shape[1]:
                    channels.append(image[modality, z])
                else:
                    # Missing context planes were normalized zeros in the original
                    # pipeline. Filling with the fitted offset reproduces that after
                    # deferred normalization.
                    channels.append(np.full(image.shape[2:], offset, dtype=np.float32))
                expanded_stats.append(channel_stats)
        context_statistics = {"type": statistics["type"], "channels": expanded_stats}
        return (
            pair,
            np.ascontiguousarray(np.stack(channels)),
            np.ascontiguousarray(mask[center_z], dtype=np.int64),
            context_statistics,
            source_has_instance_ids,
        )

    def validation_mask(self, item_index: int) -> np.ndarray | CropArray:
        """Mask-only access for planning; never decode an image or fit normalization."""
        pair_index, center_z = self.items[item_index]
        pair = self.pairs[pair_index]
        spacing = self.case_spacings.get(pair.stem)
        shape = pair.domain_shape
        if self.dimensions == "3d" and spacing is not None and self.target_spacing is not None:
            shape = tuple(int(n) for n in np.maximum(1, np.rint(np.array(shape) * spacing / np.array(self.target_spacing))))
        if pair.mask_axes is not None:
            return CropArray(pair, self.reader, mask=True, original_mask=self.task != "instance_friendly",
                             center_z=center_z, resampled_shape=shape)
        mask = load_domain_mask(pair, self.dimensions, self.reader, raw=True, original=self.task != "instance_friendly")
        if self.dimensions == "3d" and spacing is not None and self.target_spacing is not None:
            mask = resample_mask(mask, spacing, self.target_spacing)
        return mask[center_z] if center_z is not None else mask

    def mask_analysis(self, pair_index: int, mask: np.ndarray | None = None) -> MaskAnalysis:
        pair = self.pairs[pair_index]
        preparation = self.reader.session.annotations
        if preparation is not None:
            whole = tuple((0, size) for size in pair.spatial_shape)
            analysis_pair = replace(pair, region=()) if pair.region == whole else pair
            return preparation.statistics(analysis_pair, self.dimensions, self.reader, mask=mask,
                                          original=self.task != "instance_friendly")
        if pair_index not in self._mask_analyses:
            if mask is None:
                mask = load_domain_mask(pair, self.dimensions, self.reader, raw=True,
                                        original=self.task != "instance_friendly")
            self._mask_analyses[pair_index] = analyze_mask(mask, pair.mask)
        return self._mask_analyses[pair_index]

    def _sampler(self, item_index: int, mask: np.ndarray | CropArray) -> CropSampler:
        if item_index not in self._samplers:
            pair_index, center_z = self.items[item_index]
            pair = self.pairs[pair_index]
            analysis = self.mask_analysis(pair_index)
            self._samplers[item_index] = CropSampler(analysis, tuple(mask.shape),
                stable_seed(self.seed, 0, f"{pair.source_id}:{pair.stem}:{center_z}"),
                center_z=center_z, instances=self.task == "instance_friendly")
        return self._samplers[item_index]

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor | SpatialImageSample, torch.Tensor | dict[str, torch.Tensor]]:
        rng = np.random.default_rng(stable_seed(self.seed, self.epoch if self.training else 0, str(index)))
        empty = None
        if self.training:
            if self.epoch_indices is None:
                self.set_epoch(self.epoch)
            assert self.epoch_indices is not None
            if self.empty_patch_fraction:
                assert self.epoch_empty_flags is not None
                empty = bool(self.epoch_empty_flags[index])
            index = int(self.epoch_indices[index])
        else:
            index %= len(self.items)
        source_items = self._source_items[self.items[index][0]]
        candidates = (
            [index, *rng.permutation(source_items)[: self.augmentation.empty_patch_max_retries].tolist()]
            if self.training
            else [index]
        )
        if empty is not None:
            candidates.extend(int(i) for i in rng.permutation(self._eligible_items)[: self.augmentation.empty_patch_max_retries])
        loaded = {}
        cfg = replace(self.augmentation, skip_empty_patches=not empty) if empty is not None else self.augmentation
        require_background = empty is False and self.task == "instance_friendly"
        for item_index in candidates:
            if item_index not in loaded:
                loaded[item_index] = self._load_item(item_index)
            pair, image, mask, normalization_statistics, source_has_instance_ids = loaded[item_index]
            sampler = self._sampler(item_index, mask) if self.training or self.sample_count is not None else None
            if self.training and sampler is not None and empty is not True and cfg.skip_empty_patches and not sampler.has_foreground:
                continue
            object_diameter = self.instance_sizes.get(pair.stem, self.fallback_instance_size)
            image_plan = SpatialImagePlan() if self.defer_spatial else None
            try:
                image, mask, validity = apply_augmentation(
                    image,
                    mask,
                    cfg,
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
                    normalization_statistics=normalization_statistics,
                    defer_photometric=self.defer_photometric,
                    image_plan=image_plan,
                    return_validity=True,
                    sampler=sampler, empty=empty, require_background=require_background,
                )
                break
            except EmptyPatchError:
                continue
        else:
            pair, image, mask, normalization_statistics, source_has_instance_ids = loaded[index]
            fallback = replace(
                cfg,
                foreground_oversampling=True,
                foreground_probability=1.0,
                affine_probability=0,
                elastic_probability=0,
            )
            image_plan = SpatialImagePlan() if self.defer_spatial else None
            try:
                image, mask, validity = apply_augmentation(
                    image,
                    mask,
                    fallback,
                    rng=rng,
                    training=self.training,
                    object_diameter_px=self.instance_sizes.get(pair.stem, self.fallback_instance_size),
                    normalization_statistics=normalization_statistics,
                    defer_photometric=self.defer_photometric,
                    image_plan=image_plan,
                    return_validity=True,
                    sampler=self._sampler(index, mask) if self.training else None,
                    empty=empty, require_background=require_background,
                )
            except EmptyPatchError as exc:
                category = "empty" if empty else "foreground"
                raise DatasetError(f"No {category} patch could be sampled under the configured patch policy") from exc
        spacing = self.target_spacing if self.dimensions == "3d" else None
        target = prepare_target(
            self.task,
            mask,
            label_values=self.label_values,
            spacing=spacing,
            validity=validity,
            canonicalize_instances=not source_has_instance_ids,
            defer_dense=self.defer_spatial,
        )
        tensor_image = torch.from_numpy(image)
        image_t: torch.Tensor | SpatialImageSample = tensor_image
        if image_plan is not None:
            image_t = SpatialImageSample(tensor_image, image_plan)
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
    defer_photometric: bool = False,
    defer_spatial: bool = False,
    empty_patch_fraction: float = 0.0,
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
        defer_photometric=defer_photometric,
        defer_spatial=defer_spatial,
        empty_patch_fraction=empty_patch_fraction,
    )
