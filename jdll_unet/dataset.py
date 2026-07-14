"""PyTorch dataset wrappers for paired JDLL UNet data."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import AugmentationConfig, EmptyPatchError, apply_augmentation, make_augmentation_config
from .errors import DatasetError
from .io import ImageMaskPair, load_image, load_mask, normalize_image
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
    if len(pairs) < 2 or validation_fraction <= 0:
        return pairs, pairs[:1]
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * validation_fraction)))
    val = shuffled[:val_count]
    train = shuffled[val_count:] or shuffled[val_count - 1 :]
    return train, val


def inspect_dataset(pairs: list[ImageMaskPair], dimensions: str = "2d") -> DatasetInfo:
    if not pairs:
        raise ValueError("Cannot inspect an empty dataset")
    image = load_image(pairs[0].image, dimensions=dimensions)
    image_shape = tuple(int(item) for item in image.shape[1:])
    labels: set[int] = set()
    empty_mask_count = 0
    for pair in pairs:
        current_image = load_image(pair.image, dimensions=dimensions)
        mask = load_mask(pair.mask, dimensions=dimensions if dimensions in {"3d", "2.5d"} else "2d")
        if tuple(current_image.shape[1:]) != tuple(mask.shape):
            raise ValueError(
                f"Image/mask spatial shape mismatch for {pair.image.name}: "
                f"image={tuple(current_image.shape[1:])} mask={tuple(mask.shape)}"
            )
        labels.update(int(v) for v in np.unique(mask) if int(v) != 0)
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
        target = nonempty if np.any(load_mask(pair.mask, dimensions=mask_dimensions) != 0) else empty
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
        self.items: list[tuple[int, int | None]] = []
        if dimensions == "2.5d":
            for pair_index, pair in enumerate(pairs):
                depth = load_mask(pair.mask, dimensions="2.5d").shape[0]
                self.items.extend((pair_index, z) for z in range(depth))
        else:
            self.items = [(index, None) for index in range(len(pairs))]

    def __len__(self) -> int:
        return self.sample_count if self.training and self.sample_count is not None else len(self.items)

    def _load_item(self, item_index: int) -> tuple[ImageMaskPair, np.ndarray, np.ndarray]:
        pair_index, center_z = self.items[item_index % len(self.items)]
        pair = self.pairs[pair_index]
        image = normalize_image(load_image(pair.image, dimensions=self.dimensions), self.normalization)
        mask = load_mask(pair.mask, dimensions=self.dimensions if self.dimensions in {"3d", "2.5d"} else "2d")
        spacing = self.case_spacings.get(pair.stem)
        if self.dimensions == "3d" and spacing is not None and self.target_spacing is not None:
            image, mask = resample_image_mask(image, mask, spacing, self.target_spacing)
        if self.dimensions != "2.5d":
            return pair, image, mask
        assert center_z is not None
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
                channels.append(image[modality, z] if 0 <= z < image.shape[1] else np.zeros(image.shape[2:], dtype=image.dtype))
        return pair, np.ascontiguousarray(np.stack(channels)), mask[center_z]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor | dict[str, torch.Tensor]]:
        rng = np.random.default_rng(self.seed + index)
        if self.training:
            index = int(rng.integers(0, len(self.items)))
        candidates = range(len(self.items)) if self.training else range(1)
        for offset in candidates:
            pair, image, mask = self._load_item((index + offset) % len(self.items))
            object_diameter = self.instance_sizes.get(pair.stem, self.fallback_instance_size)
            try:
                image, mask = apply_augmentation(
                    image,
                    mask,
                    self.augmentation,
                    rng=rng,
                    training=self.training,
                    object_diameter_px=object_diameter,
                    spacing=(
                        self.target_spacing
                        if self.dimensions == "3d"
                        else (self.case_spacings.get(pair.stem, (1.0, 1.0, 1.0))[-2:] if self.dimensions == "2.5d" else None)
                    ),
                )
                break
            except EmptyPatchError:
                continue
        else:
            raise DatasetError("No foreground patch could be sampled from any training image")
        spacing = self.target_spacing if self.dimensions == "3d" else None
        target = prepare_target(self.task, mask, label_values=self.label_values, spacing=spacing)
        image_t = torch.from_numpy(image)
        if isinstance(target, dict):
            return image_t, {key: torch.from_numpy(value) for key, value in target.items()}
        return image_t, torch.from_numpy(target)


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
    )
