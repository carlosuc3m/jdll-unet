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
        self.items: list[tuple[int, int | None]] = []
        if dimensions == "2.5d":
            for pair_index, pair in enumerate(pairs):
                depth = load_mask(pair.mask, dimensions="2.5d").shape[0]
                self.items.extend((pair_index, z) for z in range(depth))
        else:
            self.items = [(index, None) for index in range(len(pairs))]

    def __len__(self) -> int:
        return len(self.items)

    def _load_item(self, item_index: int) -> tuple[ImageMaskPair, np.ndarray, np.ndarray]:
        pair_index, center_z = self.items[item_index]
        pair = self.pairs[pair_index]
        image = normalize_image(load_image(pair.image, dimensions=self.dimensions), self.normalization)
        mask = load_mask(pair.mask, dimensions=self.dimensions if self.dimensions in {"3d", "2.5d"} else "2d")
        if self.dimensions != "2.5d":
            return pair, image, mask
        assert center_z is not None
        radius = self.context_slices // 2
        channels: list[np.ndarray] = []
        for modality in range(image.shape[0]):
            for z in range(center_z - radius, center_z + radius + 1):
                channels.append(image[modality, z] if 0 <= z < image.shape[1] else np.zeros(image.shape[2:], dtype=image.dtype))
        return pair, np.ascontiguousarray(np.stack(channels)), mask[center_z]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor | dict[str, torch.Tensor]]:
        rng = np.random.default_rng(self.seed + index if not self.training else None)
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
                )
                break
            except EmptyPatchError:
                continue
        else:
            raise DatasetError("No foreground patch could be sampled from any training image")
        target = prepare_target(self.task, mask, label_values=self.label_values)
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
    )
