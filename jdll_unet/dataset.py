"""PyTorch dataset wrappers for paired JDLL UNet data."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import AugmentationConfig, apply_augmentation, make_augmentation_config
from .io import ImageMaskPair, load_image, load_mask, normalize_image
from .targets import prepare_target


@dataclass(slots=True)
class DatasetInfo:
    input_channels: int
    image_shape: tuple[int, ...]
    label_values: list[int]


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
    for pair in pairs:
        current_image = load_image(pair.image, dimensions=dimensions)
        mask = load_mask(pair.mask, dimensions="3d" if dimensions == "3d" else "2d")
        if tuple(current_image.shape[1:]) != tuple(mask.shape):
            raise ValueError(
                f"Image/mask spatial shape mismatch for {pair.image.name}: "
                f"image={tuple(current_image.shape[1:])} mask={tuple(mask.shape)}"
            )
        labels.update(int(v) for v in np.unique(mask) if int(v) != 0)
    return DatasetInfo(
        input_channels=int(image.shape[0]),
        image_shape=image_shape,
        label_values=sorted(labels),
    )


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
    ) -> None:
        self.pairs = pairs
        self.task = task
        self.label_values = label_values
        self.normalization = normalization
        self.augmentation = augmentation
        self.training = training
        self.dimensions = dimensions
        self.seed = seed

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor | dict[str, torch.Tensor]]:
        pair = self.pairs[index]
        image = normalize_image(load_image(pair.image, dimensions=self.dimensions), self.normalization)
        mask = load_mask(pair.mask, dimensions="3d" if self.dimensions == "3d" else "2d")
        rng = np.random.default_rng(self.seed + index if not self.training else None)
        image, mask = apply_augmentation(image, mask, self.augmentation, rng=rng, training=self.training)
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
    )
