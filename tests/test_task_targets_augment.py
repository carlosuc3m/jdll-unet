from pathlib import Path

import numpy as np
import tifffile

from jdll_unet.augment import apply_augmentation, make_augmentation_config
from jdll_unet.task_detect import detect_task
from jdll_unet.targets import binary_target, instance_targets, multiclass_target


def _write_dataset(root: Path, masks: list[np.ndarray]) -> None:
    images = root / "images"
    mask_dir = root / "masks"
    images.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    for index, mask in enumerate(masks):
        image = (mask > 0).astype(np.uint8) * 100
        tifffile.imwrite(images / f"sample_{index}.tif", image)
        tifffile.imwrite(mask_dir / f"sample_{index}.tif", mask.astype(np.uint16))


def test_task_detection_binary_multiclass_and_instance(tmp_path: Path):
    binary_root = tmp_path / "binary"
    _write_dataset(binary_root, [np.eye(16, dtype=np.uint8), np.fliplr(np.eye(16, dtype=np.uint8))])
    assert detect_task({"dataset_path": binary_root})["task"] == "binary_semantic"

    multiclass_root = tmp_path / "multiclass"
    masks = []
    for _ in range(2):
        mask = np.zeros((24, 24), dtype=np.uint16)
        mask[2:8, 2:8] = 1
        mask[12:20, 12:20] = 2
        masks.append(mask)
    _write_dataset(multiclass_root, masks)
    assert detect_task({"dataset_path": multiclass_root})["task"] == "multiclass_semantic"

    instance_root = tmp_path / "instance"
    instance_masks = []
    for offset in (0, 1):
        mask = np.zeros((64, 64), dtype=np.uint16)
        label = 1
        for y in range(4 + offset, 52, 12):
            for x in range(4, 52, 12):
                mask[y : y + 4, x : x + 4] = label
                label += 1
        instance_masks.append(mask)
    _write_dataset(instance_root, instance_masks)
    assert detect_task({"dataset_path": instance_root})["task"] == "instance_friendly"


def test_target_generation():
    mask = np.array([[0, 1, 1], [2, 2, 0], [0, 3, 0]], dtype=np.uint16)

    assert binary_target(mask).shape == (1, 3, 3)
    multiclass = multiclass_target(mask, label_values=[1, 2, 3])
    assert multiclass.dtype == np.int64
    assert set(np.unique(multiclass)) == {0, 1, 2, 3}
    instance = instance_targets(mask)
    assert sorted(instance) == ["boundary", "foreground"]
    assert instance["foreground"].shape == (1, 3, 3)
    assert instance["boundary"].shape == (1, 3, 3)


def test_augmentation_keeps_shape_and_types():
    rng = np.random.default_rng(10)
    image = rng.normal(size=(2, 32, 32)).astype(np.float32)
    mask = np.zeros((32, 32), dtype=np.int64)
    mask[8:18, 9:20] = 1
    cfg = make_augmentation_config(
        "balanced",
        patch_size=(24, 24),
        foreground_oversampling=True,
        foreground_probability=1.0,
        overrides={"affine_probability": 1.0, "noise_probability": 1.0},
    )

    aug_image, aug_mask = apply_augmentation(image, mask, cfg, rng=rng, training=True)

    assert aug_image.shape == (2, 24, 24)
    assert aug_mask.shape == (24, 24)
    assert aug_image.dtype == np.float32
    assert np.issubdtype(aug_mask.dtype, np.integer)
