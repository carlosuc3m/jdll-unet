from pathlib import Path

import numpy as np
import tifffile

from jdll_unet.augment import EmptyPatchError, apply_augmentation, make_augmentation_config, sample_patch
from jdll_unet.dataset import make_dataset, partition_empty_pairs
from jdll_unet.errors import DatasetError
from jdll_unet.io import ImageMaskPair
from jdll_unet.scale import estimate_instance_size, resize_2d_pair
from jdll_unet.targets import binary_target, instance_targets, multiclass_target
from jdll_unet.task_detect import detect_task
from jdll_unet.trainer import train


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


def test_task_detection_3d_binary_multiclass_and_instance(tmp_path: Path):
    binary_root = tmp_path / "binary3d"
    binary = np.zeros((6, 16, 16), dtype=np.uint16)
    binary[:, 4:10, 4:10] = 1
    _write_dataset(binary_root, [binary, np.flip(binary, axis=1).copy()])
    assert detect_task({"dataset_path": binary_root})["task"] == "binary_semantic"

    multiclass_root = tmp_path / "multiclass3d"
    multiclass = np.zeros((6, 16, 16), dtype=np.uint16)
    multiclass[:, 2:7, 2:7] = 1
    multiclass[:, 9:14, 9:14] = 2
    _write_dataset(multiclass_root, [multiclass, multiclass.copy()])
    assert detect_task({"dataset_path": multiclass_root})["task"] == "multiclass_semantic"

    instance_root = tmp_path / "instance3d"
    instance_masks = []
    for offset in (0, 1):
        mask = np.zeros((6, 32, 32), dtype=np.uint16)
        label = 1
        for z in range(0, 6, 2):
            for y in range(3 + offset, 24, 10):
                for x in range(3, 24, 10):
                    mask[z : z + 2, y : y + 4, x : x + 4] = label
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

    volume = np.zeros((4, 5, 6), dtype=np.uint16)
    volume[1:3, 2:4, 2:5] = 1
    assert binary_target(volume).shape == (1, 4, 5, 6)
    assert multiclass_target(volume).shape == (4, 5, 6)
    instance_volume = instance_targets(volume)
    assert instance_volume["foreground"].shape == (1, 4, 5, 6)
    assert instance_volume["boundary"].shape == (1, 4, 5, 6)


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


def test_3d_augmentation_keeps_shape_and_types():
    rng = np.random.default_rng(11)
    image = rng.normal(size=(1, 8, 24, 24)).astype(np.float32)
    mask = np.zeros((8, 24, 24), dtype=np.int64)
    mask[2:6, 8:18, 9:20] = 1
    cfg = make_augmentation_config(
        "fast",
        patch_size=(6, 16, 16),
        foreground_oversampling=True,
        foreground_probability=1.0,
        overrides={"noise_probability": 1.0, "flip_probability": 1.0},
    )

    aug_image, aug_mask = apply_augmentation(image, mask, cfg, rng=rng, training=True)

    assert aug_image.shape == (1, 6, 16, 16)
    assert aug_mask.shape == (6, 16, 16)
    assert aug_image.dtype == np.float32
    assert np.issubdtype(aug_mask.dtype, np.integer)


def test_empty_patch_retry_policy():
    image = np.zeros((1, 16, 16), dtype=np.float32)
    mask = np.zeros((16, 16), dtype=np.int64)

    with np.testing.assert_raises(EmptyPatchError):
        sample_patch(
            image,
            mask,
            (8, 8),
            np.random.default_rng(1),
            skip_empty=True,
            max_retries=2,
            include_empty_after_max_retries=False,
        )

    _, accepted_mask = sample_patch(
        image,
        mask,
        (8, 8),
        np.random.default_rng(1),
        skip_empty=True,
        max_retries=2,
        include_empty_after_max_retries=True,
    )
    assert not np.any(accepted_mask)


def test_training_dataset_moves_to_next_image_after_empty_patch(tmp_path: Path):
    empty_mask = np.zeros((16, 16), dtype=np.uint16)
    positive_mask = np.ones((16, 16), dtype=np.uint16)
    _write_dataset(tmp_path, [empty_mask, positive_mask])
    pairs = [
        ImageMaskPair(tmp_path / "images" / f"sample_{index}.tif", tmp_path / "masks" / f"sample_{index}.tif", f"sample_{index}")
        for index in range(2)
    ]
    nonempty, empty = partition_empty_pairs(pairs)
    assert nonempty == [pairs[1]]
    assert empty == [pairs[0]]

    dataset = make_dataset(
        pairs,
        task="binary_semantic",
        label_values=[1],
        normalization=None,
        profile="fast",
        patch_size=(8, 8),
        foreground_oversampling=False,
        foreground_probability=0.0,
        augmentation_overrides={
            "skip_empty_patches": True,
            "empty_patch_max_retries": 1,
            "include_empty_patches_after_max_retries": False,
        },
        training=True,
        dimensions="2d",
        seed=3,
    )

    _, target = dataset[0]
    assert target.sum().item() == 64


def test_validation_keeps_empty_patch(tmp_path: Path):
    _write_dataset(tmp_path, [np.zeros((16, 16), dtype=np.uint16)])
    pair = ImageMaskPair(tmp_path / "images/sample_0.tif", tmp_path / "masks/sample_0.tif", "sample_0")
    dataset = make_dataset(
        [pair],
        task="binary_semantic",
        label_values=[1],
        normalization=None,
        profile="fast",
        patch_size=(8, 8),
        foreground_oversampling=False,
        foreground_probability=0.0,
        augmentation_overrides={},
        training=False,
        dimensions="2d",
        seed=3,
    )

    _, target = dataset[0]
    assert target.sum().item() == 0


def test_training_fails_clearly_when_all_masks_are_empty(tmp_path: Path):
    dataset_path = tmp_path / "dataset"
    _write_dataset(dataset_path, [np.zeros((16, 16), dtype=np.uint16)] * 2)

    with np.testing.assert_raises_regex(DatasetError, "All training masks are empty"):
        train(
            {
                "model_name": "all-empty",
                "output_dir": tmp_path / "model",
                "dataset_path": dataset_path,
                "epochs": 1,
                "patch_size": [8, 8],
            }
        )


def test_instance_size_estimation_and_label_safe_resize():
    mask = np.zeros((80, 80), dtype=np.int64)
    for index in range(25):
        y = 4 + (index // 5) * 14
        x = 4 + (index % 5) * 14
        mask[y : y + 6, x : x + 6] = index + 1
    mask[:5, 30:35] = 99
    estimate = estimate_instance_size(mask, max_instances=21, exclude_border=True, seed=7)

    assert estimate is not None
    assert estimate.available_instances == 25
    assert estimate.sampled_instances == 21
    assert np.isclose(estimate.median_diameter_px, 6.77, atol=0.01)

    image = mask[None].astype(np.float32)
    resized_image, resized_mask = resize_2d_pair(image, mask, 0.5)
    assert resized_image.shape == (1, 40, 40)
    assert resized_mask.shape == (40, 40)
    assert set(np.unique(resized_mask)).issubset(set(np.unique(mask)))


def test_instance_scale_crop_reaches_canonical_diameter():
    yy, xx = np.mgrid[:64, :64]
    mask = (((yy - 32) ** 2 + (xx - 32) ** 2) <= 5**2).astype(np.int64)
    image = mask[None].astype(np.float32)
    measured = estimate_instance_size(mask, exclude_border=True)
    assert measured is not None
    cfg = make_augmentation_config(
        "fast",
        patch_size=(32, 32),
        foreground_oversampling=True,
        foreground_probability=1.0,
        overrides={
            "instance_scale_enabled": True,
            "target_object_diameter_px": 16.0,
            "training_scale_jitter": (1.0, 1.0),
            "flip_probability": 0.0,
            "rotate90_probability": 0.0,
        },
    )

    _, scaled_mask = apply_augmentation(
        image,
        mask,
        cfg,
        rng=np.random.default_rng(5),
        training=True,
        object_diameter_px=measured.median_diameter_px,
    )
    scaled_diameter = np.sqrt(4 * np.count_nonzero(scaled_mask) / np.pi)
    assert np.isclose(scaled_diameter, 16.0, atol=1.5)
