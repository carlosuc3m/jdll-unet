from pathlib import Path

import numpy as np
import pytest
import tifffile

from jdll_unet.io import discover_dataset, load_mask


def test_dataset_pairing_accepts_aliases_and_suffixes(tmp_path: Path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "labels"
    image_dir.mkdir()
    mask_dir.mkdir()
    tifffile.imwrite(image_dir / "image001_image.tif", np.zeros((8, 8), dtype=np.uint16))
    tifffile.imwrite(mask_dir / "image001_label.tif", np.ones((8, 8), dtype=np.uint16))

    splits = discover_dataset(tmp_path)

    assert len(splits.train) == 1
    assert splits.train[0].image.name == "image001_image.tif"
    assert splits.train[0].mask.name == "image001_label.tif"
    assert not splits.explicit_val


def test_explicit_train_val_layout(tmp_path: Path):
    for split in ("train", "val"):
        image_dir = tmp_path / split / "imgs"
        mask_dir = tmp_path / split / "gt"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        tifffile.imwrite(image_dir / f"{split}.tif", np.zeros((8, 8), dtype=np.uint8))
        tifffile.imwrite(mask_dir / f"{split}_mask.tif", np.ones((8, 8), dtype=np.uint8))

    splits = discover_dataset(tmp_path)

    assert len(splits.train) == 1
    assert len(splits.val) == 1
    assert splits.explicit_val


def test_rgb_mask_collapses_only_duplicate_channels(tmp_path: Path):
    duplicate = np.zeros((8, 8, 3), dtype=np.uint8)
    duplicate[..., :] = 4
    duplicate_path = tmp_path / "duplicate.tif"
    tifffile.imwrite(duplicate_path, duplicate)

    assert np.array_equal(load_mask(duplicate_path), np.full((8, 8), 4, dtype=np.int64))

    bad = duplicate.copy()
    bad[..., 1] = 3
    bad_path = tmp_path / "bad.tif"
    tifffile.imwrite(bad_path, bad)

    with pytest.raises(ValueError, match="RGB channels"):
        load_mask(bad_path)
