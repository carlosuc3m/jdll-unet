import hashlib
import os
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import tifffile

from jdll_unet.crop_reading import CropArray
from jdll_unet.errors import DataFormatError
from jdll_unet.geometry import DomainReader, inspect_pair, load_domain_image, load_domain_mask
from jdll_unet.image_reading import image_reading_session
from jdll_unet.io import ImageMaskPair, discover_dataset, fit_normalization, load_mask, normalize_image


@pytest.mark.parametrize("axes", ["YX", "XYC", "CYX", "XYCZ", "ZCYX", "XYCZT", "TZCYX"])
@pytest.mark.parametrize("mask_format", ["npy", "tif"])
def test_numpy_layouts_and_mixed_masks_use_crop_first_reads(tmp_path, axes, mask_format):
    volume = "Z" in axes
    canonical_axes = "CZYX" if volume else "CYX"
    shape = (2, 4, 128, 160) if volume else (2, 128, 160)
    image = np.random.default_rng(9).normal(2000, 100, shape).astype(np.float32)
    if axes == "YX":
        image = image[:1]
    labels = np.zeros(shape[1:], np.uint32)
    labels[..., 30:55, 45:70] = 70000
    labels[..., 80:100, 90:115] = 300
    for directory in ("images", "masks"):
        (tmp_path / directory).mkdir()
    image_path = tmp_path / "images/case.npy"
    mask_path = tmp_path / f"masks/case_mask.{mask_format}"
    array, order = image, canonical_axes
    if "T" in axes:
        array, order = array[None], "T" + order
    if "C" not in axes:
        array, order = array[0], order.replace("C", "")
    np.save(image_path, np.asfortranarray(array.transpose(tuple(order.index(axis) for axis in axes))))
    if mask_format == "npy":
        # Label volumes may omit the channel axis entirely.
        np.save(mask_path, labels.astype(">u4"))
    else:
        tifffile.imwrite(mask_path, labels, metadata={"axes": "ZYX" if volume else "YX"}, photometric="minisblack")
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (image_path, mask_path)}
    assert len(discover_dataset(tmp_path).train) == 1
    with image_reading_session() as session:
        reader = DomainReader(max_bytes=0, session=session)
        pair, _ = inspect_pair(ImageMaskPair(image_path, mask_path, "case"), reader=reader)
        np.testing.assert_array_equal(load_domain_image(pair, reader=reader, raw=True), image)
        np.testing.assert_array_equal(load_domain_mask(pair, "3d" if volume else "2d", reader), labels)
        statistics = fit_normalization(image)
        spatial = (slice(1, 3), slice(12, 40), slice(20, 58)) if volume else (slice(12, 40), slice(20, 58))
        session.pixels = Mock(side_effect=AssertionError("A crop must not decode the whole image"))
        crop = CropArray(pair, reader, statistics=statistics)
        np.testing.assert_array_equal(crop[(slice(None), *spatial)], normalize_image(image, statistics=statistics)[(slice(None), *spatial)])
        np.testing.assert_array_equal(CropArray(pair, reader, mask=True)[spatial], labels[spatial])
        if volume:
            context = CropArray(pair, reader, statistics=statistics, center_z=1, context_slices=3)
            result = context[(slice(None), *spatial[-2:])]
            expected = normalize_image(image, statistics=statistics)[:, :3, 12:40, 20:58].reshape(6, 28, 38)
            np.testing.assert_array_equal(result, expected)
        assert not reader.cache and reader.bytes == 0
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == value for p, value in hashes.items())
    if os.environ.get("JDLL_NUMPY_SMOKE") == "1" and axes == "CYX" and mask_format == "npy":
        import torch

        from jdll_unet.trainer import train

        torch.set_num_threads(2)
        output = tmp_path / "trained"
        result = train({"model_name": "numpy", "dataset_path": str(tmp_path), "output_dir": str(output),
                        "architecture": "tiny-2d", "task": "instance_friendly", "device": "cpu", "epochs": 1,
                        "steps_per_epoch": 1, "patch_size": [32, 32], "batch_size": 1, "effective_batch_size": 1,
                        "preview_count": 0, "validation": {"mode": "light"}})
        assert Path(result["model_path"]).is_file()
        assert all(hashlib.sha256(p.read_bytes()).hexdigest() == value for p, value in hashes.items())


def test_numpy_mask_validation_and_unsupported_shapes_name_the_source(tmp_path):
    path = tmp_path / "invalid.npy"
    for i, value in enumerate((np.nan, np.inf, -1, 0.5)):
        path = tmp_path / f"invalid-{i}.npy"
        np.save(path, np.full((128, 160), value, np.float32))
        with pytest.raises(DataFormatError, match="invalid-.*npy"):
            load_mask(path, "2d")
    for i, array in enumerate((np.empty((1, 1, 1, 1, 1, 1)), np.array([[object()]], object))):
        path = tmp_path / f"invalid-rank-{i}.npy"
        np.save(path, array)
        with pytest.raises(DataFormatError, match="invalid-rank-.*npy"):
            load_mask(path, "2d")
    path = tmp_path / "invalid-time.npy"
    np.save(path, np.zeros((3, 4, 1, 128, 160), np.uint16))
    with pytest.raises(DataFormatError, match="time points"):
        load_mask(path, "3d")


def test_raster_images_take_precedence_over_numpy_caches_but_masks_remain_checked(tmp_path):
    for directory in ("images", "masks"):
        (tmp_path / directory).mkdir()
    image = np.ones((128, 160), np.uint16)
    tifffile.imwrite(tmp_path / "images/case.tif", image)
    # A broken legacy cache must not even be decoded.
    (tmp_path / "images/case.npy").write_bytes(b"stale cache")
    np.save(tmp_path / "images/other.npy", image)
    for name in ("case", "other"):
        tifffile.imwrite(tmp_path / f"masks/{name}_mask.tif", image)
    pairs = discover_dataset(tmp_path).train
    assert [pair.image.name for pair in pairs] == ["case.tif", "other.npy"]
    np.save(tmp_path / "masks/case_mask.npy", image)
    from jdll_unet.errors import DatasetError

    with pytest.raises(DatasetError, match="Ambiguous mask"):
        discover_dataset(tmp_path)


@pytest.mark.parametrize("volume", [False, True])
@pytest.mark.parametrize("layout", ["contiguous", "fortran", "strided", "reversed", "big_endian"])
def test_annotation_source_ids_match_for_readonly_layouts(volume, layout):
    from skimage.measure import label

    from jdll_unet.annotations import component_labels_and_sources

    mask = np.zeros((3, 24, 28) if volume else (24, 28), np.uint32)
    mask[..., 2:8, 4:12] = 70000
    mask[..., 14:18, 18:22] = 300
    if layout == "fortran":
        mask = np.asfortranarray(mask)
    elif layout == "strided":
        mask = np.repeat(mask[..., None], 3, axis=-1)[..., 0]
    elif layout == "reversed":
        mask = mask[..., ::-1]
    elif layout == "big_endian":
        mask = mask.astype(">u4")
    mask.flags.writeable = False
    original = mask.copy()
    expected, count = label(mask, background=0, connectivity=1, return_num=True)
    source_ids = np.zeros(count + 1, np.int64)
    source_ids[expected.ravel()] = mask.ravel()
    actual, actual_ids = component_labels_and_sources(mask)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_ids, source_ids)
    np.testing.assert_array_equal(mask, original)
