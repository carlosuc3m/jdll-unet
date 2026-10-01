from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import tifffile

from jdll_unet import annotations, geometry, label_statistics, scale
from jdll_unet.annotations import AnnotationPreparation
from jdll_unet.config import AnnotationPreparationConfig, parse_training_config
from jdll_unet.errors import DataFormatError
from jdll_unet.image_reading import image_reading_session
from jdll_unet.io import ImageMaskPair, validate_mask_labels
from jdll_unet.label_statistics import analyze_mask
from jdll_unet.task_detect import detect_task_from_pairs
from jdll_unet.training_geometry import measure_case_instances


@pytest.mark.parametrize("shape", [(13, 17), (5, 13, 17)])
@pytest.mark.parametrize("dtype", [np.uint8, np.int64, np.float64])
@pytest.mark.parametrize("sparse", [False, True])
def test_tiled_statistics_match_full_resolution_reference(shape, dtype, sparse, monkeypatch):
    monkeypatch.setattr(label_statistics, "STATISTICS_CHUNK_PIXELS", 11)
    values = np.array([0, 7, 23, 127], dtype=dtype)
    if sparse and dtype != np.uint8:
        values[-1] = 2**40
    rng = np.random.default_rng(4)
    mask = values[rng.integers(0, len(values), size=shape)].T[::-1]
    analysis = analyze_mask(mask)
    assert analysis.shape == mask.shape
    assert analysis.labels == tuple(int(v) for v in np.unique(mask) if v)
    for label, region in analysis.objects.items():
        coords = np.argwhere(mask == label)
        assert region.count == len(coords)
        assert region.bounds == tuple(zip(coords.min(axis=0), coords.max(axis=0) + 1, strict=True))
    expected = tuple(int(v) for v in np.count_nonzero(mask, axis=(-2, -1)).reshape(-1))
    assert analysis.plane_positive_counts == expected
    for z, plane in enumerate(mask[None] if mask.ndim == 2 else mask):
        for label, region in analysis.planes[z].items():
            assert region.count == np.count_nonzero(plane == label)


@pytest.mark.parametrize("value,dtype", [(-1, np.int64), (2**63, np.uint64), (2**63, np.float64), (0.5, np.float32), (np.nan, np.float32), (np.inf, np.float32)])
def test_invalid_values_are_still_rejected(value, dtype):
    mask = np.zeros((4, 5), dtype=dtype)
    mask[-1, -1] = value
    with pytest.raises(DataFormatError):
        analyze_mask(mask)


def test_integer_extremes_and_empty_masks():
    for dtype in (np.int64, np.uint64):
        mask = np.array([[0, np.iinfo(np.int64).max]], dtype=dtype)
        validate_mask_labels(mask, Path("test"))
        assert analyze_mask(mask).objects[np.iinfo(np.int64).max].count == 1
    empty = analyze_mask(np.zeros((3, 4, 5), np.uint8))
    assert empty.labels == ()
    assert empty.plane_positive_counts == (0, 0, 0)
    assert scale.estimate_from_analysis(empty) is None


def object_grid(ndim):
    mask = np.zeros((80, 80) if ndim == 2 else (8, 80, 80), dtype=np.int64)
    for i in range(36):
        y, x, size = 2 + (i // 6) * 12, 2 + (i % 6) * 12, 2 + i % 6
        region = (slice(y, y + size), slice(x, x + size))
        mask[region if ndim == 2 else (slice(2, 6), *region)] = 7 + i * 100000
    return mask


@pytest.mark.parametrize("dimensions", ["2d", "2.5d", "3d"])
def test_equivalent_measure_uses_only_sampled_counts(dimensions, monkeypatch):
    mask = object_grid(2 if dimensions == "2d" else 3)
    analysis = analyze_mask(mask)
    monkeypatch.setattr(scale, "_principal_axes", Mock(side_effect=AssertionError("Unrequested axes")))
    rng = np.random.default_rng(9)
    selected = [analysis.labels[int(i)] for i in rng.choice(36, 21, replace=False)]
    values = []
    for label in selected:
        count = analysis.objects[label].count
        if dimensions == "3d":
            values.append((6 * count * 2 * 0.5 * 0.5 / np.pi) ** (1 / 3))
        else:
            values.append(np.sqrt(4 * count / (4 if dimensions == "2.5d" else 1) / np.pi))
    estimate = scale.estimate_from_analysis(analysis, seed=9, volume_xy=dimensions == "2.5d", spacing=(2, 0.5, 0.5))
    assert estimate.sampled_instances == 21
    assert estimate.available_instances == 36
    assert estimate.median_diameter_px == pytest.approx(np.median(values))
    assert estimate.median_principal_axes is None


@pytest.mark.parametrize("dimensions", ["2d", "2.5d", "3d"])
def test_principal_axes_only_measure_randomly_selected_objects(dimensions, monkeypatch):
    mask = object_grid(2 if dimensions == "2d" else 3)
    analysis = analyze_mask(mask)
    checker = Mock(wraps=scale._principal_axes)
    monkeypatch.setattr(scale, "_principal_axes", checker)
    kwargs = {"mask": mask, "volume_xy": dimensions == "2.5d", "max_instances": 7, "seed": 42,
              "measure": "principal_axes", "spacing": (3, 2, 1)}
    first = scale.estimate_from_analysis(analysis, **kwargs)
    assert checker.call_count == 7
    selected = [call.args[1] for call in checker.call_args_list]
    assert selected != list(analysis.labels[:7])
    reference = []
    for call in checker.call_args_list:
        source, label, _, spacing = call.args
        coords = np.argwhere(source == label).astype(float) * np.asarray(spacing)
        axes = 2 * np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov(coords.T)), 0))
        reference.append(np.median(axes))
    assert first.median_diameter_px == pytest.approx(np.median(reference))
    assert first == scale.estimate_from_analysis(analysis, **kwargs)


def test_25d_uses_center_sections_and_largest_z_border_sections():
    mask = np.zeros((9, 30, 30), np.int64)
    mask[2:7, 3:7, 3:7] = 7
    mask[2, 3:10, 3:10] = 7  # Larger, but not the central slice.
    mask[:3, 15:17, 15:17] = 11
    mask[1, 15:21, 15:21] = 11  # A Z-truncated instance uses its largest section.
    analysis = analyze_mask(mask)
    first = scale.estimate_from_analysis(analysis, volume_xy=True, max_instances=1)
    assert first.median_diameter_px == pytest.approx(np.sqrt(4 * 16 / np.pi))
    both = scale.estimate_from_analysis(analysis, volume_xy=True, max_instances=21)
    assert both.median_diameter_px == pytest.approx(np.median(np.sqrt(4 * np.array([16, 36]) / np.pi)))


def test_3d_border_preference_and_single_voxel_axes():
    mask = np.zeros((8, 8, 8), np.int64)
    mask[:2, :2, :2] = 1
    mask[3, 3, 3] = 7
    analysis = analyze_mask(mask)
    first = scale.estimate_from_analysis(analysis, max_instances=1, min_size=1)
    assert first.median_diameter_px == pytest.approx((6 / np.pi) ** (1 / 3))
    estimate = scale.estimate_from_analysis(analysis, mask=mask, max_instances=1, min_size=1, measure="principal_axes")
    assert estimate.median_principal_axes == (0, 0, 0)
    assert scale.estimate_from_analysis(analysis, min_size=1).sampled_instances == 2
    # Disabling preference makes the sample uniform across both identities.
    selected = np.random.default_rng(1).choice(2, 1, replace=False)[0]
    unrestricted = scale.estimate_from_analysis(analysis, max_instances=1, min_size=1, exclude_border=False, seed=1)
    assert unrestricted.median_diameter_px == pytest.approx((6 * (8 if selected == 0 else 1) / np.pi) ** (1 / 3))


def test_moments_merge_across_coordinate_chunks():
    mask = np.ones((65, 128, 128), dtype=np.uint8)
    region = analyze_mask(mask).objects[1]
    spacing = (2.0, 0.5, 1.0)
    axes = scale._principal_axes(mask, 1, region, spacing)
    # Population variance of 0..n-1, converted to the sample covariance convention.
    variance = (np.asarray(mask.shape, dtype=float) ** 2 - 1) / 12
    variance *= mask.size / (mask.size - 1) * np.asarray(spacing) ** 2
    np.testing.assert_allclose(axes, np.sort(2 * np.sqrt(variance)), rtol=1e-12)


@pytest.mark.parametrize("estimator", [scale.estimate_volume_instance_size, scale.estimate_3d_instance_size])
def test_volume_estimators_reject_planes_without_canonicalization(estimator):
    kwargs = {"spacing": (1, 1, 1)} if estimator is scale.estimate_3d_instance_size else {}
    with pytest.raises(ValueError, match="Z,Y,X"):
        estimator(np.zeros((4, 5), np.uint8), canonicalize_instances=False, **kwargs)


@pytest.mark.parametrize("dimensions", ["2d", "2.5d", "3d"])
@pytest.mark.parametrize("repair", [False, True])
def test_source_analysis_is_reused_after_inspection_and_preparation(tmp_path, monkeypatch, dimensions, repair):
    mask = object_grid(2 if dimensions == "2d" else 3)
    if repair:
        mask[mask == 100007] = 7
    axes = "YX" if mask.ndim == 2 else "ZYX"
    image, labels = tmp_path / "image.tif", tmp_path / "mask.tif"
    tifffile.imwrite(image, np.zeros(mask.shape, np.uint16), metadata={"axes": axes})
    tifffile.imwrite(labels, mask, metadata={"axes": axes})
    cfg = parse_training_config({"model_name": "reuse", "dataset_path": tmp_path, "output_dir": tmp_path})
    with image_reading_session() as session:
        session.annotations = prep = AnnotationPreparation(AnnotationPreparationConfig(ram_cache_mb=0), cache_dir=tmp_path)
        checker = Mock(wraps=annotations.analyze_mask)
        monkeypatch.setattr(annotations, "analyze_mask", checker)
        pair, _ = geometry.inspect_pair(ImageMaskPair(image, labels, "case"))
        assert checker.call_count == 1
        detect_task_from_pairs([pair], dimensions=dimensions, preparation=prep)
        prep.prepare([pair], dimensions, lambda: None)
        assert checker.call_count == (2 if repair else 1)
        with monkeypatch.context() as patch:
            patch.setattr(geometry, "load_domain_mask", Mock(side_effect=AssertionError("Repeated mask read")))
            patch.setattr(annotations, "analyze_mask", Mock(side_effect=AssertionError("Repeated mask analysis")))
            measured, _ = measure_case_instances(pair, cfg, dimensions, (1, 1, 1))
        assert measured.available_instances == 36
        assert measured.sampled_instances == 21
        # Domain summaries are separate; full-source borders/counts must not leak into a crop.
        bounds = tuple((0, size) for size in mask.shape[:-1]) + ((0, 40),)
        domain = replace(pair, region=bounds)
        summary = prep.statistics(domain, dimensions)
        assert summary.shape[-1] == 40
        assert summary is prep.statistics(domain, dimensions)
        old_source = prep.source_analysis(pair, dimensions)
        tifffile.imwrite(labels, np.zeros_like(mask), metadata={"axes": axes})
        assert prep.statistics(pair, dimensions).labels == ()
        assert prep.source_analysis(pair, dimensions) is not old_source


def test_atomic_replacement_invalidates_and_releases_old_summary(tmp_path):
    image, labels = tmp_path / "image.tif", tmp_path / "mask.tif"
    mask = np.ones((8, 8), np.uint8)
    tifffile.imwrite(image, mask)
    tifffile.imwrite(labels, mask)
    with image_reading_session() as session:
        session.annotations = prep = AnnotationPreparation()
        pair, _ = geometry.inspect_pair(ImageMaskPair(image, labels, "case"))
        assert prep.statistics(pair, "2d").labels == (1,)
        replacement = tmp_path / "replacement.tif"
        tifffile.imwrite(replacement, mask * 7)
        replacement.replace(labels)
        assert prep.statistics(pair, "2d").labels == (7,)
        assert len(prep.sources) == 1
        assert len(prep._label_sets) == 1
