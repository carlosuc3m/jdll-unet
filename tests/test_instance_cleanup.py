from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import tifffile

from jdll_unet import annotations
from jdll_unet.annotations import AnnotationPreparation
from jdll_unet.config import AnnotationPreparationConfig, InstanceScaleNormalizationConfig, parse_training_config
from jdll_unet.errors import ConfigError
from jdll_unet.geometry import inspect_pair, load_domain_mask, stable_seed
from jdll_unet.image_reading import image_reading_session
from jdll_unet.io import ImageMaskPair
from jdll_unet.label_statistics import analyze_mask
from jdll_unet.scale import aggregate_instance_statistics, estimate_from_analysis


def write_pair(root, mask, name="case"):
    axes = "YX" if mask.ndim == 2 else "ZYX"
    image, labels = root / f"{name}.tif", root / f"{name}_mask.tif"
    tifffile.imwrite(image, np.zeros(mask.shape, np.uint8), metadata={"axes": axes})
    tifffile.imwrite(labels, mask, metadata={"axes": axes})
    return inspect_pair(ImageMaskPair(image, labels, name))[0]


def example_mask(ndim):
    mask = np.zeros((32,) * ndim, np.uint16)
    for label, lo in ((7, (4,) * ndim), (9, (16, 4) if ndim == 2 else (4, 16, 4)),
                      (11, (4, 16) if ndim == 2 else (4, 4, 16))):
        mask[tuple(slice(start, start + 8) for start in lo)] = label
    mask[(1,) * ndim] = 7  # First scanned component is NOT the largest.
    mask[(30,) * ndim] = 33  # A standalone tiny ID must also be removable.
    mask[tuple(slice(23, 27) for _ in range(ndim))] = 7  # Valid duplicate ID.
    return mask


@pytest.mark.parametrize("dimensions", ["2d", "2.5d", "3d"])
@pytest.mark.parametrize("ram_mb", [0, 1])
def test_shared_reference_and_one_connectivity_pass(tmp_path, monkeypatch, dimensions, ram_mb):
    mask = example_mask(2 if dimensions == "2d" else 3)
    pair = write_pair(tmp_path, mask)
    before = pair.mask.read_bytes()
    checker = Mock(wraps=annotations.label_components)
    monkeypatch.setattr(annotations, "label_components", checker)
    references = []
    for cleanup in (False, True):
        with image_reading_session() as session:
            prep = AnnotationPreparation(
                AnnotationPreparationConfig(cleanup=cleanup, ram_cache_mb=ram_mb), seed=42,
                cache_dir=tmp_path / "cache",
            )
            session.annotations = prep
            prep.prepare([pair], dimensions, lambda: None)
            record = prep.analyze(pair, dimensions)
            estimate = prep.reference_size(pair, dimensions, (3, 0.5, 0.5), prep.scale_config, 42)
            references.append(estimate)
            result = load_domain_mask(pair, dimensions, raw=True)
            assert record.reference_diameter > 0
            assert estimate.available_instances == 3
            if cleanup:
                expected = mask.copy()
                expected[(1,) * mask.ndim] = 0
                expected[(30,) * mask.ndim] = 0
                expected[tuple(slice(23, 27) for _ in range(mask.ndim))] = 34
                np.testing.assert_array_equal(result, expected)
                assert record.removed_components == 2
                assert record.relabelled_components == 1
                measured = prep.statistics(pair, dimensions)
                exact = analyze_mask(result)
                assert measured.objects == exact.objects
                assert measured.planes == exact.planes
                assert measured.plane_positive_counts == exact.plane_positive_counts
                assert np.all(result.flat[measured.foreground_indices] != 0)
                assert len(measured.foreground_indices) == np.count_nonzero(result)
                assert record.storage == ("ram" if ram_mb else "disk")
            else:
                np.testing.assert_array_equal(result, mask)
                assert record.labels is None
                assert record.removed_components == 0
                assert not record.repair_parents
            for _ in range(3):
                prep.statistics(pair, dimensions)
                prep.reference_size(pair, dimensions, (3, 0.5, 0.5), prep.scale_config, 42)
    assert checker.call_count == 2  # One per run, not per consumer.
    assert references[0] == references[1]
    assert pair.mask.read_bytes() == before
    assert not list((tmp_path / "cache").rglob("*.npy"))


@pytest.mark.parametrize("foreground", [1, 255])
def test_binary_and_id_conventions_are_dataset_scoped(tmp_path, foreground):
    binary = np.zeros((32, 32), np.uint8)
    binary[3:7, 3:7] = binary[14:22, 14:22] = foreground
    first = write_pair(tmp_path, binary, "single_id")
    second = write_pair(tmp_path, (binary > 0).astype(np.uint8) * 7, "other_ids")
    for pairs, expected in (([first], 2), ([first, second], 1)):
        with image_reading_session() as session:
            prep = AnnotationPreparation(AnnotationPreparationConfig(cleanup=False))
            session.annotations = prep
            prep.prepare(pairs, "2d", lambda: None)
            estimate = prep.reference_size(first, "2d", (1, 1, 1), prep.scale_config, 0)
            assert estimate.available_instances == expected
            if expected == 1:
                assert estimate.median_diameter_px == pytest.approx(np.sqrt(4 * 64 / np.pi))
            np.testing.assert_array_equal(load_domain_mask(first, raw=True), binary)


def test_25d_removes_whole_thin_object_not_individual_sections(tmp_path):
    mask = example_mask(3)
    mask[15:30, 15, 15] = 40  # Volume 15, but representative XY area is only 1.
    mask[3, 5, 5] = 7  # A tiny section connected to a large object must stay.
    pair = write_pair(tmp_path, mask)
    for dimensions in ("2.5d", "3d"):
        with image_reading_session() as session:
            prep = AnnotationPreparation()
            session.annotations = prep
            prep.prepare([pair], dimensions, lambda: None)
            result = load_domain_mask(pair, dimensions, raw=True)
            assert result[3, 5, 5] == 7
            assert bool(np.any(result == 40)) == (dimensions == "3d")


def test_3d_uses_diameter_ratio_cubed_and_physical_sizing(tmp_path):
    pair = write_pair(tmp_path, example_mask(3))
    with image_reading_session() as session:
        prep = AnnotationPreparation()
        session.annotations = prep
        prep.prepare([pair], "3d", lambda: None)
        record = prep.analyze(pair, "3d")
        assert record.minimum_component_size == pytest.approx(512 / 343)
        isotropic = prep.reference_size(pair, "3d", (1, 1, 1), prep.scale_config, 0)
        anisotropic = prep.reference_size(pair, "3d", (8, 1, 1), prep.scale_config, 0)
        assert anisotropic.median_diameter_px == pytest.approx(2 * isotropic.median_diameter_px)


def test_cleanup_disabled_preserves_every_original_label_in_training_targets(tmp_path, monkeypatch):
    from jdll_unet import targets
    from jdll_unet.augment import AugmentationConfig
    from jdll_unet.dataset import JdllSegmentationDataset

    mask = example_mask(2)
    pair = write_pair(tmp_path, mask)
    monkeypatch.setattr(targets, "canonical_instance_labels", Mock(side_effect=AssertionError("Hidden repair")))
    with image_reading_session() as session:
        prep = AnnotationPreparation(AnnotationPreparationConfig(cleanup=False))
        session.annotations = prep
        prep.prepare([pair], "2d", lambda: None)
        dataset = JdllSegmentationDataset(
            [pair], "instance_friendly", None, {"type": "none"},
            AugmentationConfig(patch_size=(32, 32)), False,
        )
        _, target = dataset[0]
        instances = target["instances"].numpy()[0]
        assert instances[1, 1] == instances[5, 5] == instances[24, 24] != 0
        assert instances[30, 30] != 0


def test_storage_failure_preserves_originals_but_not_fragment_biased_diameter(tmp_path, monkeypatch):
    mask = example_mask(2)
    pair = write_pair(tmp_path, mask)
    monkeypatch.setattr(annotations.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with image_reading_session() as session:
        prep = AnnotationPreparation(AnnotationPreparationConfig(ram_cache_mb=0), cache_dir=tmp_path / "cache")
        session.annotations = prep
        prep.prepare([pair], "2d", lambda: None)
        np.testing.assert_array_equal(load_domain_mask(pair, raw=True), mask)
        estimate = prep.reference_size(pair, "2d", (1, 1, 1), prep.scale_config, 0)
        assert estimate.median_diameter_px == pytest.approx(np.sqrt(4 * 64 / np.pi))
        summary = prep.diagnostics()
        assert summary["removed_components"] == summary["repaired_components"] == 0
        assert summary["sources"][0]["skipped_reason"] == "insufficient_cache_space"


def test_no_reference_does_not_delete_annotations(tmp_path):
    mask = np.zeros((8, 8), np.uint8)
    mask[1, 1] = mask[3, 3] = 7
    pair = write_pair(tmp_path, mask)
    events = []
    with image_reading_session() as session:
        prep = AnnotationPreparation(emit=lambda event, **kw: events.append((event, kw)))
        session.annotations = prep
        prep.prepare([pair], "2d", lambda: None)
        prep.prepare([pair], "2d", lambda: None)
        assert np.count_nonzero(load_domain_mask(pair)) == 2
        assert prep.diagnostics()["removed_components"] == 0
        assert len([kw for _, kw in events if kw["reason"] == "annotation_cleanup_no_reference"]) == 1


@pytest.mark.parametrize("dimensions", ["2d", "2.5d", "3d"])
def test_principal_axes_reuse_unmodified_component_moments(tmp_path, dimensions):
    pair = write_pair(tmp_path, example_mask(2 if dimensions == "2d" else 3))
    estimates = []
    for cleanup in (True, False):
        with image_reading_session() as session:
            options = InstanceScaleNormalizationConfig(object_size_measure="principal_axes")
            prep = AnnotationPreparation(AnnotationPreparationConfig(cleanup=cleanup), scale_config=options)
            session.annotations = prep
            prep.prepare([pair], dimensions, lambda: None)
            record = prep.analyze(pair, dimensions)
            spacing = (3, 0.5, 2)
            estimates.append(prep.reference_size(pair, dimensions, spacing, options, 0))
            components, _ = annotations.component_labels_and_sources(tifffile.imread(pair.mask))
            expected = estimate_from_analysis(
                record.reference, mask=components, volume_xy=dimensions == "2.5d",
                spacing=spacing, measure="principal_axes",
            )
            assert estimates[-1].median_diameter_px == pytest.approx(expected.median_diameter_px)
    assert estimates[0] == estimates[1]


def test_holdout_reference_never_uses_other_domain_objects(tmp_path):
    pair = write_pair(tmp_path, example_mask(2))
    domain = replace(pair, region=((0, 16), (0, 16)))
    with image_reading_session() as session:
        prep = AnnotationPreparation()
        session.annotations = prep
        prep.prepare([pair, domain], "2d", lambda: None)
        estimate = prep.reference_size(domain, "2d", (1, 1, 1), prep.scale_config, 0)
        assert estimate.available_instances == 1
        assert estimate.median_diameter_px == pytest.approx(np.sqrt(4 * 64 / np.pi))


def test_reference_respects_sample_limit_and_seed(tmp_path):
    mask = np.zeros((96, 96), np.uint16)
    for i in range(25):
        y, x = 2 + 18 * (i // 5), 2 + 18 * (i % 5)
        mask[y:y + 4 + i % 5, x:x + 4 + i % 5] = i + 1
    pair = write_pair(tmp_path, mask)
    options = InstanceScaleNormalizationConfig(max_instances_per_image=3)
    with image_reading_session() as session:
        prep = AnnotationPreparation(scale_config=options, seed=42)
        session.annotations = prep
        prep.prepare([pair], "2d", lambda: None)
        estimate = prep.reference_size(
            pair, "2d", (1, 1, 1), options, stable_seed(42, 0, str(pair.image.resolve()) + str(()))
        )
        assert estimate.sampled_instances == 3
        assert prep.analyze(pair).reference_diameter == estimate.median_diameter_px
        summary = aggregate_instance_statistics([estimate])
        assert summary["object_diameter_p20_px"] == summary["object_diameter_p80_px"] == estimate.median_diameter_px


@pytest.mark.parametrize("options", [
    {"cleanup": "invalid"}, {"minimum_diameter_ratio": 0}, {"minimum_diameter_ratio": -1},
    {"minimum_diameter_ratio": 1.01}, {"minimum_diameter_ratio": float("nan")},
])
def test_new_config_validation(tmp_path, options):
    with pytest.raises(ConfigError):
        parse_training_config({"dataset_path": tmp_path, "output_dir": tmp_path, "model_name": "x",
                               "annotation_preparation": options})
