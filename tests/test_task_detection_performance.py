from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
import tifffile
from scipy import ndimage as ndi

from jdll_unet import annotations, geometry, task_detect
from jdll_unet.annotations import AnnotationPreparation, component_labels_and_sources
from jdll_unet.image_reading import image_reading_session
from jdll_unet.io import ImageMaskPair


def write_pair(root, mask):
    image_path, mask_path = root / "image.tif", root / "mask.tif"
    axes = "YX" if mask.ndim == 2 else "ZYX"
    tifffile.imwrite(image_path, np.zeros(mask.shape, np.uint8), metadata={"axes": axes})
    tifffile.imwrite(mask_path, mask, metadata={"axes": axes})
    return ImageMaskPair(image_path, mask_path, "case")


@pytest.mark.parametrize("shape", [(21, 23), (7, 21, 23)])
def test_single_pass_statistics_match_per_label_reference(shape, monkeypatch):
    rng = np.random.default_rng(7)
    mask = np.array([0, 7, 9000, 2**40], dtype=np.int64)[rng.integers(0, 4, size=shape)]
    mask = mask.T[::-1]  # Noncontiguous inputs must preserve voxel/source correspondence.
    checker = Mock(wraps=annotations.label_components)
    monkeypatch.setattr(annotations, "label_components", checker)
    result = task_detect.mask_statistics(mask)
    expected = {int(label): int(ndi.label(mask == label)[1]) for label in np.unique(mask) if label}
    assert result.connected_components_per_label == expected
    assert result.connectivity_analyzed
    assert checker.call_count == 1


def test_thousands_of_sparse_instance_ids_use_one_labeling_call(monkeypatch):
    mask = np.zeros((33, 33, 33), dtype=np.int64)
    count = mask[::2, ::2, ::2].size
    mask[::2, ::2, ::2] = (np.arange(1, count + 1).reshape((17, 17, 17)) + 2**40)
    checker = Mock(wraps=annotations.label_components)
    monkeypatch.setattr(annotations, "label_components", checker)
    stats = task_detect.mask_statistics(mask)
    assert len(stats.unique_nonzero_labels) == count
    assert set(stats.connected_components_per_label.values()) == {1}
    assert checker.call_count == 1


def test_component_mapping_across_chunk_boundary():
    mask = np.zeros((65, 128, 128), np.int64)
    mask[:, 2:25, 3:40] = 2**40
    mask[:, 30:90, 50:100] = 7
    mask[-1, 110:120, 110:120] = 7
    components, source_ids = component_labels_and_sources(mask)
    np.testing.assert_array_equal(source_ids[components], mask)
    assert sorted(source_ids.tolist()) == [0, 7, 7, 2**40]


@pytest.mark.parametrize("label_count", [0, 1, 2])
@pytest.mark.parametrize("with_preparation", [False, True])
def test_decisive_label_statistics_skip_connectivity(tmp_path, monkeypatch, label_count, with_preparation):
    mask = np.zeros((8, 16, 16), dtype=np.int64)
    if label_count:
        mask[:, 1:5, 1:5] = 1
    if label_count == 2:
        mask[:, 8:12, 8:12] = 2
    pair = write_pair(tmp_path, mask)
    monkeypatch.setattr(annotations, "label_components", Mock(side_effect=AssertionError("Unneeded connectivity")))
    prep = AnnotationPreparation() if with_preparation else None
    try:
        result = task_detect.detect_task_from_pairs([pair], dimensions="3d", preparation=prep)
        assert result["task"] == ("multiclass_semantic" if label_count == 2 else "binary_semantic")
        assert result["stats"][0]["connected_components_per_label"] == {}
        assert not result["stats"][0]["connectivity_analyzed"]
        assert prep is None or not prep.records
    finally:
        if prep is not None:
            prep.close()


def test_inspection_labels_are_reused_and_invalidated(tmp_path, monkeypatch):
    mask = np.zeros((8, 16, 16), np.int64)
    mask[:, 2:5, 2:5] = 1
    pair = write_pair(tmp_path, mask)
    with image_reading_session() as session:
        session.annotations = prep = AnnotationPreparation()
        inspected, _ = geometry.inspect_pair(pair)
        with monkeypatch.context() as patch:
            patch.setattr(geometry, "load_domain_mask", Mock(side_effect=AssertionError("Repeated mask read")))
            patch.setattr(task_detect, "load_domain_mask", Mock(side_effect=AssertionError("Repeated mask read")))
            result = task_detect.detect_task_from_pairs([inspected], dimensions="3d", preparation=prep)
            assert result["unique_label_values"] == [1]
        mask[:, 8:12, 8:12] = 2
        tifffile.imwrite(pair.mask, mask, metadata={"axes": "ZYX"})
        result = task_detect.detect_task_from_pairs([inspected], dimensions="3d", preparation=prep)
        assert result["unique_label_values"] == [1, 2]
        domain = replace(inspected, region=((0, 8), (0, 6), (0, 16)))
        result = task_detect.detect_task_from_pairs([domain], dimensions="3d", preparation=prep)
        assert result["unique_label_values"] == [1]


def test_uncertain_detection_connectivity_reused_for_preparation(tmp_path, monkeypatch):
    mask = np.zeros((8, 32, 32), np.int64)
    for index, label in enumerate((7, 7, 8, 9, 10)):
        mask[2:5, 2:5, 2 + 5 * index : 5 + 5 * index] = label
    pair = write_pair(tmp_path, mask)
    before = pair.mask.read_bytes()
    checker = Mock(wraps=annotations.label_components)
    monkeypatch.setattr(annotations, "label_components", checker)
    with image_reading_session() as session:
        session.annotations = prep = AnnotationPreparation()
        result = task_detect.detect_task_from_pairs([pair], dimensions="3d", preparation=prep)
        assert result["ambiguous"]
        assert result["stats"][0]["connected_components_per_label"] == {7: 2, 8: 1, 9: 1, 10: 1}
        assert result["stats"][0]["connectivity_analyzed"]
        prep.prepare([pair], "3d", lambda: None)
        assert np.unique(geometry.load_domain_mask(pair, "3d")).tolist() == [0, 7, 8, 9, 10, 11]
        assert checker.call_count == 1
    assert pair.mask.read_bytes() == before


def test_many_labels_do_not_force_instance_interpretation(tmp_path):
    mask = np.zeros((8, 48, 48), np.int64)
    for label in range(1, 13):
        for z in range(0, 8, 2):
            mask[z, label * 3, 2:5] = label
    pair = write_pair(tmp_path, mask)
    result = task_detect.detect_task_from_pairs([pair], dimensions="3d")
    assert result["task"] == "ambiguous"
    assert result["score"] == 1  # Many labels (+3), but repeated disconnected regions (-2).


def test_decisive_object_metadata_skips_connectivity(tmp_path, monkeypatch):
    mask = np.zeros((8, 48, 48), np.int64)
    for label in range(1, 13):
        mask[:, label * 3, 2:5] = label
    pair = write_pair(tmp_path, mask)
    monkeypatch.setattr(task_detect, "_metadata_signals", lambda _: {
        "annotation_source": "roi_manager_one_roi_per_object",
        "class_names": [], "bounding_boxes": False, "points": False,
    })
    monkeypatch.setattr(annotations, "label_components", Mock(side_effect=AssertionError("Unneeded connectivity")))
    result = task_detect.detect_task_from_pairs([pair], dataset_path=tmp_path, dimensions="3d")
    assert result["task"] == "instance_friendly"
    assert not result["stats"][0]["connectivity_analyzed"]
