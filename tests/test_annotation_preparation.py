import json
import pickle
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import tifffile

from jdll_unet import annotations, scale, targets
from jdll_unet.annotations import AnnotationPreparation, AnnotationRecord
from jdll_unet.augment import AugmentationConfig
from jdll_unet.config import AnnotationPreparationConfig, parse_training_config
from jdll_unet.dataset import JdllSegmentationDataset
from jdll_unet.errors import ConfigError
from jdll_unet.geometry import inspect_pair, load_domain_mask
from jdll_unet.image_reading import image_reading_session
from jdll_unet.io import ImageMaskPair
from jdll_unet.task_detect import detect_task_from_pairs
from jdll_unet.training_geometry import measure_case_instances


def write_pair(root, mask, name="sample"):
    root.mkdir(parents=True, exist_ok=True)
    image, labels = root / f"{name}.tif", root / f"{name}_mask.tif"
    axes = "YX" if mask.ndim == 2 else "ZYX"
    tifffile.imwrite(image, np.ones(mask.shape, np.float32), metadata={"axes": axes})
    tifffile.imwrite(labels, mask, metadata={"axes": axes})
    return inspect_pair(ImageMaskPair(image, labels, name))[0]


def disconnected_mask(ndim=2, label=7):
    mask = np.zeros((24,) * ndim, dtype=np.int64)
    mask[(slice(3, 7),) * ndim] = label
    mask[(slice(14, 18),) * ndim] = label
    return mask


@pytest.mark.parametrize("dimensions", ["2d", "2.5d", "3d"])
@pytest.mark.parametrize("ram_mb", [0, 1])
def test_shared_source_repair_and_scale_are_consistent(tmp_path, monkeypatch, dimensions, ram_mb):
    mask = disconnected_mask(2 if dimensions == "2d" else 3)
    pair = write_pair(tmp_path, mask)
    original_bytes = pair.mask.read_bytes()
    labeler = Mock(wraps=annotations.label_components)
    monkeypatch.setattr(annotations, "label_components", labeler)
    monkeypatch.setattr(scale, "canonicalize_instance_volume", Mock(side_effect=AssertionError("Repeated repair")))
    monkeypatch.setattr(targets, "canonical_instance_labels", Mock(side_effect=AssertionError("Patch repair")))
    cfg = parse_training_config(
        {
            "model_name": "prepare",
            "dataset_path": tmp_path,
            "output_dir": tmp_path / "out",
            "architecture": f"resenc-tiny-{dimensions}",
            "instance_scale_normalization": {"min_instance_area": 1},
        }
    )
    with image_reading_session() as session:
        prep = AnnotationPreparation(AnnotationPreparationConfig(ram_cache_mb=ram_mb), cache_dir=tmp_path / "cache")
        session.annotations = prep
        # A small stable label set needs no connectivity until instance preparation.
        detection = detect_task_from_pairs([pair], requested_task="auto", dimensions=dimensions, preparation=prep)
        assert not detection["stats"][0]["connectivity_analyzed"]
        assert labeler.call_count == 0
        np.testing.assert_array_equal(load_domain_mask(pair, dimensions), mask)
        prep.prepare([pair], dimensions, lambda: None)
        prepared = load_domain_mask(pair, dimensions, raw=True)
        np.testing.assert_array_equal(load_domain_mask(pair, raw=True), prepared)
        assert np.unique(prepared).tolist() == [0, 7, 8]
        np.testing.assert_array_equal(load_domain_mask(pair, dimensions, original=True), mask)
        estimate, _ = measure_case_instances(pair, cfg, dimensions, (1, 1, 1))
        assert estimate.available_instances == 2
        dataset = JdllSegmentationDataset(
            [pair],
            "instance_friendly",
            None,
            {"type": "none"},
            AugmentationConfig(patch_size=(24,) * (3 if dimensions == "3d" else 2)),
            False,
            dimensions=dimensions,
        )
        for _ in range(3):
            dataset[0]
        assert labeler.call_count == 1
        record = next(iter(prep.records.values()))
        assert record.storage == ("ram" if ram_mb else "disk")
        assert record.labels.dtype == np.uint8
        assert not record.labels.flags.writeable
        assert prep.ram_bytes <= ram_mb * 1024**2
        disk_path = record.disk_path
        report = prep.diagnostics()
        assert report["affected_id_fraction"] == report["affected_source_fraction"] == 1.0
        assert report["repaired_components"] == 1
        assert report["sources"][0]["storage_dtype"] == "uint8"
        assert report["sources"][0]["storage_bytes"] == mask.size
    assert disk_path is None or not disk_path.exists()
    assert pair.mask.read_bytes() == original_bytes


@pytest.mark.parametrize("failure", ["space", "write", "disabled"])
def test_skip_is_reported_and_never_repaired_later(tmp_path, monkeypatch, failure):
    pair = write_pair(tmp_path, disconnected_mask(label=1))
    events = []
    labeler = Mock(wraps=annotations.label_components)
    monkeypatch.setattr(annotations, "label_components", labeler)
    if failure == "space":
        monkeypatch.setattr(annotations.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    if failure == "write":
        def failed_write(stream, _labels, _dtype):
            stream.write(b"partial cache")
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(annotations, "_write_compact_labels", failed_write)
    monkeypatch.setattr(targets, "canonical_instance_labels", Mock(side_effect=AssertionError("Patch repair")))
    with image_reading_session() as session:
        prep = AnnotationPreparation(
            AnnotationPreparationConfig(ram_cache_mb=0, repair_disconnected_instances=failure != "disabled"),
            cache_dir=tmp_path / "cache",
            emit=lambda event, **kw: events.append((event, kw)),
        )
        session.annotations = prep
        prep.prepare([pair], "2d", lambda: None)
        dataset = JdllSegmentationDataset(
            [pair], "instance_friendly", None, {"type": "none"}, AugmentationConfig(patch_size=(24, 24)), False
        )
        for _ in range(3):
            _, target = dataset[0]
            assert np.unique(target["instances"]).tolist() == [0, 1]
        cfg = parse_training_config({"model_name": "x", "dataset_path": tmp_path, "output_dir": tmp_path})
        estimate, _ = measure_case_instances(pair, cfg, "2d", (1, 1, 1))
        assert estimate.available_instances == 1
        assert labeler.call_count == 1
        report = prep.diagnostics()
        assert report["repaired_components"] == 0
        assert report["sources"][0]["status"] == "repair_skipped"
        assert len([kw for _, kw in events if kw["reason"] == "annotation_repair_skipped"]) == 1
        assert not list((tmp_path / "cache").rglob("*.npy"))


def test_semantic_ids_are_not_split_and_no_connectivity_is_needed(tmp_path, monkeypatch):
    mask = disconnected_mask()
    pair = write_pair(tmp_path, mask)
    checker = Mock(side_effect=AssertionError("Semantic connectivity analysis"))
    monkeypatch.setattr(annotations, "label_components", checker)
    with image_reading_session() as session:
        prep = AnnotationPreparation()
        session.annotations = prep
        assert (
            detect_task_from_pairs([pair], requested_task="multiclass_semantic", preparation=prep)["task"]
            == "multiclass_semantic"
        )
        dataset = JdllSegmentationDataset(
            [pair], "multiclass_semantic", [7], {"type": "none"}, AugmentationConfig(patch_size=(24, 24)), False
        )
        np.testing.assert_array_equal(dataset._load_item(0)[2], mask)
        assert not prep.records


def test_unchanged_sources_and_budget_do_not_keep_unnecessary_copies(tmp_path):
    mask = disconnected_mask()
    mask[14:18, 14:18] = 9000
    pair = write_pair(tmp_path, mask)
    with image_reading_session() as session:
        prep = AnnotationPreparation(cache_dir=tmp_path / "cache")
        session.annotations = prep
        prep.prepare([pair], "2d", lambda: None)
        record = prep.analyze(pair)
        assert record.labels is None and record.storage is None
        assert prep.ram_bytes == 0
        assert not (tmp_path / "cache").exists()
        assert prep.diagnostics()["sources"][0]["status"] == "unchanged"
        # Replace the source: cached connectivity and repairs must be invalidated.
        tifffile.imwrite(pair.mask, disconnected_mask(), metadata={"axes": "YX"})
        assert np.unique(load_domain_mask(pair)).tolist() == [0, 7, 8]
        assert len(prep.records) == 1


def test_full_volume_identity_survives_slicing_and_spatial_holdout(tmp_path, monkeypatch):
    mask = np.zeros((8, 24, 24), dtype=np.int64)
    mask[1:7, 3:7, 3:7] = 7
    mask[1:7, 3:7, 14:18] = 7
    mask[6, 3:7, 3:18] = 7  # Joined only in a different context plane.
    pair = write_pair(tmp_path, mask)
    labeler = Mock(wraps=annotations.label_components)
    monkeypatch.setattr(annotations, "label_components", labeler)
    with image_reading_session() as session:
        prep = AnnotationPreparation()
        session.annotations = prep
        domain = replace(pair, region=((1, 5), (0, 24), (0, 24)))
        prep.prepare([pair, domain], "2.5d", lambda: None)
        assert prep.diagnostics()["affected_ids"] == 0
        assert labeler.call_count == 1
        cropped = load_domain_mask(domain, "2.5d")
        assert cropped.shape == (4, 24, 24)
        target = targets.prepare_target("instance_friendly", cropped[0], canonicalize_instances=False)
        assert np.unique(target["instances"]).tolist() == [0, 1]


def test_sparse_ids_do_not_change_detection_score(tmp_path):
    mask = np.zeros((40, 40), np.int64)
    for i in range(12):
        mask[2 + i * 3, 2:4] = i + 1
    pair = write_pair(tmp_path, mask)
    first = detect_task_from_pairs([pair], dimensions="2d")
    mask[mask > 0] *= 1000
    tifffile.imwrite(pair.mask, mask, metadata={"axes": "YX"})
    second = detect_task_from_pairs([pair], dimensions="2d")
    assert first["score"] == second["score"]
    assert first["task"] == second["task"] == "instance_friendly"


def test_large_ids_and_id_overflow(tmp_path):
    for value in (2**40, np.iinfo(np.int64).max):
        pair = write_pair(tmp_path, disconnected_mask(label=value), name=str(value))
        with image_reading_session() as session:
            prep = AnnotationPreparation()
            session.annotations = prep
            prep.prepare([pair], "2d", lambda: None)
            record = prep.analyze(pair)
            if value == 2**40:
                assert np.unique(load_domain_mask(pair)).tolist() == [0, value, value + 1]
            else:
                assert record.skipped_reason == "instance_id_overflow"
                np.testing.assert_array_equal(load_domain_mask(pair), disconnected_mask(label=value))


@pytest.mark.parametrize(
    "options",
    [
        {"ram_cache_mb": -1},
        {"disk_reserve_mb": "invalid"},
        {"warning_fraction": 1.1},
        {"ram_cache_mb": float("nan")},
        {"cache_dir": ""},
        {"repair_disconnected_instances": "maybe"},
    ],
)
def test_preparation_config_validation(tmp_path, options):
    with pytest.raises(ConfigError):
        parse_training_config(
            {
                "model_name": "x",
                "dataset_path": tmp_path,
                "output_dir": tmp_path,
                "annotation_preparation": options,
            }
        )


def test_config_roundtrip(tmp_path):
    cfg = parse_training_config(
        {
            "model_name": "x",
            "dataset_path": tmp_path,
            "output_dir": tmp_path,
            "annotation_preparation": {"ram_cache_mb": 0, "cache_dir": Path(tmp_path / "cache")},
        }
    )
    assert parse_training_config(cfg).annotation_preparation == cfg.annotation_preparation


def test_task_detection_uses_only_labels_in_training_region(tmp_path):
    mask = np.zeros((64, 64), dtype=np.int64)
    mask[4:10, 4:10] = 1
    for label in range(1, 16):
        mask[40:44, 2 + label * 3] = label
    pair = write_pair(tmp_path, mask)
    domain = replace(pair, region=((0, 32), (0, 64)))
    with image_reading_session() as session:
        session.annotations = AnnotationPreparation()
        result = detect_task_from_pairs([domain], dimensions="2d", preparation=session.annotations)
        assert result["task"] == "binary_semantic"
        assert result["unique_label_values"] == [1]
        assert result["stats"][0]["connected_components_per_label"] == {}
        assert not result["stats"][0]["connectivity_analyzed"]


def test_serialized_reader_reuses_cache_without_owning_files(tmp_path):
    pair = write_pair(tmp_path, disconnected_mask())
    with image_reading_session() as session:
        prep = AnnotationPreparation(
            AnnotationPreparationConfig(ram_cache_mb=0),
            cache_dir=tmp_path / "cache",
            emit=lambda *_args, **_kwargs: None,
        )
        session.annotations = prep
        prep.prepare([pair], "2d", lambda: None)
        path = prep.analyze(pair).disk_path
        restored = pickle.loads(pickle.dumps(session))
        assert restored.annotations.emit is None
        record = restored.annotations.analyze(pair)
        assert isinstance(record.labels, np.memmap)
        np.testing.assert_array_equal(record.labels, prep.analyze(pair).labels)
        restored.annotations.close()
        assert path.exists()
    assert not path.exists()


def test_atomic_source_replacement_releases_old_cache(tmp_path):
    pair = write_pair(tmp_path, disconnected_mask())
    with image_reading_session() as session:
        prep = AnnotationPreparation()
        session.annotations = prep
        prep.prepare([pair], "2d", lambda: None)
        assert prep.ram_bytes > 0
        mask = disconnected_mask()
        mask[14:18, 14:18] = 8
        replacement = tmp_path / "replacement.tif"
        tifffile.imwrite(replacement, mask, metadata={"axes": "YX"})
        replacement.replace(pair.mask)
        np.testing.assert_array_equal(load_domain_mask(pair), mask)
        assert len(prep.records) == 1
        assert prep.ram_bytes == 0


def test_cache_budget_applies_across_sources_and_policy_changes(tmp_path):
    first = write_pair(tmp_path, disconnected_mask(), "first")
    second = write_pair(tmp_path, disconnected_mask(), "second")
    budget = disconnected_mask().size  # The repaired IDs fit in uint8.
    with image_reading_session() as session:
        prep = AnnotationPreparation(
            AnnotationPreparationConfig(ram_cache_mb=budget / 1024**2), cache_dir=tmp_path / "cache"
        )
        session.annotations = prep
        prep.prepare([first, second], "2d", lambda: None)
        assert prep.ram_bytes == budget
        assert prep.analyze(first).storage == "ram"
        assert prep.analyze(second).storage == "disk"
        prep.config.repair_disconnected_instances = False
        np.testing.assert_array_equal(load_domain_mask(first), disconnected_mask())
        assert prep.ram_bytes == 0
        assert prep.analyze(first).skipped_reason == "repair_disabled"


@pytest.mark.parametrize("ram_mb", [0, 1])
@pytest.mark.parametrize(
    "label,dtype",
    [
        (254, np.uint8),
        (255, np.uint16),
        (65534, np.uint16),
        (65535, np.uint32),
        (2**32 - 2, np.uint32),
        (2**32 - 1, np.uint64),
        (2**40, np.uint64),
        (np.iinfo(np.int64).max - 1, np.uint64),
    ],
)
def test_repaired_cache_dtype_promotes_without_changing_ids(tmp_path, ram_mb, label, dtype):
    mask = disconnected_mask(label=label)
    pair = write_pair(tmp_path, mask)
    expected = mask.astype(dtype)
    expected[14:18, 14:18] = label + 1
    with image_reading_session() as session:
        prep = AnnotationPreparation(
            AnnotationPreparationConfig(ram_cache_mb=ram_mb), cache_dir=tmp_path / "cache"
        )
        session.annotations = prep
        prep.prepare([pair], "2d", lambda: None)
        record = prep.analyze(pair)
        assert record.labels.dtype == dtype
        assert not record.labels.flags.writeable
        np.testing.assert_array_equal(record.labels, expected)
        np.testing.assert_array_equal(load_domain_mask(pair), expected.astype(np.int64))
        assert prep.ram_bytes == (expected.nbytes if ram_mb else 0)
        if not ram_mb:
            assert isinstance(record.labels, np.memmap)
            assert expected.nbytes < record.disk_path.stat().st_size < expected.nbytes + 4096
            np.testing.assert_array_equal(np.load(record.disk_path, allow_pickle=False), expected)


@pytest.mark.parametrize("space_difference", [-1, 0])
def test_disk_free_space_check_uses_compact_size(tmp_path, monkeypatch, space_difference):
    mask = disconnected_mask(label=255)
    pair = write_pair(tmp_path, mask)
    required = mask.size * 2 + 4096 + 1024**2  # The new ID 256 requires uint16.
    monkeypatch.setattr(
        annotations.shutil, "disk_usage", lambda _: SimpleNamespace(free=required + space_difference)
    )
    with image_reading_session() as session:
        prep = AnnotationPreparation(
            AnnotationPreparationConfig(ram_cache_mb=0, disk_reserve_mb=1), cache_dir=tmp_path / "cache"
        )
        session.annotations = prep
        prep.prepare([pair], "2d", lambda: None)
        record = prep.analyze(pair)
        if space_difference < 0:
            assert record.skipped_reason == "insufficient_cache_space"
            assert record.labels is None
            assert not list((tmp_path / "cache").rglob("*.npy"))
        else:
            assert record.skipped_reason is None
            assert record.storage == "disk"
            assert record.labels.dtype == np.uint16


@pytest.mark.parametrize("layout", ["c", "fortran", "strided"])
def test_compact_disk_writer_handles_chunk_boundaries_and_layout(tmp_path, monkeypatch, layout):
    labels = np.arange(210, dtype=np.int64).reshape(5, 6, 7)
    if layout == "fortran":
        labels = np.asfortranarray(labels)
    elif layout == "strided":
        labels = labels.transpose(2, 0, 1)[::-1, :, ::2]
    monkeypatch.setattr(annotations, "_STORAGE_CHUNK_VOXELS", 13)
    path = tmp_path / "compact.npy"
    with path.open("wb") as stream:
        annotations._write_compact_labels(stream, labels, np.dtype(np.uint8))
    restored = np.load(path, allow_pickle=False)
    assert restored.dtype == np.uint8
    assert restored.flags.c_contiguous
    np.testing.assert_array_equal(restored, labels)


def test_ram_allocation_failure_falls_back_to_chunked_disk_write(tmp_path, monkeypatch):
    class AllocationFailureArray(np.ndarray):
        def astype(self, *args, **kwargs):
            if self.size > 13:
                raise MemoryError("Cannot allocate compact RAM copy")
            return super().astype(*args, **kwargs)

    monkeypatch.setattr(annotations, "_STORAGE_CHUNK_VOXELS", 13)
    labels = disconnected_mask().view(AllocationFailureArray)
    with image_reading_session() as session:
        prep = AnnotationPreparation(cache_dir=tmp_path / "cache")
        session.annotations = prep
        record = AnnotationRecord("test", {7: 2})
        prep.records[("test",)] = record
        prep._retain(labels, record, maximum=7)
        assert record.storage == "disk"
        assert prep.ram_bytes == 0
        assert record.labels.dtype == np.uint8
        np.testing.assert_array_equal(record.labels, labels)


def test_semantic_detection_discards_candidates_and_cleanup_on_error(tmp_path):
    pair = write_pair(tmp_path, disconnected_mask(label=1))
    with pytest.raises(RuntimeError, match="cancelled"), image_reading_session() as session:
        prep = AnnotationPreparation(AnnotationPreparationConfig(ram_cache_mb=0), cache_dir=tmp_path / "cache")
        session.annotations = prep
        result = detect_task_from_pairs([pair], dimensions="2d", preparation=prep)
        assert result["task"] == "binary_semantic"
        record = prep.analyze(pair)
        path = record.disk_path
        assert path.exists()
        np.testing.assert_array_equal(load_domain_mask(pair), disconnected_mask(label=1))
        prep.report()
        assert not path.exists()
        assert prep.diagnostics()["sources"][0]["status"] == "semantic_unchanged"
        raise RuntimeError("cancelled")
    assert not list((tmp_path / "cache").iterdir())


@pytest.mark.parametrize("dimensions,scale_enabled", [("2d", True), ("2.5d", True), ("3d", True), ("2d", False)])
def test_training_full_validation_reuse_prepared_sources(tmp_path, monkeypatch, dimensions, scale_enabled):
    from jdll_unet.trainer import train

    mask = disconnected_mask(2 if dimensions == "2d" else 3)
    dataset = tmp_path / "data"
    for split in ("train", "val"):
        image_dir, mask_dir = dataset / split / "images", dataset / split / "masks"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir()
        axes = "YX" if mask.ndim == 2 else "ZYX"
        tifffile.imwrite(image_dir / "image.tif", np.ones(mask.shape, np.float32), metadata={"axes": axes})
        tifffile.imwrite(mask_dir / "image.tif", mask, metadata={"axes": axes})
    labeler = Mock(wraps=annotations.label_components)
    monkeypatch.setattr(annotations, "label_components", labeler)
    monkeypatch.setattr(targets, "canonical_instance_labels", Mock(side_effect=AssertionError("Repeated repair")))
    events = []
    out = tmp_path / "out"
    train(
        {
            "model_name": "annotation",
            "dataset_path": dataset,
            "output_dir": out,
            "architecture": f"resenc-tiny-{dimensions}",
            "task": "instance_friendly",
            "patch_size": (24,) * (3 if dimensions == "3d" else 2),
            "batch_size": 1,
            "effective_batch_size": 1,
            "epochs": 1,
            "steps_per_epoch": 1,
            "preview_count": 0,
            "augmentation_profile": "fast",
            "validation": {"full_every": 1, "light_steps": 1},
            "annotation_preparation": {"ram_cache_mb": 0},
            "instance_scale_normalization": {"enabled": scale_enabled},
        },
        task=lambda event: events.append(event),
    )
    plan = json.loads((out / "dataset_plan.json").read_text())
    assert plan["annotation_preparation"]["sources_analyzed"] == 2
    assert plan["annotation_preparation"]["repaired_components"] == 2
    assert labeler.call_count == 2
    assert any(event.get("reason") == "annotation_preparation" for event in events)
    assert not list((out / ".annotation_cache").iterdir())
