import functools
import json
import math
import pickle
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import tifffile
import torch
from torch.utils.data import DataLoader

from jdll_unet import FullValidationController, trainer
from jdll_unet.augment import AugmentationConfig
from jdll_unet.config import (
    PostprocessingConfig,
    TrainingConfig,
    ValidationConfig,
    parse_training_config,
    resolve_validation_config,
)
from jdll_unet.crop_reading import ResizedCropArray
from jdll_unet.dataset import JdllSegmentationDataset
from jdll_unet.errors import ConfigError, TrainingError
from jdll_unet.geometry import DomainReader, inspect_pair, with_region
from jdll_unet.image_reading import image_reading_session
from jdll_unet.io import ImageMaskPair
from jdll_unet.losses import compute_loss
from jdll_unet.metrics import _instance_label_metrics, compute_metrics
from jdll_unet.targets import prepare_target
from jdll_unet.validation_control import FullValidationSchedule
from jdll_unet.validation_metrics import ValidationAccumulator
from jdll_unet.validation_previews import save_enlarged_previews
from jdll_unet.validation_sampling import PlannedValidationDataset
from jdll_unet.volumetric_validation import full_validation, regular_validation


def pair(root, name="case", shape=(16, 64, 64), mask=None):
    root.mkdir(parents=True, exist_ok=True)
    paths = ImageMaskPair(root / f"{name}.tif", root / f"{name}_mask.tif", name)
    image = np.arange(math.prod(shape), dtype=np.float32).reshape(shape) % 100
    if mask is None:
        mask = np.zeros(shape, dtype=np.uint16)
        mask[::2, ::2, ::2] = 1
    tifffile.imwrite(paths.image, image, metadata={"axes": "ZYX"}, photometric="minisblack")
    tifffile.imwrite(paths.mask, mask, metadata={"axes": "ZYX"}, photometric="minisblack")
    return inspect_pair(paths)[0]


def make_plan(tmp_path, pairs, *, dimensions="3d", patch=(4, 8, 8), options=None, batch=1,
              task="binary_semantic", labels=None, configure=None, resume_path=None):
    base = JdllSegmentationDataset(pairs, task, labels or [1], {"type": "minmax"},
                                  AugmentationConfig(patch_size=patch), False, dimensions=dimensions, seed=51)
    if configure:
        configure(base)
    events = []
    def emit(kind, **payload):
        events.append({"type": kind, **payload})
    options = options or ValidationConfig(preview_max_bytes=256 * 1024**2)
    plan = PlannedValidationDataset(base, options, batch, check_cancel=lambda: None, emit=emit,
                                    path=tmp_path / "validation_plan.json", resume_path=resume_path)
    return plan, events, emit


@pytest.mark.parametrize("batch,expected", [(1, 100), (2, 100), (4, 200)])
def test_budget_quota_plan_reuse_and_mask_first(tmp_path, batch, expected, monkeypatch):
    with image_reading_session():
        pairs = [pair(tmp_path / str(i)) for i in range(2)]
        original = JdllSegmentationDataset._load_item
        monkeypatch.setattr(JdllSegmentationDataset, "_load_item", Mock(side_effect=AssertionError("image decoded in planning")))
        data, _, _ = make_plan(tmp_path, pairs, batch=batch)
        assert len(data) == expected
        assert data.summary["batches"] == expected // batch
        assert data.summary["forced_samples"] == math.ceil(0.33 * expected)
        assert data.summary["forced_sources"] >= 1
        assert len(set(data.samples)) == expected
        assert data.summary["candidate_masks_checked"] <= math.ceil(0.33 * expected) * 16
        saved = json.loads(data.path.read_text())
        assert saved["sources"][0]["files"]["mask"]["path"] == str(pairs[0].mask.resolve())
        assert saved["patch_size"] == list(data.patch)
        assert saved["domains"][0]["center_z_in_domain"] is None
        assert len(data.forced_indices) == math.ceil(0.33 * expected)
        for index in data.forced_indices:
            domain, cell = data.samples[index]
            _, mask, valid = data.read_tile(domain, data.domains[domain].origin(cell, data.patch), image=False)
            assert np.count_nonzero(mask[valid]) / np.count_nonzero(valid) >= 0.01
        repeated, _, _ = make_plan(tmp_path, pairs, batch=batch, resume_path=data.path)
        assert repeated.samples == data.samples
        monkeypatch.setattr(JdllSegmentationDataset, "_load_item", original)
        before = data[0]
        data.base.set_epoch(99)
        after = data[0]
        torch.testing.assert_close(before[0], after[0])
        torch.testing.assert_close(before[1]["semantic"], after[1]["semantic"])


def test_small_capacity_drops_foreground_policy_and_retains_negatives(tmp_path):
    with image_reading_session():
        source = pair(tmp_path / "data", shape=(4, 12, 12), mask=np.zeros((4, 12, 12), np.uint8))
        plan, events, _ = make_plan(tmp_path, [source], patch=(8, 8, 8), batch=4)
        assert len(plan) == 1
        assert plan.summary["foreground_requirements_removed"]
        assert plan.summary["forced_samples"] == 0
        image, target, _ = plan[0]
        assert image.shape == (1, 8, 8, 8)
        assert target["valid"].sum() == 4 * 8 * 8
        assert not target["semantic"].any()
        assert any(event["type"] == "warning" for event in events)


@pytest.mark.parametrize("task", ["binary_semantic", "instance_friendly"])
def test_validation_plan_can_be_serialized_for_spawn_workers(tmp_path, task):
    with image_reading_session(lambda *args, **kwargs: None):
        source = pair(tmp_path / "data", shape=(4, 8, 8))
        plan, _, _ = make_plan(tmp_path, [source], task=task)
        restored = pickle.loads(pickle.dumps(plan))
        assert restored.samples == plan.samples
        before, after = plan[0], restored[0]
        torch.testing.assert_close(before[0], after[0])
        for key in before[1]:
            torch.testing.assert_close(before[1][key], after[1][key])


@pytest.mark.parametrize("validation,expected", [
    ({"light_steps": "auto", "minimum_batches": 12}, 12),
    ({"light_steps": 12, "minimum_batches": "auto"}, 12),
    ({"light_steps": "auto", "minimum_batches": "auto"}, 50),
])
def test_automatic_legacy_validation_aliases(tmp_path, validation, expected):
    config = parse_training_config({"model_name": "test", "dataset_path": str(tmp_path), "output_dir": str(tmp_path / "model"),
                                    "validation": validation})
    assert config.validation.light_steps == config.validation.minimum_batches == expected


def test_sparse_foreground_lowers_occupancy_without_repeating_patches(tmp_path):
    with image_reading_session():
        mask = np.zeros((16, 64, 64), np.uint8)
        mask[::4, ::8, ::8] = 1
        source = pair(tmp_path / "data", mask=mask)
        plan, _, _ = make_plan(tmp_path, [source], patch=(4, 8, 8))
        assert plan.summary["forced_samples"] == 33
        assert 0 < plan.summary["resolved_minimum_foreground"] < 0.01
        assert len(set(plan.samples)) == len(plan)


def test_no_foreground_reports_shortfall_without_failure(tmp_path):
    with image_reading_session():
        source = pair(tmp_path / "data", mask=np.zeros((16, 64, 64), np.uint8))
        plan, _, _ = make_plan(tmp_path, [source])
        assert len(plan) == 100
        assert not plan.summary["limited_capacity"]
        assert plan.summary["forced_samples"] == 0
        assert plan.summary["candidate_masks_checked"] == 0


@pytest.mark.parametrize("dimensions", ["2.5d", "3d"])
def test_grid_is_fixed_and_respects_regions_and_overlap(tmp_path, dimensions):
    with image_reading_session():
        source = pair(tmp_path / "data")
        source = with_region(source, ((0, 8), (0, 32), (0, 32)), "holdout")
        patch = (4, 12, 12) if dimensions == "3d" else (12, 12)
        data, _, _ = make_plan(tmp_path, [source], dimensions=dimensions, patch=patch)
        for domain in data.domains:
            starts = [domain.origin(i, patch) for i in range(domain.capacity)]
            for axis, extent in enumerate(patch):
                positions = sorted({s[axis] for s in starts})
                assert all(right - left >= math.ceil(extent * 0.9) for left, right in zip(positions, positions[1:], strict=False))
            for start in starts:
                assert all(a >= 0 and a + p <= n for a, p, n in zip(start, patch, domain.shape, strict=True))


@pytest.mark.parametrize("shape,output", [((1, 7, 15), (1, 12, 9)), ((1, 5, 9, 11), (1, 9, 5, 7))])
def test_lazy_instance_resize_matches_global_interpolation(shape, output):
    image = np.random.default_rng(7).random(shape, dtype=np.float32)
    resized = ResizedCropArray(image, output[1:])
    expected = torch.nn.functional.interpolate(torch.from_numpy(image[None]), size=output[1:],
                 mode="bilinear" if len(shape) == 3 else "trilinear", align_corners=False)[0].numpy()
    for selection in (tuple(slice(0, n) for n in output[1:]), tuple(slice(1, n - 1) for n in output[1:])):
        actual = resized[(slice(None), *selection)]
        np.testing.assert_allclose(actual, expected[(slice(None), *selection)], atol=2e-6)
    labels = (image[0] * 10).astype(np.uint32) * 100000
    resized_labels = ResizedCropArray(labels, output[1:], mask=True)[tuple(slice(0, n) for n in output[1:])]
    indices = [np.arange(m) * n // m for n, m in zip(shape[1:], output[1:], strict=True)]
    np.testing.assert_array_equal(resized_labels, labels[np.ix_(*indices)])


@pytest.mark.parametrize("task,channels", [("binary_semantic", 1), ("multiclass_semantic", 3), ("instance_friendly", 3)])
@pytest.mark.parametrize("deep", [False, True])
def test_streaming_losses_metrics_match_whole_population(task, channels, deep):
    rng = np.random.default_rng(10)
    labels = rng.integers(0, 3, (5, 6, 8), dtype=np.int64)
    valid = np.ones(labels.shape, bool)
    valid[0, :2] = False
    valid[3, :, :2] = False
    targets = [prepare_target(task, mask, [1, 2], validity=support, canonicalize_instances=False)
               for mask, support in zip(labels, valid, strict=True)]
    target = {key: torch.from_numpy(np.stack([item[key] for item in targets])) for key in targets[0]}
    main = torch.from_numpy(rng.normal(size=(5, channels, 6, 8)).astype(np.float32))
    heads = [main, torch.nn.functional.avg_pool2d(main, 2)] if deep else main
    weights = {"focal": 0.7, "boundary_focal": 0.3, "boundary": 0.4, "distance": 1.1}
    loss, parts = compute_loss(task, heads, target, weights, focal_gamma=1.5, focal_alpha=0.4)
    expected_metrics = compute_metrics(task, heads, target)
    outputs = []
    for batch in (1, 2, 4, 5):
        accumulator = ValidationAccumulator(task, weights, 1.5, 0.4)
        for start in range(0, 5, batch):
            current = [h[start:start + batch] for h in heads] if deep else heads[start:start + batch]
            accumulator.update(current, {key: value[start:start + batch] for key, value in target.items()})
        losses, metrics = accumulator.result()
        assert losses["total_loss"] == pytest.approx(loss.item(), abs=2e-6)
        for key, value in parts.items():
            assert losses[key] == pytest.approx(value.item(), abs=2e-6)
        assert metrics == pytest.approx(expected_metrics, abs=2e-6)
        outputs.append((losses, metrics))
    assert outputs[0][0] == pytest.approx(outputs[-1][0], abs=2e-6)


def test_sparse_instance_matching_preserves_greedy_definition():
    truth = np.array([[0, 7000000000, 7000000000, 0], [11, 11, 0, 0]], dtype=np.int64)
    pred = np.array([[0, 8, 8, 0], [4, 4, 0, 0]], dtype=np.uint32)
    result = _instance_label_metrics(pred, truth)
    assert result["object_f1"] == result["aggregated_jaccard"] == result["panoptic_quality"] == 1
    assert result["split_error_rate"] == result["merge_error_rate"] == 0


def test_control_tokens_boundary_phase_and_arrival_during_full_pass():
    control = FullValidationController()
    events = []
    schedule = FullValidationSchedule(5, control, lambda kind, **payload: events.append({"type": kind, **payload}))
    assert schedule.boundary(4) is None
    assert schedule.boundary(5)["reason"] == "periodic"
    schedule.started()
    schedule.finish("completed")
    control.request_full_validation("one")
    control.request_full_validation("one")
    assert schedule.boundary(8)["request_ids"] == ["one"]
    schedule.started()
    control.request_full_validation("two")
    schedule.poll()
    schedule.finish("completed")
    assert schedule.next_epoch == 13
    assert schedule.boundary(9)["request_ids"] == ["two"]
    schedule.started()
    schedule.finish("completed")
    assert schedule.next_epoch == 14
    assert schedule.boundary(10) is None
    schedule.close()
    assert [event["status"] for event in events].count("pending") == 2
    assert len({event["run_id"] for event in events}) == 1


def test_polling_transport_duplicate_requests_do_not_repeat_and_manual_stays_disabled():
    schedule = FullValidationSchedule(0, lambda: "token", lambda *args, **kwargs: None)
    assert schedule.boundary(1)["reason"] == "requested"
    schedule.started()
    schedule.finish("completed")
    assert schedule.next_epoch is None
    assert schedule.boundary(2) is None


class CountingModel(torch.nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        self.channels = channels
        self.calls = 0

    def forward(self, image):
        self.calls += 1
        center = image[:, image.shape[1] // 2:image.shape[1] // 2 + 1]
        return (center * 2 - 1).repeat(1, self.channels, *([1] * (image.ndim - 2)))


@pytest.mark.parametrize("dimensions", ["2.5d", "3d"])
@pytest.mark.parametrize("blending", ["constant", "gaussian"])
def test_single_forward_enlarged_previews_geometry_and_retention(tmp_path, dimensions, blending):
    with image_reading_session():
        source = pair(tmp_path / "data")
        patch = (4, 8, 8) if dimensions == "3d" else (8, 8)
        cfg = ValidationConfig(minimum_batches=4, minimum_samples=4, preview_max_bytes=256 * 1024**2,
                               tile_blending=blending)
        data, events, emit = make_plan(tmp_path, [source], dimensions=dimensions, patch=patch, options=cfg)
        loader = DataLoader(data, batch_size=2, generator=torch.Generator().manual_seed(8))
        model = CountingModel()
        rng = torch.get_rng_state().clone()
        for epoch in (1, 2, 3):
            calls = model.calls
            losses, metrics, anchors = regular_validation(model, loader, data, torch.device("cpu"), epoch,
                weights={}, focal_gamma=2, focal_alpha=None, preview_count=4, progress_interval=1,
                emit=emit, check_cancel=lambda: None)
            assert model.calls - calls == 2
            assert len(anchors) == 4
            result = save_enlarged_previews(data, anchors, model, torch.device("cpu"), tmp_path, epoch,
                                            PostprocessingConfig(), emit=emit, check_cancel=lambda: None)
            manifest = json.loads(Path(result["latest_preview_path"]).read_text())
            assert manifest["additional_tiles"] == 12
            assert model.calls - calls == 14
            assert manifest["actual_preview_count"] == 4
            for item in manifest["items"]:
                shape = tuple(item["shape"])
                assert shape == ((4, 14, 14) if dimensions == "3d" else (14, 14))
                assert item["tile_layout"] == ([1, 2, 2] if dimensions == "3d" else [2, 2])
                prediction = np.load(item["assets"]["prediction"]["path"])
                image = np.load(item["assets"]["image"]["path"])
                expected = image[image.shape[0] // 2] >= 0.5
                np.testing.assert_array_equal(prediction > 0, expected)
                assert item["original_z_index"] is not None if dimensions == "2.5d" else item["original_z_index"] is None
            assert losses and metrics
        assert len(list((tmp_path / "previews").glob("volume_epoch_*"))) == 2
        assert not (tmp_path / "previews/epoch_0001.json").exists()
        assert torch.equal(rng, torch.get_rng_state())


def test_budget_skip_keeps_previous_manifest_usable(tmp_path):
    with image_reading_session():
        source = pair(tmp_path / "data")
        cfg = ValidationConfig(minimum_batches=1, minimum_samples=1, preview_max_bytes=1024**2)
        data, events, emit = make_plan(tmp_path, [source], options=cfg)
        model = CountingModel()
        def run(epoch):
            _, _, anchors = regular_validation(model, DataLoader(data, batch_size=1), data, torch.device("cpu"), epoch,
                weights={}, focal_gamma=2, focal_alpha=None, preview_count=1, progress_interval=1,
                emit=emit, check_cancel=lambda: None)
            return anchors
        save_enlarged_previews(data, run(1), model, torch.device("cpu"), tmp_path, 1, PostprocessingConfig(),
                              emit=emit, check_cancel=lambda: None)
        before = (tmp_path / "previews/latest.json").read_bytes()
        anchors = run(2)
        data.options.preview_max_bytes = 1
        assert save_enlarged_previews(data, anchors, model, torch.device("cpu"), tmp_path, 2, PostprocessingConfig(),
                                     emit=emit, check_cancel=lambda: None) is None
        assert (tmp_path / "previews/latest.json").read_bytes() == before


@pytest.mark.parametrize("dimensions", ["2.5d", "3d"])
def test_full_validation_reuses_normalization_and_has_native_coverage(tmp_path, dimensions, monkeypatch):
    with image_reading_session():
        source = pair(tmp_path / "data", shape=(4, 16, 16))
        patch = (4, 8, 8) if dimensions == "3d" else (8, 8)
        data, events, emit = make_plan(tmp_path, [source], dimensions=dimensions, patch=patch)
        data[0]
        monkeypatch.setattr("jdll_unet.dataset.fit_normalization", Mock(side_effect=AssertionError("normalization repeated")))
        result = full_validation(CountingModel(), data, torch.device("cpu"), 1, PostprocessingConfig(),
                                 emit=emit, check_cancel=lambda: None)
        assert result["skipped_domains"] == 0
        assert result["evaluated_domains"] == (4 if dimensions == "2.5d" else 1)
        assert result["evaluated_centers"]["case"] == (list(range(4)) if dimensions == "2.5d" else None)


@pytest.mark.parametrize("dimensions,budget", [("2d", 256), ("2.5d", 600), ("3d", 600)])
def test_configuration_roundtrip_and_dimensional_defaults(tmp_path, dimensions, budget):
    cfg = parse_training_config({"model_name": "x", "output_dir": tmp_path, "dataset_path": tmp_path})
    resolve_validation_config(cfg, dimensions, "resenc-tiny-2d" if dimensions == "2d" else "resenc-large-" + dimensions)
    assert cfg.preview_count == (20 if dimensions == "2d" else 4)
    assert cfg.validation.full_every == (5 if dimensions == "2d" else 0)
    assert cfg.validation.early_stopping_patience == (20 if dimensions == "2d" else 0)
    assert cfg.validation.preview_max_bytes == budget * 1024**2
    restored = parse_training_config(cfg.request_dict())
    assert asdict(restored.validation) == asdict(cfg.validation)


@pytest.mark.parametrize("field,value", [("full_every", -1), ("full_every", True), ("minimum_batches", 0),
                                        ("max_sampling_overlap", 0.2), ("minimum_foreground", 0),
                                        ("preview_max_bytes", False), ("tile_overlap", float("nan"))])
def test_invalid_validation_config(tmp_path, field, value):
    with pytest.raises(ConfigError):
        parse_training_config({"model_name": "x", "output_dir": tmp_path, "dataset_path": tmp_path,
                               "validation": {field: value}})


def training_config(tmp_path, epochs=2):
    for split in ("train", "val"):
        paths = pair(tmp_path / "data" / split, shape=(8, 16, 16))
        images = tmp_path / "data" / split / "images"
        masks = tmp_path / "data" / split / "masks"
        images.mkdir()
        masks.mkdir()
        paths.image.rename(images / "case.tif")
        paths.mask.rename(masks / "case.tif")
    return {"model_name": "validation", "output_dir": tmp_path / "model", "dataset_path": tmp_path / "data",
            "architecture": "resenc-tiny-3d", "patch_size": [8, 16, 16], "batch_size": 1,
            "effective_batch_size": 1, "steps_per_epoch": 1, "epochs": epochs, "task": "binary_semantic",
            "preview_count": 0}


def test_training_default_disables_full_and_selects_every_epoch(tmp_path, monkeypatch):
    config = training_config(tmp_path)
    monkeypatch.setattr(trainer, "full_validation", Mock(side_effect=AssertionError("unexpected full validation")))
    events = []
    result = trainer.train(config, task=events.append)
    history = json.loads((config["output_dir"] / "metrics.json").read_text())["history"]
    assert len(history) == 2
    best = torch.load(config["output_dir"] / "weights_best.pt", weights_only=False)
    assert best["metrics"]["val_metrics"]["dice"] == result["best_score"]
    assert best["model_config"]["validation_selection"]["scope"] == "regular_patches"
    assert sum(e["type"] == "checkpoint" and e["kind"] == "last" for e in events) == 2


def test_manual_full_failure_preserves_checkpoint_and_regular_monitor(tmp_path, monkeypatch):
    config = training_config(tmp_path)
    controller = FullValidationController()
    controller.request_full_validation("request")
    seen = []
    def fail(*args, **kwargs):
        saved = torch.load(config["output_dir"] / "weights_last.pt", weights_only=False)
        assert saved["epoch"] == 1
        assert not saved["metrics"].get("validation_pending")
        assert (config["output_dir"] / "weights_best.pt").exists()
        seen.append(saved["metrics"]["val_metrics"]["dice"])
        raise MemoryError("simulated diagnostic allocation failure")
    monkeypatch.setattr(trainer, "full_validation", fail)
    events = []
    result = trainer.train(config, task=events.append, control=controller)
    assert len(seen) == 1
    assert result["full_validation_failures"] == 1
    assert result["metrics"]["epoch"] == 2
    assert any(e["type"] == "full_validation" and e["status"] == "failed" for e in events)
    assert not any("full_validation" in row for row in json.loads((config["output_dir"] / "metrics.json").read_text())["history"])


def test_volumetric_cancel_after_validation_retains_completed_checkpoint(tmp_path):
    config = training_config(tmp_path)
    def stop(event):
        if event.get("message") == "UNet validation epoch 1":
            return False
    result = trainer.train(config, task=stop)
    assert result["cancelled"]
    last = torch.load(config["output_dir"] / "weights_last.pt", weights_only=False)
    assert last["epoch"] == 1 and not last["metrics"].get("cancelled")
    assert (config["output_dir"] / "weights_cancelled.pt").exists()


def test_direct_config_has_same_defaults_and_legacy_batch_alias(tmp_path):
    cfg = TrainingConfig("direct", tmp_path, tmp_path, validation=ValidationConfig(light_steps=3))
    parsed = parse_training_config(cfg)
    resolve_validation_config(parsed, "3d", "resenc-tiny-3d")
    assert parsed.preview_count == 4
    assert parsed.validation.minimum_batches == parsed.validation.light_steps == 3
    assert parsed.validation.preview_max_bytes == 256 * 1024**2


def test_requests_during_validation_wait_and_full_scores_do_not_select_best(tmp_path, monkeypatch):
    config = training_config(tmp_path, epochs=3)
    config["validation"] = {"full_every": 5}
    controller = FullValidationController()
    requests = []
    events = []
    def callback(event):
        events.append(event)
        if event["type"] == "validation" and event.get("status") == "started" and event["epoch"] == 1:
            controller.request_full_validation("late")
    def full(*args, **kwargs):
        epoch = args[3]
        requests.append(epoch)
        if epoch == 2:
            controller.request_full_validation("during_full")
        kwargs["emit"]("full_validation", status="progress", epoch=epoch, current=1, maximum=1)
        return {"mean_dice": 1.0, "per_case_dice": {"case": 1.0}}
    monkeypatch.setattr(trainer, "full_validation", full)
    result = trainer.train(config, task=callback, control=controller)
    assert requests == [2, 3]
    assert result["best_score"] < 1
    history = json.loads((config["output_dir"] / "metrics.json").read_text())["history"]
    assert result["best_score"] == max(row["val_metrics"]["dice"] for row in history)
    assert json.loads((config["output_dir"] / "validation_state.json").read_text())["next_epoch"] == 8
    progress = [e for e in events if e["type"] == "full_validation" and e.get("status") == "progress"]
    assert all(e["run_id"] and e["pass_id"] and e["request_ids"] for e in progress)


def test_resume_retains_plan_selection_schedule_rng_and_no_stale_requests(tmp_path, monkeypatch):
    config = training_config(tmp_path, epochs=3)
    config["validation"] = {"full_every": 2}
    controller = FullValidationController()
    controller.request_full_validation("original_run")
    full_epochs = []
    def full(*args, **kwargs):
        full_epochs.append(args[3])
        return {"mean_dice": 0.0, "per_case_dice": {"case": 0.0}}
    monkeypatch.setattr(trainer, "full_validation", full)
    class Interruption(RuntimeError):
        pass
    def stop(event):
        if event["type"] == "full_validation" and event.get("status") == "completed":
            raise Interruption()
    with pytest.raises(Interruption):
        trainer.train(config, task=stop, control=controller)
    assert full_epochs == [1]
    original = config["output_dir"]
    before = (original / "validation_plan.json").read_bytes()
    checkpoint = original / "weights_last.pt"
    monkeypatch.setattr(trainer, "_train", functools.partial(trainer._train, resume_from=checkpoint))
    events = []
    resumed = trainer.train({**config, "output_dir": tmp_path / "resumed"}, task=events.append)
    assert full_epochs == [1, 3]
    assert (tmp_path / "resumed/validation_plan.json").read_bytes() == before
    assert resumed["metrics"]["epoch"] == 3
    assert next(e for e in events if e["type"] == "training_resumed")["augmentation_rng_restored"]
    assert not any(e.get("request_id") == "original_run" for e in events)


def test_resume_rejects_changed_sources(tmp_path):
    with image_reading_session():
        source = pair(tmp_path / "data")
        data, _, _ = make_plan(tmp_path, [source])
        mask = tifffile.imread(source.mask)
        mask.flat[0] = 17
        tifffile.imwrite(source.mask, mask, metadata={"axes": "ZYX"}, photometric="minisblack")
        with pytest.raises(TrainingError, match="sources, geometry"):
            make_plan(tmp_path, [source], resume_path=data.path)


def test_anisotropic_scaled_preview_keeps_original_large_ids(tmp_path):
    with image_reading_session():
        shape = (8, 24, 24)
        source = pair(tmp_path / "data", shape=shape, mask=np.full(shape, 70001, np.uint32))
        def configure(base):
            base.case_spacings = {"case": (2, 1, 0.5)}
            base.target_spacing = (1, 1, 1)
            base.augmentation.instance_scale_enabled = True
            base.augmentation.target_object_diameter_px = 8
            base.instance_sizes = {"case": 4}
        cfg = ValidationConfig(minimum_batches=1, minimum_samples=1, preview_max_bytes=1024**2 * 32)
        data, _, emit = make_plan(tmp_path, [source], task="instance_friendly", patch=(4, 8, 8), options=cfg,
                                  configure=configure)
        model = CountingModel(3)
        _, _, anchors = regular_validation(model, DataLoader(data), data, torch.device("cpu"), 1, weights={},
            focal_gamma=2, focal_alpha=None, preview_count=1, progress_interval=1, emit=emit, check_cancel=lambda: None)
        result = save_enlarged_previews(data, anchors, model, torch.device("cpu"), tmp_path, 1,
                                        PostprocessingConfig(), emit=emit, check_cancel=lambda: None)
        item = json.loads(Path(result["preview_path"]).read_text())["items"][0]
        truth = np.load(item["assets"]["target"]["path"])
        assert truth.dtype == np.uint32 and truth.max() == 70001
        assert item["shape"][0] == 4
        assert item["spacing_grid_shape"] == [16, 24, 12]
        assert item["spacing"] != [1, 1, 1]
        expected_image, expected_mask, expected_valid = data.read_tile(data.samples[item["sample_index"]][0],
            tuple(item["model_grid_origin"]), shape=tuple(item["shape"]))
        np.testing.assert_allclose(np.load(item["assets"]["image"]["path"]), expected_image, atol=1e-6)
        np.testing.assert_array_equal(truth, expected_mask)
        np.testing.assert_array_equal(np.load(item["assets"]["validity"]["path"]), expected_valid)


@pytest.mark.parametrize("task", ["instance_friendly", "multiclass_semantic"])
def test_full_sparse_labels_do_not_relabel_or_reanalyse_sources(tmp_path, monkeypatch, task):
    with image_reading_session():
        shape = (4, 12, 12)
        mask = np.zeros(shape, np.uint32)
        mask[:, 2:5, 2:5] = 70001
        mask[:, 7:10, 7:10] = 40000001
        source = pair(tmp_path / "data", shape=shape, mask=mask)
        data, _, emit = make_plan(tmp_path, [source], task=task, labels=[70001, 40000001], patch=(4, 8, 8))
        data[0]
        data.base.mask_analysis(0)
        monkeypatch.setattr("jdll_unet.annotations.analyze_mask", Mock(side_effect=AssertionError("Repeated analysis")))
        monkeypatch.setattr("jdll_unet.targets.compact_instance_labels", Mock(side_effect=AssertionError("Full relabel")))
        result = full_validation(CountingModel(3), data, torch.device("cpu"), 1, PostprocessingConfig(),
                                 emit=emit, check_cancel=lambda: None)
        assert math.isfinite(result["mean_dice"])
        assert result["evaluated_domains"] == 1


def test_full_shallow_domain_uses_symmetric_padding(tmp_path):
    with image_reading_session():
        source = pair(tmp_path / "data", shape=(4, 8, 8))
        data, _, emit = make_plan(tmp_path, [source], patch=(8, 8, 8))
        class CheckPadding(CountingModel):
            def forward(self, image):
                assert not image[:, :, :2].any()
                assert image[:, :, 2:6].any()
                assert not image[:, :, 6:].any()
                return super().forward(image)
        full_validation(CheckPadding(), data, torch.device("cpu"), 1, PostprocessingConfig(),
                         emit=emit, check_cancel=lambda: None)


def test_full_memory_guard_precedes_allocation_and_model_work(tmp_path, monkeypatch):
    with image_reading_session():
        data, _, emit = make_plan(tmp_path, [pair(tmp_path / "data")])
        monkeypatch.setattr("jdll_unet.volumetric_validation.available_ram", lambda: 1)
        model = CountingModel()
        with pytest.raises(MemoryError, match="Regular checkpoints"):
            full_validation(model, data, torch.device("cpu"), 1, PostprocessingConfig(),
                             emit=emit, check_cancel=lambda: None)
        assert model.calls == 0


def test_failed_preview_publication_does_not_expose_partial_assets(tmp_path, monkeypatch):
    import jdll_unet.validation_previews as previews
    with image_reading_session():
        cfg = ValidationConfig(minimum_batches=1, minimum_samples=1, preview_max_bytes=1024**2)
        data, _, emit = make_plan(tmp_path, [pair(tmp_path / "data")], options=cfg)
        model = CountingModel()
        def generate(epoch):
            return regular_validation(model, DataLoader(data), data, torch.device("cpu"), epoch, weights={},
                focal_gamma=2, focal_alpha=None, preview_count=1, progress_interval=1, emit=emit, check_cancel=lambda: None)[2]
        save_enlarged_previews(data, generate(1), model, torch.device("cpu"), tmp_path, 1, PostprocessingConfig(),
                              emit=emit, check_cancel=lambda: None)
        latest = (tmp_path / "previews/latest.json").read_bytes()
        original = previews.write_json
        def fail_latest(path, payload):
            if path.name == "latest.json":
                raise OSError("publication failed")
            original(path, payload)
        monkeypatch.setattr(previews, "write_json", fail_latest)
        with pytest.raises(OSError, match="publication"):
            save_enlarged_previews(data, generate(2), model, torch.device("cpu"), tmp_path, 2, PostprocessingConfig(),
                                  emit=emit, check_cancel=lambda: None)
        assert (tmp_path / "previews/latest.json").read_bytes() == latest
        assert not (tmp_path / "previews/epoch_0002.json").exists()
        assert not (tmp_path / "previews/volume_epoch_0002").exists()


@pytest.mark.parametrize("cache_bytes", [0, 1024])
def test_repeated_validation_crops_reuse_statistics_after_cache_eviction(tmp_path, monkeypatch, cache_bytes):
    with image_reading_session() as session:
        source = pair(tmp_path / "data")
        session.domain_reader = DomainReader(max_bytes=cache_bytes, session=session)
        cfg = ValidationConfig(minimum_batches=4, minimum_samples=4, preview_max_bytes=1024**2)
        data, _, _ = make_plan(tmp_path, [source], options=cfg)
        data[0]
        monkeypatch.setattr("jdll_unet.dataset.fit_normalization", Mock(side_effect=AssertionError("Repeated fit")))
        monkeypatch.setattr("jdll_unet.dataset.analyze_mask", Mock(side_effect=AssertionError("Repeated analysis")))
        monkeypatch.setattr(session, "pixels", Mock(side_effect=AssertionError("Full-volume pixel read")))
        region = Mock(wraps=session.region)
        monkeypatch.setattr(session, "region", region)
        for _epoch in range(3):
            for index in range(len(data)):
                data[index]
        assert region.call_count > 0
        assert session.domain_reader.bytes <= cache_bytes
        assert len(data.base._normalization_statistics) == 1


def test_optional_validation_does_not_change_training_rng_or_checkpoint(tmp_path):
    config = training_config(tmp_path, epochs=2)
    trainer.train(config)
    baseline = torch.load(config["output_dir"] / "weights_last.pt", weights_only=False)
    result = trainer.train({**config, "output_dir": tmp_path / "with_diagnostics", "preview_count": 1,
                            "validation": {"full_every": 1}})
    diagnostics = torch.load(tmp_path / "with_diagnostics/weights_last.pt", weights_only=False)
    for key in baseline["state_dict"]:
        torch.testing.assert_close(baseline["state_dict"][key], diagnostics["state_dict"][key], rtol=0, atol=0)
    assert baseline["metrics"]["val_metrics"] == diagnostics["metrics"]["val_metrics"]
    assert baseline["metrics"]["val_losses"] == diagnostics["metrics"]["val_losses"]
    assert result["full_validation_failures"] == 0


def test_cancel_full_validation_preserves_regular_checkpoint_and_records_status(tmp_path):
    config = training_config(tmp_path, epochs=2)
    control = FullValidationController()
    control.request_full_validation("cancel_me")
    events = []
    def cancel(event):
        events.append(event)
        if event["type"] == "full_validation" and event.get("status") == "started":
            return False
    result = trainer.train(config, task=cancel, control=control)
    assert result["cancelled"]
    last = torch.load(config["output_dir"] / "weights_last.pt", weights_only=False)
    assert last["epoch"] == 1 and last["metrics"]["validation_selection"]["scope"] == "regular_patches"
    full = json.loads((config["output_dir"] / "full_validation_metrics.json").read_text())["history"]
    assert full[-1]["status"] == "cancelled"
    assert events[-1]["type"] == "full_validation" and events[-1]["status"] == "closed"


def test_simultaneous_requests_coalesce_and_final_late_request_is_not_run():
    controller = FullValidationController()
    events = []
    schedule = FullValidationSchedule(5, controller, lambda kind, **payload: events.append({"type": kind, **payload}))
    controller.request_full_validation("same_boundary")
    assert schedule.boundary(5)["reason"] == "requested_and_periodic"
    schedule.started()
    schedule.finish("completed")
    controller.request_full_validation("too_late")
    schedule.close()
    assert events[-1]["unserved_request_ids"] == ["too_late"]
    assert sum(e["status"] == "started" for e in events) == 1


def test_preview_scaling_uses_only_center_target_for_foreground(tmp_path):
    with image_reading_session():
        mask = np.zeros((3, 32, 32), dtype=np.uint8)
        mask[0] = 1
        source = pair(tmp_path / "data", shape=mask.shape, mask=mask)
        source = replace(source, eligible_centers=(1,))
        cfg = ValidationConfig(minimum_batches=1, minimum_samples=1, preview_max_bytes=1024**2)
        data, _, _ = make_plan(tmp_path, [source], dimensions="2.5d", patch=(8, 8), options=cfg)
        image, target, _ = data[0]
        assert image.shape == (3, 8, 8)
        assert not target["semantic"].any()
        assert data.summary["forced_samples"] == 0
