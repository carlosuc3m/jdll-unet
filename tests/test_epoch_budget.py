import json
import math

import numpy as np
import pytest
import tifffile
import torch

from jdll_unet import trainer
from jdll_unet.config import TrainingConfig, parse_training_config, resolve_steps_per_epoch
from jdll_unet.errors import ConfigError


@pytest.fixture
def request_config(tmp_path):
    return {
        "model_name": "epoch-budget",
        "output_dir": tmp_path / "model",
        "dataset_path": tmp_path / "data",
        "architecture": "resenc-tiny-2d",
    }


@pytest.mark.parametrize("prefix", ["", "resenc-", "residual-"])
@pytest.mark.parametrize("dimensions", ["2d", "2.5d", "3d"])
@pytest.mark.parametrize("batch", [4, 16, 32])
def test_small_patch_budget(request_config, prefix, dimensions, batch):
    config = parse_training_config({**request_config, "architecture": f"{prefix}tiny-{dimensions}"})
    for cases in (1, 100, 150, 1001):
        patches = max(1000, 10 * cases)
        steps = resolve_steps_per_epoch(config, config.architecture, cases, batch)
        assert steps == math.ceil(patches / batch)
        assert patches <= steps * batch < patches + batch


@pytest.mark.parametrize("preset", ["medium", "big", "large"])
@pytest.mark.parametrize("dimensions", ["2d", "2.5d", "3d"])
def test_larger_presets_keep_step_budget(request_config, preset, dimensions):
    config = parse_training_config({**request_config, "architecture": f"resenc-{preset}-{dimensions}"})
    for batch in (4, 16, 32):
        for cases in (1, 150, 1001):
            assert resolve_steps_per_epoch(config, config.architecture, cases, batch) == max(
                250, math.ceil(10 * cases / batch)
            )


@pytest.mark.parametrize("as_dataclass", [False, True])
def test_patch_budget_options_and_roundtrip(request_config, as_dataclass):
    request = {**request_config, "minimum_patches_per_epoch": 65, "expected_patches_per_case": 20}
    config = parse_training_config(TrainingConfig(**request) if as_dataclass else request)
    for parsed in (config, parse_training_config(config)):
        assert parsed.minimum_steps_per_epoch is None
        assert resolve_steps_per_epoch(parsed, parsed.architecture, 1, 16) == 5
        assert resolve_steps_per_epoch(parsed, parsed.architecture, 10, 16) == 13


@pytest.mark.parametrize("architecture", ["resenc-tiny-2d", "resenc-medium-2d"])
@pytest.mark.parametrize("as_dataclass", [False, True])
def test_explicit_steps_and_legacy_minima_remain_overrides(request_config, architecture, as_dataclass):
    request = {**request_config, "architecture": architecture, "minimum_steps_per_epoch": 250}
    config = parse_training_config(TrainingConfig(**request) if as_dataclass else request)
    assert resolve_steps_per_epoch(config, architecture, 150, 32) == 250
    for parsed in (config, parse_training_config(config)):
        assert parsed.minimum_steps_per_epoch == 250
    config.steps_per_epoch = 7
    assert resolve_steps_per_epoch(config, architecture, 150, 32) == 7


def test_custom_minimum_step_floor(request_config):
    config = parse_training_config({**request_config, "minimum_steps_per_epoch": 50})
    assert resolve_steps_per_epoch(config, config.architecture, 150, 32) == 50
    config.architecture = "resenc-medium-2d"
    assert resolve_steps_per_epoch(config, config.architecture, 150, 32) == 50


@pytest.mark.parametrize("field", ["minimum_patches_per_epoch", "minimum_steps_per_epoch"])
@pytest.mark.parametrize("value", [0, -1])
def test_budget_validation(request_config, field, value):
    with pytest.raises(ConfigError, match="positive"):
        parse_training_config({**request_config, field: value})


@pytest.mark.parametrize(
    "device,microbatch,effective,steps",
    [
        ("cpu", "auto", 16, 5),
        ("cpu", 4, 16, 5),
        pytest.param(
            "cuda", "auto", 32, 3,
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
        ),
    ],
)
def test_auto_budget_controls_training_and_exports(request_config, monkeypatch, device, microbatch, effective, steps):
    root = request_config["dataset_path"]
    for folder in ("images", "masks"):
        (root / folder).mkdir(parents=True)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 1
    for index in range(2):
        tifffile.imwrite(root / "images" / f"{index}.tif", mask * 200)
        tifffile.imwrite(root / "masks" / f"{index}.tif", mask)
    datasets = []
    make_dataset = trainer.make_dataset

    def record_dataset(*args, **kwargs):
        dataset = make_dataset(*args, **kwargs)
        if dataset.training:
            datasets.append(dataset)
        return dataset

    monkeypatch.setattr(trainer, "make_dataset", record_dataset)
    events = []
    result = trainer.train(
        {
            **request_config,
            "device": device,
            "epochs": 2,
            "patch_size": [16, 16],
            "batch_size": microbatch,
            "minimum_patches_per_epoch": 65,
            "task": "binary_semantic",
            "preview_count": 0,
            "progress_update_interval": 1,
            "validation": {"mode": "light", "light_steps": 1},
        },
        task=events.append,
    )
    exported = json.loads(json.dumps(result["config"]["training"]))
    assert exported["steps_per_epoch"] == steps
    assert exported["minimum_patches_per_epoch"] == 65
    assert exported["minimum_steps_per_epoch"] is None
    assert exported["effective_batch_size"] == effective
    assert exported["accumulation_steps"] == (4 if microbatch == 4 else 1)
    assert len(datasets) == 1 and len(datasets[0]) == steps * effective
    plan = next(event for event in events if event["type"] == "training_plan")
    assert plan["steps_per_epoch"] == steps
    progress = [event for event in events if event["type"] == "progress" and "train/total_loss" in event.get("losses", {})]
    assert [event["current"] for event in progress] == list(range(1, 2 * steps + 1))
    assert all(event["maximum"] == 2 * steps for event in progress)
    for setting in (steps, "auto"):
        parsed = parse_training_config({**exported, "steps_per_epoch": setting})
        assert resolve_steps_per_epoch(parsed, parsed.architecture, 1, effective) == steps
