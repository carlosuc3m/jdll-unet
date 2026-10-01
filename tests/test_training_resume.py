import functools
import json

import numpy as np
import pytest
import tifffile
import torch

from jdll_unet import trainer
from jdll_unet.errors import TrainingError


class InterruptedRun(RuntimeError):
    pass


@pytest.fixture
def interrupted_run(tmp_path):
    images, masks = tmp_path / "data/images", tmp_path / "data/masks"
    images.mkdir(parents=True)
    masks.mkdir()
    mask = np.zeros((32, 32), np.uint8)
    mask[8:24, 8:24] = 1
    for index in range(3):
        tifffile.imwrite(images / f"{index}.tif", mask.astype(np.float32))
        tifffile.imwrite(masks / f"{index}.tif", mask)
    config = {"model_name": "resume", "dataset_path": images.parent, "output_dir": tmp_path / "original",
              "architecture": "tiny-2d", "device": "cpu", "epochs": 3, "steps_per_epoch": 1,
              "patch_size": [32, 32], "batch_size": 1, "effective_batch_size": 1,
              "task": "binary_semantic", "preview_count": 0}

    def stop(payload):
        if payload.get("message") == "UNet validation epoch 1":
            raise InterruptedRun()

    with pytest.raises(InterruptedRun):
        trainer.train(config, task=stop)
    return config


def test_completed_epoch_resume_retains_optimizer_schedule_history_and_best(interrupted_run, tmp_path, monkeypatch):
    config = interrupted_run
    original = config["output_dir"]
    checkpoint = original / "weights_last.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    before = checkpoint.read_bytes()
    output = tmp_path / "resumed"
    events = []
    original_step = torch.optim.AdamW.step
    previous_steps = []

    def step(optimizer, *args, **kwargs):
        previous_steps.append(next(iter(optimizer.state.values()))["step"].item())
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    monkeypatch.setattr(trainer, "_train", functools.partial(trainer._train, resume_from=checkpoint))
    trainer.train({**config, "output_dir": output, "data_cache_mb": 0}, task=events.append)
    result = torch.load(output / "weights_last.pt", map_location="cpu", weights_only=False)
    assert previous_steps == [1, 2]
    assert checkpoint.read_bytes() == before
    assert result["epoch"] == 3
    assert result["scheduler_state_dict"]["step_count"] == 3
    assert result["scheduler_state_dict"]["epoch_count"] == 3
    assert result["scheduler_state_dict"]["current_lrs"] == [0.0]
    assert result["scheduler_state_dict"]["base_lrs"] == saved["scheduler_state_dict"]["base_lrs"]
    history = json.loads((output / "metrics.json").read_text())
    assert [row["epoch"] for row in history["history"]] == [1, 2, 3]
    assert history["history"][0] == saved["metrics"]
    best = torch.load(output / "weights_best.pt", map_location="cpu", weights_only=False)
    assert best["metrics"]["val_metrics"]["dice"] == history["best_score"]
    resumed = next(event for event in events if event["type"] == "training_resumed")
    assert resumed["step"] == 1
    assert resumed["learning_rate"] == saved["scheduler_state_dict"]["current_lrs"][0]


@pytest.mark.parametrize("invalid", ["settings", "unfinished", "history"])
def test_resume_rejects_incompatible_or_incomplete_state(interrupted_run, tmp_path, monkeypatch, invalid):
    config = dict(interrupted_run)
    checkpoint = config["output_dir"] / "weights_last.pt"
    if invalid == "settings":
        config["effective_batch_size"] = 2
        message = "changed training settings"
    elif invalid == "unfinished":
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state["metrics"]["validation_pending"] = True
        torch.save(state, checkpoint)
        message = "completed, validated epoch"
    else:
        (config["output_dir"] / "metrics.json").write_text(json.dumps({"history": [], "best_score": 0}))
        message = "metric history"
    monkeypatch.setattr(trainer, "_train", functools.partial(trainer._train, resume_from=checkpoint))
    with pytest.raises(TrainingError, match=message):
        trainer.train({**config, "output_dir": tmp_path / "resumed"})
