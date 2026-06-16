import pytest
import torch

from jdll_unet.config import LRSchedulerConfig, parse_training_config
from jdll_unet.errors import ConfigError
from jdll_unet.schedulers import LearningRateScheduler


def _optimizer(lr: float = 0.01) -> torch.optim.Optimizer:
    return torch.optim.AdamW(torch.nn.Linear(1, 1).parameters(), lr=lr)


def test_poly_scheduler_decays_to_min_lr():
    optimizer = _optimizer(lr=0.01)
    scheduler = LearningRateScheduler(
        optimizer,
        LRSchedulerConfig(type="poly", min_lr=0.001, poly_power=1.0),
        total_steps=4,
    )

    scheduler.step_batch()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.00775)
    scheduler.step_batch()
    scheduler.step_batch()
    scheduler.step_batch()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.001)


def test_cosine_scheduler_decays_to_min_lr():
    optimizer = _optimizer(lr=0.01)
    scheduler = LearningRateScheduler(
        optimizer,
        LRSchedulerConfig(type="cosine", min_lr=0.001),
        total_steps=2,
    )

    scheduler.step_batch()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0055)
    scheduler.step_batch()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.001)


def test_plateau_scheduler_reduces_after_patience():
    optimizer = _optimizer(lr=0.01)
    scheduler = LearningRateScheduler(
        optimizer,
        LRSchedulerConfig(type="plateau", min_lr=0.001, plateau_factor=0.5, plateau_patience=0),
        total_steps=10,
    )

    assert scheduler.step_epoch(0.5) is False
    assert scheduler.step_epoch(0.5) is True
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.005)
    assert scheduler.step_epoch(0.6) is False
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.005)


def test_scheduler_config_defaults_aliases_and_validation(tmp_path):
    base = {
        "model_name": "model",
        "output_dir": str(tmp_path / "out"),
        "dataset_path": str(tmp_path / "data"),
    }

    assert parse_training_config(base).lr_scheduler.type == "poly"
    assert parse_training_config({**base, "lr_scheduler": "cosine"}).lr_scheduler.type == "cosine"
    assert parse_training_config({**base, "learning_rate_scheduler": {"type": "constant"}}).lr_scheduler.type == "none"

    with pytest.raises(ConfigError, match="lr_scheduler.poly_power"):
        parse_training_config({**base, "lr_scheduler": {"type": "poly", "poly_power": 0}})
