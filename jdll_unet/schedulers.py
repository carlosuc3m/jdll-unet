"""Learning-rate scheduling utilities for training."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import torch

from .config import LRSchedulerConfig
from .errors import ConfigError


class LearningRateScheduler:
    """Small stateful scheduler with per-step and per-epoch update hooks."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: LRSchedulerConfig,
        total_steps: int,
        total_epochs: int | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.config = config
        self.type = config.type
        self.total_steps = max(1, int(total_steps))
        self.total_epochs = max(1, int(total_epochs or total_steps))
        self.step_count = 0
        self.epoch_count = 0
        self.best_score: float | None = None
        self.bad_epochs = 0
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.min_lr = float(config.min_lr)
        if self.type != "none" and any(self.min_lr > base_lr for base_lr in self.base_lrs):
            raise ConfigError("lr_scheduler.min_lr cannot exceed the optimizer learning rate")

    @property
    def current_lrs(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    @property
    def current_lr(self) -> float:
        return self.current_lrs[0]

    def config_dict(self) -> dict[str, Any]:
        payload = asdict(self.config)
        payload["type"] = self.type
        payload["step_scope"] = "epoch" if self.type in {"poly", "plateau"} else "batch"
        return payload

    def state_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "total_steps": self.total_steps,
            "step_count": self.step_count,
            "epoch_count": self.epoch_count,
            "total_epochs": self.total_epochs,
            "base_lrs": self.base_lrs,
            "current_lrs": self.current_lrs,
            "best_score": self.best_score,
            "bad_epochs": self.bad_epochs,
            "config": self.config_dict(),
        }

    def step_batch(self) -> None:
        self.step_count += 1
        if self.type == "cosine":
            self._set_lrs(self._scheduled_lrs(self._cosine_decay()))

    def step_epoch(self, score: float) -> bool:
        self.epoch_count += 1
        if self.type == "poly":
            progress = min(self.epoch_count, self.total_epochs) / self.total_epochs
            self._set_lrs(self._scheduled_lrs((1.0 - progress) ** float(self.config.poly_power)))
            return False
        if self.type != "plateau":
            return False
        if self.best_score is None or score > self.best_score + float(self.config.plateau_threshold):
            self.best_score = score
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        if self.bad_epochs <= int(self.config.plateau_patience):
            return False

        self.bad_epochs = 0
        old_lrs = self.current_lrs
        new_lrs = [max(self.min_lr, lr * float(self.config.plateau_factor)) for lr in old_lrs]
        self._set_lrs(new_lrs)
        return any(new_lr < old_lr for old_lr, new_lr in zip(old_lrs, new_lrs, strict=True))

    def _cosine_decay(self) -> float:
        progress = min(self.step_count, self.total_steps) / self.total_steps
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _scheduled_lrs(self, decay: float) -> list[float]:
        return [self.min_lr + (base_lr - self.min_lr) * decay for base_lr in self.base_lrs]

    def _set_lrs(self, learning_rates: list[float]) -> None:
        for group, lr in zip(self.optimizer.param_groups, learning_rates, strict=True):
            group["lr"] = float(lr)
