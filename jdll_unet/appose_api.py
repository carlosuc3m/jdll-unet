"""Appose-facing public API for Java-generated scripts."""

from __future__ import annotations

from typing import Any

from .infer import infer as _infer
from .task_detect import detect_task as _detect_task
from .trainer import train as _train


def train(config: dict, task: Any = None) -> dict:
    return _train(config, task=task)


def infer(config: dict, inputs: dict, task: Any = None) -> dict:
    return _infer(config, inputs=inputs, task=task)


def detect_task(config: dict) -> dict:
    return _detect_task(config)
