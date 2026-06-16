"""Appose-facing public API for Java-generated scripts."""

from __future__ import annotations

from typing import Any

from .callbacks import CallbackDispatcher
from .infer import infer as _infer
from .task_detect import detect_task as _detect_task
from .trainer import train as _train


def _emit_error(task: Any, exc: Exception) -> None:
    CallbackDispatcher(task).emit("error", message=str(exc), error_class=exc.__class__.__name__)


def train(config: dict, task: Any = None) -> dict:
    try:
        return _train(config, task=task)
    except Exception as exc:
        _emit_error(task, exc)
        raise


def infer(config: dict, inputs: dict, task: Any = None) -> dict:
    try:
        return _infer(config, inputs=inputs, task=task)
    except Exception as exc:
        _emit_error(task, exc)
        raise


def detect_task(config: dict) -> dict:
    return _detect_task(config)
