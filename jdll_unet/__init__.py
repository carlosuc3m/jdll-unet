"""Lightweight JDLL-owned UNet backend."""

from .appose_api import detect_task, infer, train
from .callbacks import CallbackDispatcher, CallbackEvent
from .errors import (
    ConfigError,
    DataFormatError,
    DatasetError,
    InferenceCancelled,
    InferenceError,
    JdllUnetError,
    ModelLoadError,
)
from .validation_control import FullValidationController

__all__ = [
    "CallbackDispatcher",
    "CallbackEvent",
    "FullValidationController",
    "ConfigError",
    "DataFormatError",
    "DatasetError",
    "InferenceCancelled",
    "InferenceError",
    "JdllUnetError",
    "ModelLoadError",
    "detect_task",
    "infer",
    "train",
]

__version__ = "0.1.0"
