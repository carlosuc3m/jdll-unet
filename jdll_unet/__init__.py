"""Lightweight JDLL-owned UNet backend."""

from .appose_api import detect_task, infer, train
from .errors import ConfigError, DataFormatError, DatasetError, InferenceError, JdllUnetError, ModelLoadError

__all__ = [
    "ConfigError",
    "DataFormatError",
    "DatasetError",
    "InferenceError",
    "JdllUnetError",
    "ModelLoadError",
    "detect_task",
    "infer",
    "train",
]

__version__ = "0.1.0"
