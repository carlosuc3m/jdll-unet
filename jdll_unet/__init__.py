"""Lightweight JDLL-owned UNet backend."""

from .appose_api import detect_task, infer, train

__all__ = ["detect_task", "infer", "train"]

__version__ = "0.1.0"
