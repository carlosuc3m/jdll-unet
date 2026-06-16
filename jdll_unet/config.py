"""Configuration parsing and conservative defaults for JDLL UNet."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from .errors import ConfigError

AUTO = "auto"
SUPPORTED_TASKS = {"auto", "binary_semantic", "multiclass_semantic", "instance_friendly", "classes", "objects"}
SUPPORTED_AUGMENTATION_PROFILES = {"auto", "fast", "light-balanced", "balanced", "strong"}
DEFAULT_LOSS_WEIGHTS = {
    "dice": 1.0,
    "bce": 1.0,
    "cross_entropy": 1.0,
    "boundary": 0.5,
}


def _default_loss_weights() -> dict[str, float]:
    return dict(DEFAULT_LOSS_WEIGHTS)


@dataclass(slots=True)
class NormalizationConfig:
    type: str = "percentile"
    low: float = 1.0
    high: float = 99.8
    eps: float = 1e-6


@dataclass(slots=True)
class PostprocessingConfig:
    threshold: float = 0.5
    min_object_size: int = 0
    fill_holes: bool = False
    connected_components: bool = True


@dataclass(slots=True)
class ArchitectureConfig:
    name: str = "tiny-2d"
    input_channels: int = 1
    output_channels: int = 1
    base_channels: int = 16
    depth: int = 3
    convs_per_level: int = 2
    normalization: str = "batch"
    activation: str = "relu"
    dropout: float = 0.0
    dimensions: str = "2d"
    context_slices: int = 3


@dataclass(slots=True)
class TrainingConfig:
    model_name: str
    output_dir: Path
    dataset_path: Path
    starting_point: str = "scratch"
    base_model: Path | None = None
    architecture: str = "tiny-2d"
    device: str = "cpu"
    epochs: int = 100
    seed: int = 42
    task: str = AUTO
    axes: str = AUTO
    input_channels: int | str = AUTO
    output_classes: int | str = AUTO
    patch_size: tuple[int, int] | str = AUTO
    batch_size: int | str = AUTO
    learning_rate: float | str = AUTO
    optimizer: str = "adamw"
    weight_decay: float = 1e-5
    validation_fraction: float = 0.15
    foreground_oversampling: bool | str = AUTO
    foreground_probability: float | str = AUTO
    augmentation_profile: str = AUTO
    num_workers: int = 0
    mixed_precision: bool | str = AUTO
    save_every_epoch: bool = True
    preview_count: int = 20
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    loss_weights: dict[str, float] = field(default_factory=_default_loss_weights)
    augmentation: dict[str, Any] = field(default_factory=dict)


def _path_or_none(value: Any) -> Path | None:
    if value in (None, "", AUTO):
        return None
    return Path(value)


def _coerce_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"1", "true", "yes", "y", "on"}:
            return True
        if lower in {"0", "false", "no", "n", "off"}:
            return False
    raise ConfigError(f"{name} must be a boolean")


def _auto_or_bool(value: Any, name: str) -> bool | str:
    return AUTO if value == AUTO else _coerce_bool(value, name)


def _auto_or_int(value: Any, name: str) -> int | str:
    if value == AUTO:
        return AUTO
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be 'auto' or an integer") from exc
    return parsed


def _auto_or_float(value: Any, name: str) -> float | str:
    if value == AUTO:
        return AUTO
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be 'auto' or a number") from exc
    return parsed


def _as_tuple2(value: Any) -> tuple[int, int] | str:
    if value == AUTO:
        return AUTO
    if isinstance(value, int):
        parsed = (value, value)
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        parsed = (int(value[0]), int(value[1]))
    else:
        raise ConfigError("patch_size must be 'auto', an int, or a length-2 sequence")
    if parsed[0] <= 0 or parsed[1] <= 0:
        raise ConfigError("patch_size values must be positive")
    return parsed


def _nested_dataclass(cls: type, value: Any):
    if isinstance(value, cls):
        return value
    if value is None:
        return cls()
    if isinstance(value, Mapping):
        valid = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = sorted(set(value) - valid)
        if unknown:
            raise ConfigError(f"Unknown {cls.__name__} field(s): {', '.join(unknown)}")
        return cls(**{k: v for k, v in value.items() if k in valid})
    raise ConfigError(f"Expected mapping for {cls.__name__}")


def _loss_weights(value: Any) -> dict[str, float]:
    if value is None:
        return _default_loss_weights()
    if not isinstance(value, Mapping):
        raise ConfigError("loss_weights must be a mapping")
    unknown = sorted(set(value) - set(DEFAULT_LOSS_WEIGHTS))
    if unknown:
        raise ConfigError(f"Unknown loss weight(s): {', '.join(unknown)}")
    weights = _default_loss_weights()
    for key, raw in value.items():
        parsed = float(raw)
        if parsed < 0:
            raise ConfigError(f"loss weight {key} cannot be negative")
        weights[key] = parsed
    return weights


def _validate_normalization(config: NormalizationConfig) -> None:
    if config.type not in {"percentile", "minmax", "zscore", "none"}:
        raise ConfigError(f"Unsupported normalization type: {config.type}")
    if config.eps <= 0:
        raise ConfigError("normalization.eps must be positive")
    if config.type == "percentile" and not 0 <= config.low < config.high <= 100:
        raise ConfigError("normalization percentile bounds must satisfy 0 <= low < high <= 100")


def _validate_postprocessing(config: PostprocessingConfig) -> None:
    if not 0 <= config.threshold <= 1:
        raise ConfigError("postprocessing.threshold must be in [0, 1]")
    if config.min_object_size < 0:
        raise ConfigError("postprocessing.min_object_size cannot be negative")


def _validate_auto_positive_int(value: int | str, name: str) -> None:
    if value != AUTO and int(value) <= 0:
        raise ConfigError(f"{name} must be positive")


def _validate_auto_positive_float(value: float | str, name: str) -> None:
    if value != AUTO and float(value) <= 0:
        raise ConfigError(f"{name} must be positive")


def parse_training_config(config: Mapping[str, Any] | TrainingConfig) -> TrainingConfig:
    """Parse and validate a training config supplied by Java, JSON, or tests."""

    if isinstance(config, TrainingConfig):
        parsed = config
    else:
        missing = [key for key in ("model_name", "output_dir", "dataset_path") if key not in config]
        if missing:
            raise ValueError(f"Missing required training config fields: {', '.join(missing)}")

        raw = dict(config)
        parsed = TrainingConfig(
            model_name=str(raw["model_name"]),
            output_dir=Path(raw["output_dir"]),
            dataset_path=Path(raw["dataset_path"]),
            starting_point=str(raw.get("starting_point", "scratch")),
            base_model=_path_or_none(raw.get("base_model")),
            architecture=str(raw.get("architecture", "tiny-2d")),
            device=str(raw.get("device", "cpu")),
            epochs=int(raw.get("epochs", 100)),
            seed=int(raw.get("seed", 42)),
            task=str(raw.get("task", AUTO)),
            axes=str(raw.get("axes", AUTO)),
            input_channels=_auto_or_int(raw.get("input_channels", AUTO), "input_channels"),
            output_classes=_auto_or_int(raw.get("output_classes", AUTO), "output_classes"),
            patch_size=_as_tuple2(raw.get("patch_size", AUTO)),
            batch_size=_auto_or_int(raw.get("batch_size", AUTO), "batch_size"),
            learning_rate=_auto_or_float(raw.get("learning_rate", AUTO), "learning_rate"),
            optimizer=str(raw.get("optimizer", "adamw")).lower(),
            weight_decay=float(raw.get("weight_decay", 1e-5)),
            validation_fraction=float(raw.get("validation_fraction", 0.15)),
            foreground_oversampling=_auto_or_bool(raw.get("foreground_oversampling", AUTO), "foreground_oversampling"),
            foreground_probability=_auto_or_float(raw.get("foreground_probability", AUTO), "foreground_probability"),
            augmentation_profile=str(raw.get("augmentation_profile", AUTO)),
            num_workers=int(raw.get("num_workers", 0)),
            mixed_precision=raw.get("mixed_precision", AUTO),
            save_every_epoch=_coerce_bool(raw.get("save_every_epoch", True), "save_every_epoch"),
            preview_count=int(raw.get("preview_count", 20)),
            normalization=_nested_dataclass(NormalizationConfig, raw.get("normalization")),
            postprocessing=_nested_dataclass(PostprocessingConfig, raw.get("postprocessing")),
            loss_weights=_loss_weights(raw.get("loss_weights")),
            augmentation=dict(raw.get("augmentation", {})),
        )

    if not parsed.model_name.strip():
        raise ConfigError("model_name cannot be empty")
    if "/" in parsed.model_name or "\\" in parsed.model_name:
        raise ConfigError("model_name must be a name, not a path")
    if parsed.starting_point not in {"scratch", "fine_tune", "finetune"}:
        raise ConfigError("starting_point must be 'scratch' or 'fine_tune'")
    if parsed.starting_point in {"fine_tune", "finetune"} and parsed.base_model is None:
        raise ConfigError("base_model is required when starting_point is fine_tune")
    if parsed.task not in SUPPORTED_TASKS:
        raise ConfigError(f"Unsupported task: {parsed.task}")
    if parsed.augmentation_profile not in SUPPORTED_AUGMENTATION_PROFILES:
        raise ConfigError(f"Unsupported augmentation_profile: {parsed.augmentation_profile}")
    if parsed.optimizer not in {"adamw", "adam"}:
        raise ConfigError("optimizer must be 'adamw' or 'adam'")
    if parsed.epochs < 1:
        raise ConfigError("epochs must be at least 1")
    if not 0.0 <= parsed.validation_fraction < 1.0:
        raise ConfigError("validation_fraction must be in [0, 1)")
    if parsed.weight_decay < 0:
        raise ConfigError("weight_decay cannot be negative")
    if parsed.num_workers < 0:
        raise ConfigError("num_workers cannot be negative")
    if parsed.preview_count < 0:
        raise ConfigError("preview_count cannot be negative")
    if parsed.foreground_probability != AUTO and not 0 <= float(parsed.foreground_probability) <= 1:
        raise ConfigError("foreground_probability must be in [0, 1]")
    _validate_auto_positive_int(parsed.input_channels, "input_channels")
    _validate_auto_positive_int(parsed.output_classes, "output_classes")
    _validate_auto_positive_int(parsed.batch_size, "batch_size")
    _validate_auto_positive_float(parsed.learning_rate, "learning_rate")
    _validate_normalization(parsed.normalization)
    _validate_postprocessing(parsed.postprocessing)
    architecture_defaults(str(parsed.architecture))
    return parsed


def resolve_device(device: str | None) -> torch.device:
    """Resolve CPU/CUDA/MPS with graceful fallback for Appose-launched scripts."""

    requested = (device or "cpu").lower()
    if requested in {"auto", "acceleration", "accelerated"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested.startswith("cuda"):
        return torch.device(requested if torch.cuda.is_available() else "cpu")
    if requested == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device("cpu")


def architecture_defaults(
    architecture: str,
    input_channels: int = 1,
    output_channels: int = 1,
    normalization: str = "batch",
) -> ArchitectureConfig:
    name = architecture.lower()
    dimensions = "2.5d" if "2.5" in name or "25d" in name else "2d"
    if name.startswith("medium"):
        return ArchitectureConfig(
            name=architecture,
            input_channels=input_channels,
            output_channels=output_channels,
            base_channels=32,
            depth=4,
            normalization=normalization,
            dimensions=dimensions,
        )
    if name.startswith("tiny"):
        return ArchitectureConfig(
            name=architecture,
            input_channels=input_channels,
            output_channels=output_channels,
            base_channels=16,
            depth=3,
            normalization=normalization,
            dimensions=dimensions,
        )
    raise ConfigError(f"Unsupported architecture: {architecture}")


def default_patch_size(architecture: str, image_shape: tuple[int, int] | None = None) -> tuple[int, int]:
    preferred = (128, 128) if architecture.lower().startswith("medium") else (96, 96)
    if image_shape is None:
        return preferred
    return (min(preferred[0], int(image_shape[0])), min(preferred[1], int(image_shape[1])))


def default_batch_size(architecture: str, device: torch.device) -> int:
    if architecture.lower().startswith("medium"):
        return 2 if device.type == "cpu" else 4
    return 4 if device.type == "cpu" else 8


def default_learning_rate(optimizer: str) -> float:
    return 1e-3 if optimizer in {"adam", "adamw"} else 1e-3


def default_mixed_precision(value: bool | str, device: torch.device) -> bool:
    if isinstance(value, bool):
        return value and device.type == "cuda"
    if str(value).lower() == AUTO:
        return device.type == "cuda"
    return str(value).lower() in {"1", "true", "yes"} and device.type == "cuda"


def default_foreground_probability(task: str, architecture: str) -> float:
    if task == "instance_friendly":
        return 0.67
    if architecture.lower().startswith("medium"):
        return 0.5
    return 0.4


def default_augmentation_profile(architecture: str, device: torch.device) -> str:
    if architecture.lower().startswith("medium") and device.type in {"cuda", "mps"}:
        return "balanced"
    if architecture.lower().startswith("medium"):
        return "light-balanced"
    return "fast"


def model_folder_config(
    train_config: TrainingConfig,
    task: str,
    arch: ArchitectureConfig,
    input_axes: str = "yx",
    output_axes: str = "yx",
    label_values: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "format": "jdll-unet",
        "format_version": 1,
        "model_name": train_config.model_name,
        "task": task,
        "architecture": arch.name,
        "architecture_config": asdict(arch),
        "input_axes": input_axes,
        "output_axes": output_axes,
        "input_channels": arch.input_channels,
        "num_classes": arch.output_channels,
        "label_values": label_values,
        "normalization": asdict(train_config.normalization),
        "postprocessing": asdict(train_config.postprocessing),
        "training": to_jsonable(asdict(train_config)),
    }


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(to_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n")
    os.replace(tmp_path, path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc
