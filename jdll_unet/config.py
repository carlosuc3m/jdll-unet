"""Configuration parsing and conservative defaults for JDLL UNet."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import MISSING, asdict, dataclass, field, fields
from numbers import Real
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy as np
import torch

from .errors import ConfigError

AUTO = "auto"
T = TypeVar("T")
SUPPORTED_TASKS = {"auto", "binary_semantic", "multiclass_semantic", "instance_friendly", "classes", "objects"}
SUPPORTED_AUGMENTATION_PROFILES = {"auto", "fast", "light-balanced", "balanced", "strong"}
SUPPORTED_LR_SCHEDULERS = {"poly", "cosine", "plateau", "none", "constant"}
SUPPORTED_MODEL_NORMALIZATIONS = {"group", "instance", "batch", "none", "identity"}
DEFAULT_LOSS_WEIGHTS = {
    "dice": 1.0,
    "bce": 1.0,
    "cross_entropy": 1.0,
    "focal": 0.0,
    "boundary": 0.5,
    "boundary_focal": 0.0,
    "distance": 1.0,
    "distance_background": 0.05,
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
    method: str = "distance_boundary_watershed"
    seed_distance_threshold: float = 0.35
    seed_boundary_threshold: float = 0.5
    seed_h: float = 0.1
    min_seed_size: int = 3
    boundary_weight: float = 1.0
    connectivity: str = "face"
    min_object_size_physical: float | None = None
    min_seed_size_physical: float | None = None


@dataclass(slots=True)
class LRSchedulerConfig:
    type: str = "poly"
    min_lr: float = 0.0
    poly_power: float = 0.9
    plateau_factor: float = 0.5
    plateau_patience: int = 5
    plateau_threshold: float = 1e-4

    @property
    def step_scope(self) -> str:
        return "epoch" if self.type in {"poly", "plateau"} else "batch"


@dataclass(slots=True)
class AnnotationPreparationConfig:
    repair_disconnected_instances: bool = True
    ram_cache_mb: float = 128.0
    cache_dir: str | None = None
    disk_reserve_mb: float = 256.0
    warning_fraction: float = 0.1


@dataclass(slots=True)
class InstanceScaleNormalizationConfig:
    enabled: bool = True
    target_object_fraction: float = 0.25
    object_size_measure: str = "equivalent_sphere_diameter"
    max_instances_per_image: int = 21
    exclude_border_instances: bool = True
    min_instance_area: int = 4
    training_scale_jitter: tuple[float, float] = (0.5, 2.0)
    jitter_distribution: str = "log_uniform"
    min_effective_scale: float = 0.25
    max_effective_scale: float = 4.0


@dataclass(slots=True)
class SpacingConfig:
    default_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    known_fraction_threshold: float = 0.5
    target_spacing: str | tuple[float, float, float] = AUTO
    anisotropy_threshold: float = 3.0
    kernel_anisotropy_threshold: float = 2.0
    max_upsampling: float = 3.0
    minimum_feature_map_size: int = 4


@dataclass(slots=True)
class ContextConfig:
    stride_policy: str = "nearest_physical"
    stride: int = 1
    spacing: str | float = AUTO


@dataclass(slots=True)
class ValidationConfig:
    mode: str = "full"
    light_every: int = 1
    full_every: int | None = None
    light_steps: int = 50
    early_stopping_patience: int | None = None
    minimum_batches: int = 50
    minimum_samples: int = 100
    foreground_fraction: float = 0.33
    minimum_foreground: float = 0.01
    minimum_source_fraction: float = 0.5
    max_sampling_overlap: float = 0.10
    candidate_attempts: int = 16
    preview_max_bytes: int | str = AUTO
    tile_overlap: float = 0.25
    tile_blending: str = "constant"

    def __post_init__(self) -> None:
        if self.light_steps != self.minimum_batches:
            if self.light_steps == 50:
                self.light_steps = self.minimum_batches
            elif self.minimum_batches == 50:
                self.minimum_batches = self.light_steps
            else:
                raise ConfigError("validation.light_steps and minimum_batches must agree")


def _validation_config(value: Any) -> ValidationConfig:
    if isinstance(value, Mapping):
        value = dict(value)
        for name in ("light_steps", "minimum_batches"):
            if value.get(name) == AUTO:
                value.pop(name)
        if "light_steps" in value:
            if "minimum_batches" in value and value["minimum_batches"] != value["light_steps"]:
                raise ConfigError("validation.light_steps and minimum_batches must agree when both are supplied")
            value.setdefault("minimum_batches", value["light_steps"])
    result = _nested_dataclass(ValidationConfig, value)
    result.light_steps = result.minimum_batches
    return result


def resolve_validation_config(config: TrainingConfig, dimensions: str, preset: str) -> None:
    """Resolve dimensional defaults after automatic architecture planning."""
    validation = config.validation
    volumetric = dimensions in {"2.5d", "3d"}
    if validation.full_every is None:
        validation.full_every = 0 if volumetric or validation.mode == "light" else 5
    if validation.early_stopping_patience is None:
        validation.early_stopping_patience = 0 if volumetric else 20
    if volumetric and validation.light_every != 1:
        raise ConfigError("Volumetric regular validation runs every epoch; validation.light_every must be 1")
    if volumetric and validation.early_stopping_patience:
        raise ConfigError("Volumetric early stopping is disabled; set validation.early_stopping_patience=0 and use cancellation")
    if volumetric and validation.mode == "light" and validation.full_every:
        raise ConfigError("validation.mode='light' conflicts with a positive full_every; use mode='full' or full_every=0")
    if validation.preview_max_bytes == AUTO:
        validation.preview_max_bytes = (600 if any(part in {"big", "large"} for part in preset.split("-")) else 256) * 1024**2
    if volumetric and "preview_count" not in config._provided_fields:
        config.preview_count = 4


@dataclass(slots=True)
class ArchitectureConfig:
    name: str = "resenc-tiny-2d"
    input_channels: int = 1
    output_channels: int = 1
    base_channels: int = 16
    depth: int = 3
    convs_per_level: int = 2
    normalization: str = "group"
    activation: str = "relu"
    dropout: float = 0.0
    dimensions: str = "2d"
    context_slices: int = 3
    block_type: str = "residual"
    deep_supervision: bool = False
    channels: tuple[int, ...] = ()
    encoder_blocks: tuple[int, ...] = ()
    kernels: tuple[tuple[int, ...], ...] = ()
    strides: tuple[tuple[int, ...], ...] = ()
    reference_memory_gb: int = 4
    normalize_projection: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.normalize_projection, bool):
            raise ConfigError("normalize_projection must be a boolean")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ArchitectureConfig:
        """Load saved architecture metadata without changing legacy checkpoint behavior."""
        values = dict(payload)
        values.setdefault("normalize_projection", False)
        return cls(**values)


@dataclass(slots=True, init=False)
class TrainingConfig:
    model_name: str
    output_dir: Path
    dataset_path: Path
    starting_point: str = "scratch"
    base_model: Path | None = None
    architecture: str = AUTO
    device: str = "cpu"
    epochs: int = 100
    seed: int = 42
    task: str = AUTO
    axes: str = AUTO
    input_channels: int | str = AUTO
    output_classes: int | str = AUTO
    model_normalization: str = "group"
    patch_size: tuple[int, ...] | str = AUTO
    batch_size: int | str = AUTO
    learning_rate: float | str = AUTO
    optimizer: str = "adamw"
    weight_decay: float = 1e-5
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    annotation_preparation: AnnotationPreparationConfig = field(default_factory=AnnotationPreparationConfig)
    instance_scale_normalization: InstanceScaleNormalizationConfig = field(
        default_factory=InstanceScaleNormalizationConfig
    )
    validation_fraction: float = 0.15
    foreground_oversampling: bool | str = AUTO
    foreground_probability: float | str = AUTO
    skip_empty_images: bool = True
    skip_empty_patches: bool = True
    empty_patch_fraction: float = 0.0
    empty_patch_max_retries: int = 8
    include_empty_patches_after_max_retries: bool = False
    max_padding_ratio: float = 1.0
    max_empty_plane_fraction: float = 0.20
    augmentation_profile: str = AUTO
    num_workers: int = 0
    data_cache_mb: float | str = AUTO
    mixed_precision: bool | str = AUTO
    deep_supervision: bool | str = AUTO
    context_slices: int | str = AUTO
    context: ContextConfig = field(default_factory=ContextConfig)
    spacing: SpacingConfig = field(default_factory=SpacingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    effective_batch_size: int = 4
    steps_per_epoch: int | str = AUTO
    minimum_steps_per_epoch: int | None = None
    expected_patches_per_case: int = 10
    memory_fraction: float = 0.8
    focal_gamma: float = 2.0
    focal_alpha: float | None = None
    auto_focal: bool = False
    auto_focal_foreground_threshold: float = 0.05
    auto_focal_boundary_threshold: float = 0.02
    auto_focal_weight: float = 0.5
    auto_boundary_focal_weight: float = 0.25
    auto_focal_sample_limit: int = 64
    progress_update_interval: int | str = AUTO
    log_update_interval: int | str = AUTO
    save_every_epoch: bool = True
    preview_count: int = 20
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    loss_weights: dict[str, float] = field(default_factory=_default_loss_weights)
    augmentation: dict[str, Any] = field(default_factory=dict)
    minimum_patches_per_epoch: int = 1000
    _provided_fields: frozenset[str] = field(default_factory=frozenset, init=False, repr=False, compare=False)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        options = [option for option in fields(type(self)) if option.init]
        if len(args) > len(options):
            raise TypeError("Too many positional TrainingConfig arguments")
        supplied = dict(kwargs)
        for option, value in zip(options, args, strict=False):
            if option.name in supplied:
                raise TypeError(f"Multiple values for {option.name}")
            supplied[option.name] = value
        unknown = set(supplied) - {option.name for option in options}
        if unknown:
            raise TypeError(f"Unknown TrainingConfig fields: {sorted(unknown)}")
        self._provided_fields = frozenset(supplied)
        for option in options:
            if option.name in supplied:
                value = supplied[option.name]
            elif option.default is not MISSING:
                value = option.default
            elif option.default_factory is not MISSING:
                value = option.default_factory()
            else:
                raise TypeError(f"Missing required TrainingConfig field: {option.name}")
            setattr(self, option.name, value)

    def request_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values.pop("_provided_fields")
        inheritable = {
            "architecture",
            "model_normalization",
            "normalization",
            "deep_supervision",
            "context_slices",
            "context",
            "spacing",
            "instance_scale_normalization",
            "effective_batch_size",
            "preview_count",
        }
        for option in fields(type(self)):
            if option.name in inheritable and option.name not in self._provided_fields:
                default = option.default_factory() if option.default_factory is not MISSING else option.default
                if getattr(self, option.name) == default:
                    values.pop(option.name, None)
        return values


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


def _optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", AUTO}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be null, 'auto', or a number") from exc


def _as_spatial_tuple(value: Any) -> tuple[int, ...] | str:
    if value == AUTO:
        return AUTO
    if isinstance(value, int):
        parsed: tuple[int, ...] = (value, value)
    elif isinstance(value, (list, tuple)) and len(value) in {2, 3}:
        parsed = tuple(int(item) for item in value)
    else:
        raise ConfigError("patch_size must be 'auto', an int, or a length-2 or length-3 sequence")
    if any(item <= 0 for item in parsed):
        raise ConfigError("patch_size values must be positive")
    return parsed


def _nested_dataclass(cls: type[T], value: Any) -> T:
    if isinstance(value, cls):
        return value
    if value is None or value == AUTO:
        return cls()
    if isinstance(value, Mapping):
        valid = {item.name for item in fields(cast(Any, cls))}
        unknown = sorted(set(value) - valid)
        if unknown:
            raise ConfigError(f"Unknown {cls.__name__} field(s): {', '.join(unknown)}")
        defaults = cls()
        return cls(**{k: getattr(defaults, k) if v == AUTO else v for k, v in value.items() if k in valid})
    raise ConfigError(f"Expected mapping for {cls.__name__}")


def _lr_scheduler_config(value: Any) -> LRSchedulerConfig:
    if value is None or value == AUTO:
        return LRSchedulerConfig()
    if isinstance(value, LRSchedulerConfig):
        return value
    if isinstance(value, str):
        return LRSchedulerConfig(type=value)
    if isinstance(value, Mapping):
        valid = {field.name for field in LRSchedulerConfig.__dataclass_fields__.values()}
        unknown = sorted(set(value) - valid - {"step_scope"})
        if unknown:
            raise ConfigError(f"Unknown LRSchedulerConfig field(s): {', '.join(unknown)}")
        parsed = LRSchedulerConfig(**{key: value[key] for key in value if key in valid})
        _validate_lr_scheduler(parsed)
        if "step_scope" in value and value["step_scope"] != parsed.step_scope:
            raise ConfigError(f"lr_scheduler.step_scope must be {parsed.step_scope!r} for {parsed.type!r}")
        return parsed
    raise ConfigError("lr_scheduler must be a string or mapping")


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


def _model_normalization(value: Any) -> str:
    normalization = str(value).lower()
    if normalization == AUTO:
        normalization = "group"
    if normalization not in SUPPORTED_MODEL_NORMALIZATIONS:
        raise ConfigError(f"Unsupported model_normalization: {normalization}")
    return "none" if normalization == "identity" else normalization


def _validate_postprocessing(config: PostprocessingConfig) -> None:
    if not 0 <= config.threshold <= 1:
        raise ConfigError("postprocessing.threshold must be in [0, 1]")
    if config.min_object_size < 0:
        raise ConfigError("postprocessing.min_object_size cannot be negative")
    if config.method not in {"distance_boundary_watershed", "connected_components"}:
        raise ConfigError("Unsupported postprocessing.method")
    if config.connectivity not in {"face", "full"}:
        raise ConfigError("postprocessing.connectivity must be 'face' or 'full'")
    for name in ("seed_distance_threshold", "seed_boundary_threshold", "seed_h"):
        if not 0 <= float(getattr(config, name)) <= 1:
            raise ConfigError(f"postprocessing.{name} must be in [0, 1]")
    if config.min_seed_size < 1 or config.boundary_weight < 0:
        raise ConfigError("postprocessing seed size and boundary weight are invalid")
    for name in ("min_object_size_physical", "min_seed_size_physical"):
        value = getattr(config, name)
        if value is not None and float(value) < 0:
            raise ConfigError(f"postprocessing.{name} cannot be negative")


def _validate_lr_scheduler(config: LRSchedulerConfig) -> None:
    config.type = str(config.type).lower()
    if config.type not in SUPPORTED_LR_SCHEDULERS:
        raise ConfigError(f"Unsupported lr_scheduler.type: {config.type}")
    if config.type == "constant":
        config.type = "none"
    try:
        config.min_lr = float(config.min_lr)
        config.poly_power = float(config.poly_power)
        config.plateau_factor = float(config.plateau_factor)
        config.plateau_patience = int(config.plateau_patience)
        config.plateau_threshold = float(config.plateau_threshold)
    except (TypeError, ValueError) as exc:
        raise ConfigError("lr_scheduler numeric options must be valid numbers") from exc
    if config.min_lr < 0:
        raise ConfigError("lr_scheduler.min_lr cannot be negative")
    if config.poly_power <= 0:
        raise ConfigError("lr_scheduler.poly_power must be positive")
    if not 0 < config.plateau_factor < 1:
        raise ConfigError("lr_scheduler.plateau_factor must be in (0, 1)")
    if config.plateau_patience < 0:
        raise ConfigError("lr_scheduler.plateau_patience cannot be negative")
    if config.plateau_threshold < 0:
        raise ConfigError("lr_scheduler.plateau_threshold cannot be negative")


def _validate_auto_positive_int(value: int | str, name: str) -> None:
    if value != AUTO and int(value) <= 0:
        raise ConfigError(f"{name} must be positive")


def _validate_auto_positive_float(value: float | str, name: str) -> None:
    if value != AUTO and float(value) <= 0:
        raise ConfigError(f"{name} must be positive")


def _max_empty_plane_fraction(value: Any, name: str = "max_empty_plane_fraction") -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConfigError(f"{name} must be a finite number in [0, 1), not a boolean")
    if not 0 <= value < 1 or not math.isfinite(value):
        raise ConfigError(f"{name} must be finite and in [0, 1)")
    return float(value)


def parse_training_config(
    config: Mapping[str, Any] | TrainingConfig, *, source_architecture: ArchitectureConfig | None = None
) -> TrainingConfig:
    """Parse and validate a training config supplied by Java, JSON, or tests."""

    if isinstance(config, TrainingConfig):
        return parse_training_config(config.request_dict(), source_architecture=source_architecture)
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
            architecture=str(raw.get("architecture", AUTO)),
            device=str(raw.get("device", "cpu")),
            epochs=int(raw.get("epochs", 100)),
            seed=int(raw.get("seed", 42)),
            task=str(raw.get("task", AUTO)),
            axes=str(raw.get("axes", AUTO)),
            input_channels=_auto_or_int(raw.get("input_channels", AUTO), "input_channels"),
            output_classes=_auto_or_int(raw.get("output_classes", AUTO), "output_classes"),
            model_normalization=_model_normalization(
                raw.get(
                    "model_normalization",
                    raw.get("network_normalization", raw.get("architecture_normalization", "group")),
                )
            ),
            patch_size=_as_spatial_tuple(raw.get("patch_size", AUTO)),
            batch_size=_auto_or_int(raw.get("batch_size", AUTO), "batch_size"),
            learning_rate=_auto_or_float(raw.get("learning_rate", AUTO), "learning_rate"),
            optimizer=str(raw.get("optimizer", "adamw")).lower(),
            weight_decay=float(raw.get("weight_decay", 1e-5)),
            lr_scheduler=_lr_scheduler_config(
                raw.get("lr_scheduler", raw.get("learning_rate_scheduler", raw.get("scheduler")))
            ),
            instance_scale_normalization=_nested_dataclass(
                InstanceScaleNormalizationConfig, raw.get("instance_scale_normalization")
            ),
            annotation_preparation=_nested_dataclass(
                AnnotationPreparationConfig, raw.get("annotation_preparation")
            ),
            validation_fraction=float(raw.get("validation_fraction", 0.15)),
            foreground_oversampling=_auto_or_bool(raw.get("foreground_oversampling", AUTO), "foreground_oversampling"),
            foreground_probability=_auto_or_float(raw.get("foreground_probability", AUTO), "foreground_probability"),
            skip_empty_images=_coerce_bool(raw.get("skip_empty_images", True), "skip_empty_images"),
            skip_empty_patches=_coerce_bool(raw.get("skip_empty_patches", True), "skip_empty_patches"),
            empty_patch_fraction=_max_empty_plane_fraction(raw.get("empty_patch_fraction", 0.0), "empty_patch_fraction"),
            empty_patch_max_retries=int(raw.get("empty_patch_max_retries", 8)),
            include_empty_patches_after_max_retries=_coerce_bool(
                raw.get("include_empty_patches_after_max_retries", False),
                "include_empty_patches_after_max_retries",
            ),
            augmentation_profile=str(raw.get("augmentation_profile", AUTO)),
            max_padding_ratio=float(raw.get("max_padding_ratio", 1.0)),
            max_empty_plane_fraction=_max_empty_plane_fraction(raw.get("max_empty_plane_fraction", 0.20)),
            num_workers=int(raw.get("num_workers", 0)),
            data_cache_mb=_auto_or_float(raw.get("data_cache_mb", AUTO), "data_cache_mb"),
            mixed_precision=raw.get("mixed_precision", AUTO),
            deep_supervision=_auto_or_bool(raw.get("deep_supervision", AUTO), "deep_supervision"),
            context_slices=_auto_or_int(
                AUTO if raw.get("context_slices") is None else raw["context_slices"], "context_slices"
            ),
            context=_nested_dataclass(ContextConfig, raw.get("context")),
            spacing=_nested_dataclass(SpacingConfig, raw.get("spacing")),
            validation=_validation_config(raw.get("validation")),
            effective_batch_size=int(raw.get("effective_batch_size", 4)),
            steps_per_epoch=_auto_or_int(raw.get("steps_per_epoch", AUTO), "steps_per_epoch"),
            minimum_steps_per_epoch=(
                int(raw["minimum_steps_per_epoch"]) if raw.get("minimum_steps_per_epoch") is not None else None
            ),
            minimum_patches_per_epoch=int(raw.get("minimum_patches_per_epoch", 1000)),
            expected_patches_per_case=int(raw.get("expected_patches_per_case", 10)),
            memory_fraction=float(raw.get("memory_fraction", 0.8)),
            focal_gamma=float(raw.get("focal_gamma", 2.0)),
            focal_alpha=_optional_float(raw.get("focal_alpha"), "focal_alpha"),
            auto_focal=_coerce_bool(raw.get("auto_focal", False), "auto_focal"),
            auto_focal_foreground_threshold=float(raw.get("auto_focal_foreground_threshold", 0.05)),
            auto_focal_boundary_threshold=float(raw.get("auto_focal_boundary_threshold", 0.02)),
            auto_focal_weight=float(raw.get("auto_focal_weight", 0.5)),
            auto_boundary_focal_weight=float(raw.get("auto_boundary_focal_weight", 0.25)),
            auto_focal_sample_limit=int(raw.get("auto_focal_sample_limit", 64)),
            progress_update_interval=_auto_or_int(
                raw.get("progress_update_interval", AUTO), "progress_update_interval"
            ),
            log_update_interval=_auto_or_int(raw.get("log_update_interval", AUTO), "log_update_interval"),
            save_every_epoch=_coerce_bool(raw.get("save_every_epoch", True), "save_every_epoch"),
            preview_count=int(raw.get("preview_count", 20)),
            normalization=_nested_dataclass(NormalizationConfig, raw.get("normalization")),
            postprocessing=_nested_dataclass(PostprocessingConfig, raw.get("postprocessing")),
            loss_weights=_loss_weights(raw.get("loss_weights")),
            augmentation=dict(raw.get("augmentation", {})),
        )
        parsed._provided_fields = frozenset(raw)

    if not parsed.model_name.strip():
        raise ConfigError("model_name cannot be empty")
    if not math.isfinite(parsed.max_padding_ratio) or parsed.max_padding_ratio < 0:
        raise ConfigError("max_padding_ratio must be finite and nonnegative")
    if "/" in parsed.model_name or "\\" in parsed.model_name:
        raise ConfigError("model_name must be a name, not a path")
    if parsed.starting_point not in {"scratch", "fine_tune", "finetune"}:
        raise ConfigError("starting_point must be 'scratch' or 'fine_tune'")
    if parsed.starting_point in {"fine_tune", "finetune"} and parsed.base_model is None:
        raise ConfigError("base_model is required when starting_point is fine_tune")
    if parsed.starting_point in {"fine_tune", "finetune"} and parsed.learning_rate != AUTO:
        raise ConfigError("Numeric learning_rate is supported only for scratch training; use 'auto' for fine-tuning")
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
    if parsed.data_cache_mb != AUTO and (
        not math.isfinite(float(parsed.data_cache_mb)) or float(parsed.data_cache_mb) < 0
    ):
        raise ConfigError("data_cache_mb must be 'auto' or a finite nonnegative number")
    if parsed.preview_count < 0:
        raise ConfigError("preview_count cannot be negative")
    if parsed.focal_gamma <= 0:
        raise ConfigError("focal_gamma must be positive")
    if parsed.focal_alpha is not None and not 0 < parsed.focal_alpha < 1:
        raise ConfigError("focal_alpha must be in (0, 1)")
    if not 0 <= parsed.auto_focal_foreground_threshold <= 1:
        raise ConfigError("auto_focal_foreground_threshold must be in [0, 1]")
    if not 0 <= parsed.auto_focal_boundary_threshold <= 1:
        raise ConfigError("auto_focal_boundary_threshold must be in [0, 1]")
    if parsed.auto_focal_weight < 0:
        raise ConfigError("auto_focal_weight cannot be negative")
    if parsed.auto_boundary_focal_weight < 0:
        raise ConfigError("auto_boundary_focal_weight cannot be negative")
    if parsed.auto_focal_sample_limit < 1:
        raise ConfigError("auto_focal_sample_limit must be at least 1")
    if parsed.foreground_probability != AUTO and not 0 <= float(parsed.foreground_probability) <= 1:
        raise ConfigError("foreground_probability must be in [0, 1]")
    if parsed.empty_patch_max_retries < 0:
        raise ConfigError("empty_patch_max_retries cannot be negative")
    if parsed.context_slices != AUTO and (int(parsed.context_slices) < 1 or int(parsed.context_slices) % 2 == 0):
        raise ConfigError("context_slices must be a positive odd integer")
    if parsed.context.stride_policy not in {"adjacent", "fixed_stride", "nearest_physical"}:
        raise ConfigError("context.stride_policy must be adjacent, fixed_stride, or nearest_physical")
    if int(parsed.context.stride) < 1:
        raise ConfigError("context.stride must be at least 1")
    if parsed.context.spacing not in (AUTO, None) and float(parsed.context.spacing) <= 0:
        raise ConfigError("context.spacing must be 'auto' or positive")
    if min(parsed.effective_batch_size, parsed.minimum_patches_per_epoch, parsed.expected_patches_per_case) < 1:
        raise ConfigError("effective batch and training-step settings must be positive")
    if parsed.minimum_steps_per_epoch is not None and parsed.minimum_steps_per_epoch < 1:
        raise ConfigError("minimum_steps_per_epoch must be null or positive")
    if not 0 < parsed.memory_fraction <= 1:
        raise ConfigError("memory_fraction must be in (0, 1]")
    _validate_auto_positive_int(parsed.steps_per_epoch, "steps_per_epoch")
    spacing = parsed.spacing
    parsed_default_spacing = tuple(float(value) for value in spacing.default_spacing)
    if len(parsed_default_spacing) != 3 or any(value <= 0 for value in parsed_default_spacing):
        raise ConfigError("spacing.default_spacing must contain three positive Z,Y,X values")
    spacing.default_spacing = parsed_default_spacing
    if not 0 <= spacing.known_fraction_threshold <= 1:
        raise ConfigError("spacing.known_fraction_threshold must be in [0, 1]")
    if spacing.anisotropy_threshold <= 1 or spacing.kernel_anisotropy_threshold <= 1:
        raise ConfigError("spacing anisotropy thresholds must be greater than 1")
    if spacing.max_upsampling < 1 or spacing.minimum_feature_map_size < 2:
        raise ConfigError("spacing safeguards are invalid")
    validation = parsed.validation
    if validation.mode not in {"light", "full"}:
        raise ConfigError("validation.mode must be 'light' or 'full'")
    for name in ("light_every", "light_steps", "minimum_batches", "minimum_samples", "candidate_attempts"):
        value = getattr(validation, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigError(f"validation.{name} must be a positive integer")
    for name in ("full_every", "early_stopping_patience"):
        value = getattr(validation, name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ConfigError(f"validation.{name} must be null or a nonnegative integer; 0 disables it")
    for name in ("foreground_fraction", "minimum_foreground", "minimum_source_fraction", "max_sampling_overlap", "tile_overlap"):
        value = getattr(validation, name)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= float(value) <= 1:
            raise ConfigError(f"validation.{name} must be a finite fraction in [0, 1]")
    if validation.minimum_foreground <= 0 or validation.max_sampling_overlap > 0.10 or validation.tile_overlap >= 1:
        raise ConfigError("Validation requires positive foreground occupancy, sampling overlap <= 0.10, and tile_overlap < 1")
    if validation.preview_max_bytes != AUTO and (
        isinstance(validation.preview_max_bytes, bool) or not isinstance(validation.preview_max_bytes, int)
        or validation.preview_max_bytes < 1
    ):
        raise ConfigError("validation.preview_max_bytes must be 'auto' or a positive integer byte count")
    if validation.tile_blending not in {"constant", "gaussian"}:
        raise ConfigError("validation.tile_blending must be 'constant' or 'gaussian'")
    _validate_auto_positive_int(parsed.input_channels, "input_channels")
    _validate_auto_positive_int(parsed.output_classes, "output_classes")
    _validate_auto_positive_int(parsed.batch_size, "batch_size")
    _validate_auto_positive_int(parsed.progress_update_interval, "progress_update_interval")
    _validate_auto_positive_int(parsed.log_update_interval, "log_update_interval")
    _validate_auto_positive_float(parsed.learning_rate, "learning_rate")
    parsed.model_normalization = _model_normalization(parsed.model_normalization)
    _validate_lr_scheduler(parsed.lr_scheduler)
    preparation = parsed.annotation_preparation
    preparation.repair_disconnected_instances = _coerce_bool(
        preparation.repair_disconnected_instances, "annotation_preparation.repair_disconnected_instances"
    )
    for name in ("ram_cache_mb", "disk_reserve_mb", "warning_fraction"):
        try:
            value = float(getattr(preparation, name))
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"annotation_preparation.{name} must be a finite nonnegative number") from exc
        if not math.isfinite(value) or value < 0 or (name == "warning_fraction" and value > 1):
            raise ConfigError(f"Invalid annotation_preparation.{name}")
        setattr(preparation, name, value)
    if preparation.cache_dir is not None:
        if not isinstance(preparation.cache_dir, (str, Path)) or not str(preparation.cache_dir).strip():
            raise ConfigError("annotation_preparation.cache_dir must be a nonempty path or null")
        preparation.cache_dir = str(preparation.cache_dir)

    scale_cfg = parsed.instance_scale_normalization
    scale_cfg.enabled = _coerce_bool(scale_cfg.enabled, "instance_scale_normalization.enabled")
    scale_cfg.exclude_border_instances = _coerce_bool(
        scale_cfg.exclude_border_instances, "instance_scale_normalization.exclude_border_instances"
    )
    try:
        scale_cfg.target_object_fraction = float(scale_cfg.target_object_fraction)
        scale_cfg.max_instances_per_image = int(scale_cfg.max_instances_per_image)
        scale_cfg.min_instance_area = int(scale_cfg.min_instance_area)
        scale_cfg.min_effective_scale = float(scale_cfg.min_effective_scale)
        scale_cfg.max_effective_scale = float(scale_cfg.max_effective_scale)
    except (TypeError, ValueError) as exc:
        raise ConfigError("instance_scale_normalization numeric options must be valid numbers") from exc
    if not 0 < scale_cfg.target_object_fraction < 1:
        raise ConfigError("instance_scale_normalization.target_object_fraction must be in (0, 1)")
    aliases = {"equivalent_diameter": "equivalent_sphere_diameter"}
    scale_cfg.object_size_measure = aliases.get(scale_cfg.object_size_measure, scale_cfg.object_size_measure)
    if scale_cfg.object_size_measure not in {"equivalent_sphere_diameter", "principal_axes"}:
        raise ConfigError("instance_scale_normalization.object_size_measure must be equivalent_sphere_diameter or principal_axes")
    if int(scale_cfg.max_instances_per_image) < 1:
        raise ConfigError("instance_scale_normalization.max_instances_per_image must be at least 1")
    if int(scale_cfg.min_instance_area) < 1:
        raise ConfigError("instance_scale_normalization.min_instance_area must be at least 1")
    try:
        parsed_jitter = tuple(float(item) for item in scale_cfg.training_scale_jitter)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            "instance_scale_normalization.training_scale_jitter must contain two positive ordered values"
        ) from exc
    if len(parsed_jitter) != 2 or not 0 < parsed_jitter[0] <= parsed_jitter[1]:
        raise ConfigError("instance_scale_normalization.training_scale_jitter must contain two positive ordered values")
    scale_cfg.training_scale_jitter = parsed_jitter
    if scale_cfg.jitter_distribution != "log_uniform":
        raise ConfigError("instance_scale_normalization.jitter_distribution must be 'log_uniform'")
    if not 0 < float(scale_cfg.min_effective_scale) <= float(scale_cfg.max_effective_scale):
        raise ConfigError("instance_scale_normalization effective scale bounds must be positive and ordered")
    _validate_normalization(parsed.normalization)
    _validate_postprocessing(parsed.postprocessing)
    if parsed.architecture == AUTO and parsed.starting_point not in {"fine_tune", "finetune"}:
        parsed.architecture = "resenc-tiny-2d"
    if parsed.architecture == AUTO:
        return parsed
    arch = source_architecture or architecture_defaults(
        str(parsed.architecture), normalization=parsed.model_normalization
    )
    if parsed.context_slices == AUTO and parsed.starting_point not in {"fine_tune", "finetune"}:
        parsed.context_slices = default_context_slices(parsed.architecture)
    if arch.dimensions == "2.5d" and parsed.context_slices != AUTO and int(parsed.context_slices) < 3:
        raise ConfigError("2.5D models require context_slices to be at least 3")
    if parsed.patch_size != AUTO:
        expected_dims = 3 if arch.dimensions == "3d" else 2
        if len(parsed.patch_size) != expected_dims:
            raise ConfigError(f"patch_size for {arch.dimensions} models must have {expected_dims} value(s)")
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
    normalization: str = "group",
    deep_supervision: bool = False,
) -> ArchitectureConfig:
    normalization = _model_normalization(normalization)
    name = architecture.lower().replace("25d", "2.5d")
    dimensions = "3d" if name.endswith("-3d") else "2.5d" if name.endswith("-2.5d") else "2d"
    stripped = name.removeprefix("resenc-").removeprefix("residual-")
    preset = stripped.split("-")[0]
    presets = {
        "tiny": ((16, 32, 64, 128), (1, 2, 2, 2), 4),
        "medium": ((24, 48, 96, 192, 320), (1, 2, 2, 2, 2), 8),
        "big": ((32, 64, 128, 256, 320), (1, 3, 4, 4, 4), 16),
        "large": ((32, 64, 128, 256, 384, 512), (1, 3, 4, 6, 6, 6), 24),
    }
    if preset not in presets:
        raise ConfigError(f"Unsupported architecture: {architecture}")
    channels, blocks, memory = presets[preset]
    legacy_3d = {
        "tiny-3d": ((12, 24, 48), (2, 2, 2), 4),
        "medium-3d": ((24, 48, 96), (2, 2, 2), 8),
    }
    if name in legacy_3d:
        channels, blocks, memory = legacy_3d[name]
    # Legacy non-resenc names remain loadable, while all new preset names default to ResEnc.
    block_type = "conv" if name in {"tiny-2d", "medium-2d", "tiny-3d", "medium-3d", "tiny-2.5d", "medium-2.5d"} else "residual"
    return ArchitectureConfig(
        name=architecture,
        input_channels=input_channels,
        output_channels=output_channels,
        base_channels=channels[0],
        depth=len(channels),
        convs_per_level=2,
        normalization=normalization,
        dimensions=dimensions,
        block_type=block_type,
        deep_supervision=deep_supervision,
        channels=channels,
        encoder_blocks=blocks,
        reference_memory_gb=memory,
    )


def default_patch_size(architecture: str, image_shape: tuple[int, ...] | None = None) -> tuple[int, ...]:
    name = architecture.lower()
    preferred: tuple[int, ...]
    if name.endswith("-3d") and "large" in name:
        preferred = (48, 160, 160)
    elif name.endswith("-3d") and "big" in name:
        preferred = (32, 128, 128)
    elif name.endswith("-3d") and "medium" in name:
        preferred = (24, 96, 96)
    elif name.endswith("-3d"):
        preferred = (16, 64, 64)
    else:
        preferred = (
            (512, 512)
            if "large" in name
            else (384, 384)
            if "big" in name
            else (256, 256)
            if "medium" in name
            else (128, 128)
        )
    if image_shape is None:
        return preferred
    return tuple(min(preferred[index], int(image_shape[index])) for index in range(len(preferred)))


def _is_tiny_2d(architecture: str) -> bool:
    name = architecture.lower().removeprefix("resenc-").removeprefix("residual-")
    return name == "tiny-2d"


def default_effective_batch_size(architecture: str, device: torch.device) -> int:
    return {"cpu": 16, "cuda": 32}.get(device.type, 4) if _is_tiny_2d(architecture) else 4


def default_batch_size(architecture: str, device: torch.device) -> int:
    if _is_tiny_2d(architecture):
        return default_effective_batch_size(architecture, device)
    name = architecture.lower()
    if name.endswith("-3d"):
        return 2 if device.type != "cpu" and "tiny" in name else 1
    if "large" in name:
        return 1
    if "big" in name:
        return 1 if device.type == "cpu" else 2
    if "medium" in name:
        return 2 if device.type == "cpu" else 4
    return 4


def resolve_steps_per_epoch(
    config: TrainingConfig, architecture: str, training_cases: int, effective_batch_size: int
) -> int:
    """Resolve optimizer updates using the actual batch after accumulation."""
    if config.steps_per_epoch != AUTO:
        return int(config.steps_per_epoch)
    patches = config.expected_patches_per_case * training_cases
    name = architecture.lower().removeprefix("resenc-").removeprefix("residual-")
    if name.startswith("tiny-"):
        patches = max(config.minimum_patches_per_epoch, patches)
        minimum_steps = 1
    else:
        minimum_steps = 250
    if config.minimum_steps_per_epoch is not None:
        minimum_steps = config.minimum_steps_per_epoch
    return max(minimum_steps, math.ceil(patches / effective_batch_size))


def default_context_slices(architecture: str) -> int:
    name = architecture.lower()
    if not name.replace("25d", "2.5d").endswith("-2.5d"):
        return 3
    if "large" in name:
        return 11
    if "big" in name:
        return 9
    if "medium" in name:
        return 7
    return 5


def default_deep_supervision(architecture: str) -> bool:
    name = architecture.lower()
    return any(preset in name for preset in ("medium", "big", "large")) and name.startswith(
        ("resenc-", "residual-")
    )


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
    if "medium" in architecture.lower():
        return 0.5
    return 0.4


def default_augmentation_profile(architecture: str, device: torch.device) -> str:
    name = architecture.lower()
    return "strong" if any(preset in name for preset in ("big", "large")) else "balanced"


def default_progress_update_interval(device: torch.device) -> int:
    return 1 if device.type == "cpu" else 5


def default_log_update_interval(device: torch.device) -> int:
    return 10 if device.type == "cpu" else 50


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
        "training": to_jsonable({key: value for key, value in asdict(train_config).items() if not key.startswith("_")}),
    }


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(to_jsonable(dict(payload)), indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp_path, path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc
