"""Configuration parsing and conservative defaults for JDLL UNet."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar, cast

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
    full_every: int = 5
    light_steps: int = 50
    early_stopping_patience: int = 20


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


@dataclass(slots=True)
class TrainingConfig:
    model_name: str
    output_dir: Path
    dataset_path: Path
    starting_point: str = "scratch"
    base_model: Path | None = None
    architecture: str = "resenc-tiny-2d"
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
    instance_scale_normalization: InstanceScaleNormalizationConfig = field(
        default_factory=InstanceScaleNormalizationConfig
    )
    validation_fraction: float = 0.15
    foreground_oversampling: bool | str = AUTO
    foreground_probability: float | str = AUTO
    skip_empty_images: bool = True
    skip_empty_patches: bool = True
    empty_patch_max_retries: int = 8
    include_empty_patches_after_max_retries: bool = False
    augmentation_profile: str = AUTO
    num_workers: int = 0
    mixed_precision: bool | str = AUTO
    deep_supervision: bool | str = False
    context_slices: int = 3
    context: ContextConfig = field(default_factory=ContextConfig)
    spacing: SpacingConfig = field(default_factory=SpacingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    effective_batch_size: int = 4
    steps_per_epoch: int | str = AUTO
    minimum_steps_per_epoch: int = 250
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
    if value is None:
        return cls()
    if isinstance(value, Mapping):
        valid = {item.name for item in fields(cast(Any, cls))}
        unknown = sorted(set(value) - valid)
        if unknown:
            raise ConfigError(f"Unknown {cls.__name__} field(s): {', '.join(unknown)}")
        return cls(**{k: v for k, v in value.items() if k in valid})
    raise ConfigError(f"Expected mapping for {cls.__name__}")


def _lr_scheduler_config(value: Any) -> LRSchedulerConfig:
    if value is None:
        return LRSchedulerConfig()
    if isinstance(value, LRSchedulerConfig):
        return value
    if isinstance(value, str):
        return LRSchedulerConfig(type=value)
    if isinstance(value, Mapping):
        valid = {field.name for field in LRSchedulerConfig.__dataclass_fields__.values()}
        unknown = sorted(set(value) - valid)
        if unknown:
            raise ConfigError(f"Unknown LRSchedulerConfig field(s): {', '.join(unknown)}")
        return LRSchedulerConfig(**{key: value[key] for key in value if key in valid})
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
            architecture=str(raw.get("architecture", "resenc-tiny-2d")),
            device=str(raw.get("device", "cpu")),
            epochs=int(raw.get("epochs", 100)),
            seed=int(raw.get("seed", 42)),
            task=str(raw.get("task", AUTO)),
            axes=str(raw.get("axes", AUTO)),
            input_channels=_auto_or_int(raw.get("input_channels", AUTO), "input_channels"),
            output_classes=_auto_or_int(raw.get("output_classes", AUTO), "output_classes"),
            model_normalization=_model_normalization(
                raw.get("model_normalization", raw.get("network_normalization", raw.get("architecture_normalization", "group")))
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
            validation_fraction=float(raw.get("validation_fraction", 0.15)),
            foreground_oversampling=_auto_or_bool(raw.get("foreground_oversampling", AUTO), "foreground_oversampling"),
            foreground_probability=_auto_or_float(raw.get("foreground_probability", AUTO), "foreground_probability"),
            skip_empty_images=_coerce_bool(raw.get("skip_empty_images", True), "skip_empty_images"),
            skip_empty_patches=_coerce_bool(raw.get("skip_empty_patches", True), "skip_empty_patches"),
            empty_patch_max_retries=int(raw.get("empty_patch_max_retries", 8)),
            include_empty_patches_after_max_retries=_coerce_bool(
                raw.get("include_empty_patches_after_max_retries", False),
                "include_empty_patches_after_max_retries",
            ),
            augmentation_profile=str(raw.get("augmentation_profile", AUTO)),
            num_workers=int(raw.get("num_workers", 0)),
            mixed_precision=raw.get("mixed_precision", AUTO),
            deep_supervision=_auto_or_bool(raw.get("deep_supervision", False), "deep_supervision"),
            context_slices=int(raw.get("context_slices", 3)),
            context=_nested_dataclass(ContextConfig, raw.get("context")),
            spacing=_nested_dataclass(SpacingConfig, raw.get("spacing")),
            validation=_nested_dataclass(ValidationConfig, raw.get("validation")),
            effective_batch_size=int(raw.get("effective_batch_size", 4)),
            steps_per_epoch=_auto_or_int(raw.get("steps_per_epoch", AUTO), "steps_per_epoch"),
            minimum_steps_per_epoch=int(raw.get("minimum_steps_per_epoch", 250)),
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
            progress_update_interval=_auto_or_int(raw.get("progress_update_interval", AUTO), "progress_update_interval"),
            log_update_interval=_auto_or_int(raw.get("log_update_interval", AUTO), "log_update_interval"),
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
    if parsed.context_slices < 1 or parsed.context_slices % 2 == 0:
        raise ConfigError("context_slices must be a positive odd integer")
    if parsed.context.stride_policy not in {"adjacent", "fixed_stride", "nearest_physical"}:
        raise ConfigError("context.stride_policy must be adjacent, fixed_stride, or nearest_physical")
    if int(parsed.context.stride) < 1:
        raise ConfigError("context.stride must be at least 1")
    if parsed.context.spacing != AUTO and float(parsed.context.spacing) <= 0:
        raise ConfigError("context.spacing must be 'auto' or positive")
    if parsed.effective_batch_size < 1 or parsed.minimum_steps_per_epoch < 1 or parsed.expected_patches_per_case < 1:
        raise ConfigError("effective batch and training-step settings must be positive")
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
    if min(validation.light_every, validation.full_every, validation.light_steps, validation.early_stopping_patience) < 1:
        raise ConfigError("validation intervals, steps, and patience must be positive")
    _validate_auto_positive_int(parsed.input_channels, "input_channels")
    _validate_auto_positive_int(parsed.output_classes, "output_classes")
    _validate_auto_positive_int(parsed.batch_size, "batch_size")
    _validate_auto_positive_int(parsed.progress_update_interval, "progress_update_interval")
    _validate_auto_positive_int(parsed.log_update_interval, "log_update_interval")
    _validate_auto_positive_float(parsed.learning_rate, "learning_rate")
    parsed.model_normalization = _model_normalization(parsed.model_normalization)
    _validate_lr_scheduler(parsed.lr_scheduler)
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
    arch = architecture_defaults(str(parsed.architecture), normalization=parsed.model_normalization)
    if arch.dimensions == "2.5d" and parsed.context_slices < 3:
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
        preferred = (64, 160, 160)
    elif name.endswith("-3d") and "big" in name:
        preferred = (48, 144, 144)
    elif name.endswith("-3d") and "medium" in name:
        preferred = (16, 128, 128)
    elif name.endswith("-3d"):
        preferred = (16, 96, 96)
    else:
        preferred = (192, 192) if "large" in name else (160, 160) if "big" in name else (128, 128) if "medium" in name else (96, 96)
    if image_shape is None:
        return preferred
    return tuple(min(preferred[index], int(image_shape[index])) for index in range(len(preferred)))


def default_batch_size(architecture: str, device: torch.device) -> int:
    name = architecture.lower()
    if name.endswith("-3d"):
        if device.type == "cpu" or any(preset in name for preset in ("medium", "big", "large")):
            return 1
        return 2
    if "medium" in name:
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
