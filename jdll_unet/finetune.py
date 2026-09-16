"""Strict source-model recovery and auditable fine-tuning adaptation."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch

from .config import ArchitectureConfig
from .errors import ModelLoadError
from .model import build_unet

_INPUT_TENSORS = {
    "encoders.0.block.0.weight",
    "encoders.0.body.0.weight",
    "encoders.0.projection.weight",
}


@dataclass(frozen=True, slots=True)
class SourceModel:
    requested_path: Path
    checkpoint_path: Path
    architecture: ArchitectureConfig
    model_config: dict[str, Any]
    state_dict: dict[str, torch.Tensor]
    task: str
    label_values: tuple[int, ...] | None
    learning_rate: float | None
    base_learning_rate: float = 1e-3
    learning_rate_provenance: str = "fallback_default"
    recovery_sources: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FineTuneReport:
    source_model: str
    source_checkpoint: str
    source_architecture: str
    source_input_channels: int
    target_input_channels: int
    source_output_channels: int
    target_output_channels: int
    source_learning_rate: float | None
    backbone_learning_rate: float
    adapted_layers_learning_rate: float | None
    input_adaptation: str
    output_adaptation: str
    copied_tensors: tuple[str, ...]
    adapted_tensors: tuple[str, ...]
    reinitialized_tensors: tuple[str, ...]
    missing_tensors: tuple[str, ...]
    unexpected_tensors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _architecture_from_payload(payload: Any, source: str) -> ArchitectureConfig:
    if not isinstance(payload, dict):
        raise ModelLoadError(f"Missing or invalid architecture_config in {source}")
    try:
        values = dict(payload)
        required = {
            "name",
            "dimensions",
            "input_channels",
            "output_channels",
            "depth",
            "normalization",
            "block_type",
            "deep_supervision",
        }
        if not required.issubset(values) or any(value == "auto" for value in values.values() if isinstance(value, str)):
            raise ValueError("Architecture-defining values are missing or unresolved")
        for key in ("channels", "encoder_blocks"):
            if key in values:
                values[key] = tuple(int(value) for value in values[key])
        for key in ("kernels", "strides"):
            if key in values:
                values[key] = tuple(tuple(int(item) for item in value) for value in values[key])
        architecture = ArchitectureConfig(**values)
        build_unet(architecture)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ModelLoadError(f"Unsupported source architecture metadata in {source}: {exc}") from exc
    return architecture


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelLoadError(f"Cannot read source model configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ModelLoadError(f"Source model configuration {path} must contain a JSON object")
    return cast(dict[str, Any], value)


def _source_learning_rate(config: dict[str, Any]) -> float | None:
    training = config.get("training")
    if not isinstance(training, dict):
        return None
    for key in ("learning_rate", "backbone_learning_rate"):
        value = training.get(key)
        if value is None or value == "auto":
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ModelLoadError(f"Corrupt {key} in source training metadata") from exc
        if math.isfinite(parsed) and parsed > 0:
            return parsed
        raise ModelLoadError(f"Corrupt {key} in source training metadata: expected a finite positive LR")
    return None


def _recover_saved_values(
    saved: dict[str, Any], trusted: dict[str, Any], sources: list[str], prefix: str = "model_config"
) -> dict[str, Any]:
    recovered = dict(saved)
    for key, value in trusted.items():
        current = recovered.get(key)
        path = f"{prefix}.{key}"
        if key not in recovered or (
            isinstance(current, str)
            and current == "auto"
            and key not in {"model_name", "dataset_path", "output_dir", "base_model"}
        ):
            recovered[key] = value
            sources.append(f"checkpoint.{path}")
        elif isinstance(current, dict) and isinstance(value, dict):
            recovered[key] = _recover_saved_values(current, value, sources, path)
    return recovered


def recover_base_learning_rate(config: dict[str, Any], _visited: tuple[str, ...] = ()) -> tuple[float, str]:
    training = config.get("training", {})
    if not isinstance(training, dict):
        training = {}
    for key in (
        "base_learning_rate",
        "learning_rate",
        "backbone_learning_rate",
        "adapted_layers_learning_rate",
        "source_learning_rate",
    ):
        value = training.get(key)
        if value is not None and value != "auto":
            try:
                rate = float(value)
            except (TypeError, ValueError) as exc:
                raise ModelLoadError(f"Corrupt source LR metadata: {key}") from exc
            if isinstance(value, bool) or not math.isfinite(rate) or rate <= 0:
                raise ModelLoadError(f"Corrupt source LR metadata: {key} must be finite and positive")
    base = training.get("base_learning_rate")
    if base is not None and base != "auto":
        return float(base), "saved_base_learning_rate"
    initial = _source_learning_rate(config)
    if training.get("starting_point") == "scratch" and initial is not None:
        return initial, "legacy_scratch_initial_learning_rate"
    report = training.get("fine_tuning_initialization")
    if isinstance(report, dict):
        # The previous JDLL rule used 0.1 * source-initial-LR. Only a source
        # explicitly identified as scratch establishes lineage unambiguously.
        source_config = report.get("source_config")
        if isinstance(source_config, dict) and source_config.get("training", {}).get("starting_point") == "scratch":
            return recover_base_learning_rate(source_config)[0], "legacy_saved_scratch_provenance"
        path = report.get("source_model")
        recorded_source = report.get("source_learning_rate")
        recorded_backbone = report.get("backbone_learning_rate")
        if path and recorded_source is not None and recorded_backbone is not None:
            try:
                known_rule = (
                    math.isfinite(float(recorded_source))
                    and float(recorded_source) > 0
                    and math.isclose(float(recorded_backbone), 0.1 * float(recorded_source))
                )
            except (TypeError, ValueError):
                known_rule = False
            parent = Path(path)
            parent_config_path = parent / "config.json" if parent.is_dir() else parent.parent / "config.json"
            identity = str(parent_config_path.resolve())
            if known_rule and identity not in _visited and len(_visited) < 64 and parent_config_path.is_file():
                parent_config = _read_json(parent_config_path)
                parent_training = parent_config.get("training", {})
                matching = any(
                    value is not None and value != "auto" and float(value) == float(recorded_source)
                    for key in ("learning_rate", "adapted_layers_learning_rate", "backbone_learning_rate")
                    for value in [parent_training.get(key)]
                )
                if matching:
                    base, provenance = recover_base_learning_rate(parent_config, (*_visited, identity))
                    if provenance != "fallback_default":
                        return base, f"legacy_source_lineage:{identity}"
    warnings.warn(
        "Cannot recover original scratch LR; using fallback base_learning_rate=0.001", RuntimeWarning, stacklevel=2
    )
    return 1e-3, "fallback_default"


def resolve_source_model(base_model: Path | str, device: torch.device | str = "cpu") -> SourceModel:
    requested = Path(base_model)
    checkpoint = requested / "model.pt" if requested.is_dir() else requested
    if not checkpoint.exists() or not checkpoint.is_file():
        raise ModelLoadError(f"Base model checkpoint does not exist: {checkpoint}")
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
    except Exception as exc:
        raise ModelLoadError(f"Cannot load base model checkpoint {checkpoint}: {exc}") from exc
    if not isinstance(state, dict):
        raise ModelLoadError(f"Base model checkpoint {checkpoint} is not a JDLL UNet checkpoint")
    raw_state = state.get("state_dict")
    if not isinstance(raw_state, dict) or not all(isinstance(value, torch.Tensor) for value in raw_state.values()):
        raise ModelLoadError(f"Base model checkpoint {checkpoint} has no valid state_dict")
    state_dict = cast(dict[str, torch.Tensor], raw_state)

    checkpoint_config = state.get("model_config")
    folder_config_path = checkpoint.parent / "config.json"
    folder_config = _read_json(folder_config_path) if folder_config_path.exists() else None
    if folder_config is None and not isinstance(checkpoint_config, dict):
        raise ModelLoadError(
            f"Source model {checkpoint} has no recoverable config.json or embedded model_config"
        )
    model_config = folder_config or cast(dict[str, Any], checkpoint_config)
    recovery_sources: list[str] = []
    if isinstance(checkpoint_config, dict):
        model_config = _recover_saved_values(model_config, checkpoint_config, recovery_sources)
    if model_config.get("format") not in {None, "jdll-unet"} or int(model_config.get("format_version", 1)) != 1:
        raise ModelLoadError(f"Unsupported source model schema in {checkpoint}")

    checkpoint_arch_payload = state.get("architecture_config")
    config_arch_payload = model_config.get("architecture_config")
    embedded_arch = checkpoint_config.get("architecture_config") if isinstance(checkpoint_config, dict) else None
    trusted_arch = checkpoint_arch_payload or embedded_arch
    if isinstance(config_arch_payload, dict) and isinstance(trusted_arch, dict):
        merged = dict(config_arch_payload)
        for key, value in trusted_arch.items():
            if key not in merged or merged[key] == "auto":
                merged[key] = value
                recovery_sources.append(f"checkpoint.architecture_config.{key}")
        config_arch_payload = merged
    architecture = _architecture_from_payload(
        config_arch_payload if config_arch_payload is not None else trusted_arch,
        str(checkpoint),
    )
    if checkpoint_arch_payload is not None:
        checkpoint_arch = _architecture_from_payload(checkpoint_arch_payload, str(checkpoint))
        if asdict(checkpoint_arch) != asdict(architecture):
            raise ModelLoadError("Source config.json and checkpoint architecture_config disagree")
    if embedded_arch is not None and asdict(_architecture_from_payload(embedded_arch, str(checkpoint))) != asdict(
        architecture
    ):
        raise ModelLoadError("Source config.json and embedded checkpoint architecture_config disagree")
    if isinstance(checkpoint_config, dict):
        for key in ("task", "label_values", "normalization"):
            if key in model_config and key in checkpoint_config and model_config[key] != checkpoint_config[key]:
                raise ModelLoadError(f"Source config.json and checkpoint {key} disagree")
        for key in ("base_learning_rate", "learning_rate", "backbone_learning_rate", "adapted_layers_learning_rate"):
            saved = model_config.get("training", {}).get(key)
            embedded = checkpoint_config.get("training", {}).get(key)
            if saved not in (None, "auto") and embedded not in (None, "auto") and saved != embedded:
                raise ModelLoadError(f"Source config.json and checkpoint {key} disagree")

    source_model = build_unet(architecture)
    try:
        source_model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ModelLoadError(f"Source checkpoint tensors disagree with architecture metadata: {exc}") from exc

    task = str(model_config.get("task", state.get("task", "")))
    if task not in {"binary_semantic", "multiclass_semantic", "instance_friendly"}:
        raise ModelLoadError(f"Source model has unsupported or missing task semantics: {task!r}")
    raw_labels = model_config.get("label_values")
    labels = tuple(int(value) for value in raw_labels) if isinstance(raw_labels, list) else None
    expected_outputs = (
        1
        if task == "binary_semantic"
        else 3
        if task == "instance_friendly"
        else len([value for value in labels or () if value != 0]) + 1
        if labels is not None
        else None
    )
    if expected_outputs is not None and architecture.output_channels != expected_outputs:
        raise ModelLoadError(
            f"Source task metadata expects {expected_outputs} output channels but architecture records "
            f"{architecture.output_channels}"
        )
    base_lr, provenance = recover_base_learning_rate(model_config)
    return SourceModel(
        requested.resolve(),
        checkpoint.resolve(),
        architecture,
        model_config,
        state_dict,
        task,
        labels,
        _source_learning_rate(model_config),
        base_lr,
        provenance,
        tuple(recovery_sources),
    )


def target_architecture(
    source: SourceModel,
    *,
    input_channels: int,
    output_channels: int,
    context_slices: int,
) -> ArchitectureConfig:
    return replace(
        source.architecture,
        input_channels=input_channels,
        output_channels=output_channels,
        context_slices=context_slices,
    )


def resolve_finetune_learning_rates(source_learning_rate: float | None) -> tuple[float, float]:
    if source_learning_rate is None:
        return 1e-4, 1e-3
    return source_learning_rate * 0.1, source_learning_rate


def _adapt_flat_input(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    source_channels, target_channels = source.shape[1], target.shape[1]
    result = torch.zeros_like(target)
    if source_channels == 1 and target_channels > 1:
        result.copy_(source.repeat(1, target_channels, *([1] * (source.ndim - 2))) / target_channels)
    elif source_channels > 1 and target_channels == 1:
        result.copy_(source.sum(dim=1, keepdim=True))
    else:
        shared = min(source_channels, target_channels)
        result[:, :shared].copy_(source[:, :shared])
    return result


def _centered_plane_pairs(source_context: int, target_context: int) -> list[tuple[int, int]]:
    overlap = min(source_context, target_context)
    source_start = (source_context - overlap) // 2
    target_start = (target_context - overlap) // 2
    return [(source_start + index, target_start + index) for index in range(overlap)]


def _adapt_25d_input(
    source: torch.Tensor,
    target: torch.Tensor,
    source_context: int,
    target_context: int,
) -> torch.Tensor:
    if source.shape[1] % source_context or target.shape[1] % target_context:
        raise ModelLoadError("2.5D input channels are not divisible by their context-plane counts")
    source_images = source.shape[1] // source_context
    target_images = target.shape[1] // target_context
    result = torch.zeros_like(target)
    plane_pairs = _centered_plane_pairs(source_context, target_context)
    if source_images == 1 and target_images > 1:
        image_pairs = [(0, target_image, 1.0 / target_images) for target_image in range(target_images)]
    elif source_images > 1 and target_images == 1:
        image_pairs = [(source_image, 0, 1.0) for source_image in range(source_images)]
    else:
        image_pairs = [
            (image, image, 1.0) for image in range(min(source_images, target_images))
        ]
    for source_image, target_image, scale in image_pairs:
        for source_plane, target_plane in plane_pairs:
            source_index = source_image * source_context + source_plane
            target_index = target_image * target_context + target_plane
            result[:, target_index].add_(source[:, source_index] * scale)
    return result


def _head_tensor(name: str) -> bool:
    return name.startswith("out.") or name.startswith("deep_supervision_heads.")


def _output_mapping(
    source: SourceModel,
    target_task: str,
    target_labels: tuple[int, ...],
    target_outputs: int,
) -> tuple[dict[int, int] | None, bool]:
    source_outputs = source.architecture.output_channels
    if source.task != target_task:
        return None, False
    if target_task in {"binary_semantic", "instance_friendly"}:
        unchanged = source_outputs == target_outputs
        return ({index: index for index in range(target_outputs)} if unchanged else None), unchanged
    if source.label_values is None:
        return None, False
    source_labels = tuple(int(value) for value in source.label_values if int(value) != 0)
    target_labels = tuple(int(value) for value in target_labels if int(value) != 0)
    mapping = {0: 0}
    source_indexes = {label: index + 1 for index, label in enumerate(source_labels)}
    for target_index, label in enumerate(target_labels, start=1):
        if label in source_indexes:
            mapping[target_index] = source_indexes[label]
    unchanged = source_labels == target_labels and source_outputs == target_outputs
    return mapping, unchanged


def initialize_finetune_model(
    model: torch.nn.Module,
    source: SourceModel,
    *,
    target_task: str,
    target_label_values: list[int],
    backbone_learning_rate: float,
    adapted_learning_rate: float,
) -> tuple[FineTuneReport, set[str]]:
    target_state = model.state_dict()
    source_state = source.state_dict
    source_context = source.architecture.context_slices
    target_context = cast(ArchitectureConfig, model.config).context_slices
    dimensions = cast(ArchitectureConfig, model.config).dimensions
    output_mapping, output_unchanged = _output_mapping(
        source,
        target_task,
        tuple(target_label_values),
        cast(ArchitectureConfig, model.config).output_channels,
    )
    copied: list[str] = []
    adapted: list[str] = []
    reinitialized: list[str] = []
    missing: list[str] = []
    resolved: dict[str, torch.Tensor] = {}

    for name, target_tensor in target_state.items():
        source_tensor = source_state.get(name)
        if name in _INPUT_TENSORS:
            if source_tensor is None:
                resolved[name] = target_tensor
                reinitialized.append(name)
                missing.append(name)
            elif source_tensor.shape == target_tensor.shape and (
                dimensions != "2.5d" or source_context == target_context
            ):
                resolved[name] = source_tensor
                copied.append(name)
            elif (
                source_tensor.ndim == target_tensor.ndim
                and source_tensor.shape[0] == target_tensor.shape[0]
                and source_tensor.shape[2:] == target_tensor.shape[2:]
            ):
                resolved[name] = (
                    _adapt_25d_input(source_tensor, target_tensor, source_context, target_context)
                    if dimensions == "2.5d"
                    else _adapt_flat_input(source_tensor, target_tensor)
                )
                adapted.append(name)
            else:
                raise ModelLoadError(f"Unexplained input-layer mismatch for tensor {name}")
            continue
        if _head_tensor(name):
            if source_tensor is None:
                resolved[name] = target_tensor
                reinitialized.append(name)
                missing.append(name)
            elif output_unchanged and source_tensor.shape == target_tensor.shape:
                resolved[name] = source_tensor
                copied.append(name)
            elif output_mapping is not None and source_tensor.ndim == target_tensor.ndim:
                value = target_tensor.clone()
                for target_channel, source_channel in output_mapping.items():
                    if target_channel < value.shape[0] and source_channel < source_tensor.shape[0]:
                        value[target_channel].copy_(source_tensor[source_channel])
                resolved[name] = value
                adapted.append(name)
            else:
                resolved[name] = target_tensor
                reinitialized.append(name)
            continue
        if source_tensor is None:
            raise ModelLoadError(f"Unexplained missing backbone tensor: {name}")
        if source_tensor.shape != target_tensor.shape:
            raise ModelLoadError(
                f"Unexplained backbone mismatch for {name}: source={tuple(source_tensor.shape)} "
                f"target={tuple(target_tensor.shape)}"
            )
        resolved[name] = source_tensor
        copied.append(name)

    unexpected = [
        name
        for name in source_state
        if name not in target_state and name not in _INPUT_TENSORS and not _head_tensor(name)
    ]
    if unexpected:
        raise ModelLoadError(f"Unexpected source backbone tensors: {', '.join(unexpected)}")
    allowed_unexpected = [
        name for name in source_state if name not in target_state and name not in unexpected
    ]
    model.load_state_dict(resolved, strict=True)
    adapted_parameters = {
        name
        for name, _parameter in model.named_parameters()
        if name in set(adapted) | set(reinitialized)
    }
    has_adaptation = bool(adapted_parameters)
    report = FineTuneReport(
        str(source.requested_path),
        str(source.checkpoint_path),
        source.architecture.name,
        source.architecture.input_channels,
        cast(ArchitectureConfig, model.config).input_channels,
        source.architecture.output_channels,
        cast(ArchitectureConfig, model.config).output_channels,
        source.learning_rate,
        backbone_learning_rate,
        adapted_learning_rate if has_adaptation else None,
        (
            "Input convolution unchanged"
            if source.architecture.input_channels == cast(ArchitectureConfig, model.config).input_channels
            and source_context == target_context
            else f"Adapted input convolution from {source.architecture.input_channels} to "
            f"{cast(ArchitectureConfig, model.config).input_channels} channels"
        ),
        (
            "Output heads unchanged"
            if output_unchanged
            else "Adapted output heads using matching class identities"
            if output_mapping is not None
            else "Reinitialized primary and deep-supervision heads"
        ),
        tuple(sorted(copied)),
        tuple(sorted(adapted)),
        tuple(sorted(reinitialized)),
        tuple(sorted(missing)),
        tuple(sorted(allowed_unexpected)),
    )
    return report, adapted_parameters
