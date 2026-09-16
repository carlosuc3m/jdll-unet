"""Simple Appose-friendly PyTorch training loop."""

from __future__ import annotations

import logging
import math
import os
import random
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import imageio.v3 as imageio
import numpy as np
import torch
from torch.utils.data import DataLoader

from .callbacks import CallbackDispatcher
from .config import (
    AUTO,
    ArchitectureConfig,
    TrainingConfig,
    architecture_defaults,
    default_augmentation_profile,
    default_foreground_probability,
    default_learning_rate,
    default_log_update_interval,
    default_mixed_precision,
    default_progress_update_interval,
    model_folder_config,
    parse_training_config,
    resolve_device,
    write_json,
)
from .dataset import JdllSegmentationDataset, make_dataset, partition_empty_pairs
from .errors import DatasetError, ModelLoadError
from .finetune import (
    FineTuneReport,
    SourceModel,
    initialize_finetune_model,
    resolve_finetune_learning_rates,
    resolve_source_model,
)
from .geometry import load_domain_image, load_domain_mask
from .image_reading import current_read_session, image_reading_session
from .io import ImageMaskPair, normalize_image
from .losses import compute_loss, primary_logits
from .metrics import compute_metrics, primary_metric
from .model import build_unet
from .planning import (
    resample_image_mask,
    resample_mask,
    resolve_context_stride,
    restore_continuous_maps,
)
from .postprocess import postprocess_instance
from .scale import (
    InstanceSizeEstimate,
    aggregate_instance_statistics,
)
from .schedulers import LearningRateScheduler
from .semantic_scale import semantic_scale_diagnostics
from .targets import boundary_target
from .training_geometry import resolve_training_geometry


class PlanningCancelled(Exception):
    """Internal cooperative stop before any model/optimizer state exists."""


class TrainingStopped(Exception):
    def __init__(self, result: dict[str, Any]) -> None:
        super().__init__("Training cancelled")
        self.result = result


def _setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"jdll_unet.training.{output_dir}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(output_dir / "training.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _available_memory_bytes(device: torch.device) -> int | None:
    if device.type == "cuda":
        try:
            device_index = (device.index if device.index is not None else torch.cuda.current_device())
            free_bytes, _total_bytes = torch.cuda.mem_get_info(device_index)
            return int(free_bytes)
        except (RuntimeError, TypeError, ValueError):
            return None

    if device.type == "cpu":
        try:
            return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (AttributeError, OSError, ValueError):
            return None
    return None


def _move_target(
    target: torch.Tensor | dict[str, torch.Tensor], device: torch.device
) -> torch.Tensor | dict[str, torch.Tensor]:
    if isinstance(target, dict):
        return {key: value.to(device, non_blocking=True) for key, value in target.items()}
    return target.to(device, non_blocking=True)


def _mean_dict(values: list[dict[str, float]]) -> dict[str, float]:
    if not values:
        return {}
    keys = sorted(set().union(*(item.keys() for item in values)))
    return {key: float(np.mean([item[key] for item in values if key in item])) for key in keys}


def _full_volume_validation(
    model: torch.nn.Module,
    pairs: list[ImageMaskPair],
    *,
    task: str,
    dimensions: str,
    device: torch.device,
    patch_size: tuple[int, ...],
    normalization: object,
    context_slices: int,
    context_policy: str,
    context_fixed_stride: int,
    context_target_spacing: float | None,
    case_spacings: dict[str, tuple[float, float, float]],
    target_spacing: tuple[float, float, float] | None,
    instance_sizes: dict[str, float] | None = None,
    target_object_size: float | None = None,
    label_values: list[int] | None = None,
    min_scale: float = 0.25,
    max_scale: float = 4.0,
    check_cancel: Any = None,
) -> dict[str, Any]:
    from .infer import _context_stack, tiled_predict
    from .targets import prepare_target

    per_case: dict[str, float] = {}
    per_case_metrics: dict[str, dict[str, float]] = {}
    evaluated_centers: dict[str, list[int] | None] = {}
    for pair in pairs:
        if check_cancel is not None:
            check_cancel()
        image = load_domain_image(pair, dimensions)
        mask = load_domain_mask(pair, dimensions)
        spacing = case_spacings.get(pair.stem, (1.0, 1.0, 1.0))
        object_size = (instance_sizes or {}).get(pair.stem)
        scale = (
            float(np.clip(target_object_size / object_size, min_scale, max_scale))
            if object_size and target_object_size
            else 1.0
        )
        image = normalize_image(image, normalization) if dimensions != "2d" or image.ndim == 3 else image
        if dimensions == "3d" and target_spacing is not None:
            image, _ = resample_image_mask(image, mask, spacing, target_spacing)
        is_plane_volume = mask.ndim == 3 and dimensions in {"2d", "2.5d"}
        centers = (
            pair.eligible_centers
            if pair.eligible_centers is not None
            else tuple(range(mask.shape[0]))
            if is_plane_volume
            else None
        )
        evaluated_centers[pair.stem] = (
            [z + (pair.region[0][0] if pair.region else 0) for z in centers] if centers is not None else None
        )
        metrics = []
        for z in centers if centers is not None else (None,):
            if check_cancel is not None:
                check_cancel()
            if dimensions == "2.5d":
                assert z is not None
                stride = resolve_context_stride(
                    context_policy,
                    fixed_stride=context_fixed_stride,
                    target_spacing=context_target_spacing,
                    z_spacing=spacing[0],
                )
                current_image = _context_stack(image, z, context_slices, stride)
            elif z is not None:
                current_image = normalize_image(image[:, z], normalization)
            else:
                current_image = image
            current_mask = mask[z] if z is not None else mask
            if scale != 1.0:
                size = tuple(max(1, round(length * scale)) for length in current_image.shape[1:])
                current_image = torch.nn.functional.interpolate(
                    torch.from_numpy(np.ascontiguousarray(current_image[None])),
                    size=size,
                    mode="trilinear" if dimensions == "3d" else "bilinear",
                    align_corners=False,
                )[0].numpy()
            logits = tiled_predict(model, current_image, device, patch_size, overlap=0.5)
            if logits.shape[1:] != current_mask.shape:
                logits = restore_continuous_maps(logits, current_mask.shape)
            target = prepare_target(
                task,
                current_mask,
                label_values=label_values,
                spacing=spacing if dimensions == "3d" else None,
                validity=np.ones(current_mask.shape, dtype=bool),
            )
            assert isinstance(target, dict)
            tensors = {key: torch.from_numpy(value[None]) for key, value in target.items()}
            metrics.append(compute_metrics(task, torch.from_numpy(logits[None]), tensors))
        per_case_metrics[pair.stem] = _mean_dict(metrics)
        per_case[pair.stem] = primary_metric(task, per_case_metrics[pair.stem])
    if not per_case:
        raise DatasetError("Full validation has no eligible real targets")
    return {
        "mean_dice": float(np.mean(list(per_case.values()))),
        "per_case_dice": per_case,
        "per_case_metrics": per_case_metrics,
        "evaluated_centers": evaluated_centers,
    }


def _sample_pairs(pairs: list[ImageMaskPair], sample_limit: int) -> list[ImageMaskPair]:
    if len(pairs) <= sample_limit:
        return pairs
    if sample_limit == 1:
        return [pairs[0]]
    indexes = np.linspace(0, len(pairs) - 1, num=sample_limit, dtype=int)
    return [pairs[int(index)] for index in indexes]


def _estimate_target_sparsity(
    pairs: list[ImageMaskPair],
    task: str,
    sample_limit: int,
    dimensions: str,
) -> dict[str, float | int | None]:
    sampled = _sample_pairs(pairs, sample_limit)
    foreground_pixels = 0
    boundary_pixels = 0
    total_pixels = 0
    for pair in sampled:
        mask = load_domain_mask(pair, dimensions=dimensions)
        foreground_pixels += int(np.count_nonzero(mask))
        total_pixels += int(mask.size)
        if task == "instance_friendly":
            planes = mask if dimensions in {"2d", "2.5d"} and mask.ndim == 3 else (mask,)
            boundary_pixels += sum(int(np.count_nonzero(boundary_target(plane))) for plane in planes)

    foreground_ratio = float(foreground_pixels / total_pixels) if total_pixels else 0.0
    boundary_ratio = float(boundary_pixels / total_pixels) if total_pixels and task == "instance_friendly" else None
    return {
        "sample_count": len(sampled),
        "foreground_ratio": foreground_ratio,
        "boundary_ratio": boundary_ratio,
    }


def _resolve_loss_weights(
    train_config: Any,
    train_pairs: list[ImageMaskPair],
    task: str,
    dimensions: str,
) -> tuple[dict[str, float], dict[str, Any]]:
    weights = dict(train_config.loss_weights)
    if not train_config.auto_focal:
        return weights, {
            "sample_count": 0,
            "foreground_ratio": None,
            "boundary_ratio": None,
            "auto_focal_enabled": False,
            "foreground_focal_enabled": weights.get("focal", 0.0) > 0,
            "boundary_focal_enabled": weights.get("boundary_focal", 0.0) > 0,
        }

    stats = _estimate_target_sparsity(train_pairs, task, train_config.auto_focal_sample_limit, dimensions)
    if train_config.auto_focal and stats["foreground_ratio"] <= train_config.auto_focal_foreground_threshold:
        weights["focal"] = max(weights.get("focal", 0.0), train_config.auto_focal_weight)
    if (
        train_config.auto_focal
        and task == "instance_friendly"
        and stats["boundary_ratio"] is not None
        and stats["boundary_ratio"] <= train_config.auto_focal_boundary_threshold
    ):
        weights["boundary_focal"] = max(weights.get("boundary_focal", 0.0), train_config.auto_boundary_focal_weight)
    stats["auto_focal_enabled"] = bool(train_config.auto_focal)
    stats["foreground_focal_enabled"] = weights.get("focal", 0.0) > 0
    stats["boundary_focal_enabled"] = weights.get("boundary_focal", 0.0) > 0
    return weights, stats


def _tensor_losses_to_float(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu().item()) for key, value in losses.items()}


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst.with_name(f".{dst.name}.tmp")
    shutil.copyfile(src, tmp_path)
    os.replace(tmp_path, dst)


def _atomic_image_write(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
    imageio.imwrite(tmp_path, image)
    os.replace(tmp_path, path)


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LearningRateScheduler | None,
    epoch: int,
    task: str,
    arch: ArchitectureConfig,
    metrics: dict[str, Any],
    model_config: dict[str, Any],
) -> None:
    _atomic_torch_save(
        {
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "task": task,
            "model_config": model_config,
            "architecture_config": asdict(arch),
            "metrics": metrics,
        },
        path,
    )


def _save_previews(
    output_dir: Path,
    epoch: int,
    task: str,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    preview_count: int,
) -> dict[str, str] | None:
    if preview_count <= 0:
        return None
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    saved: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for images, target_batch in loader:
            images = images.to(device)
            logits = primary_logits(model(images)).detach().cpu()
            images_cpu = images.detach().cpu().numpy()
            target_cpu = _target_to_numpy(target_batch)
            support = target_batch.get("valid") if isinstance(target_batch, dict) else None
            predictions = _predictions_to_visual_targets(task, logits, support)
            for idx in range(images_cpu.shape[0]):
                if len(saved) >= preview_count:
                    break
                base = f"preview_{len(saved):03d}"
                image_path = (preview_dir / f"{base}_image.png").resolve()
                target_path = (preview_dir / f"{base}_target.png").resolve()
                pred_path = (preview_dir / f"{base}_prediction.png").resolve()
                overlay_path = (preview_dir / f"{base}_overlay.png").resolve()
                z_index = _preview_z_index(images_cpu[idx])
                preview_image = images_cpu[idx]
                if getattr(loader.dataset, "dimensions", None) == "2.5d":
                    context = cast(JdllSegmentationDataset, loader.dataset).context_slices
                    preview_image = preview_image[context // 2 :: context]
                image_rgb = _image_preview_rgb(preview_image, z_index)
                target_rgb = _target_preview_rgb(task, target_cpu, idx, z_index)
                pred_rgb = _prediction_preview_rgb(task, predictions[idx], z_index)
                if isinstance(target_cpu, dict) and "valid" in target_cpu:
                    valid = _slice_for_preview(target_cpu["valid"][idx, 0], z_index)
                    image_rgb[~valid] = 0
                    target_rgb[~valid] = 0
                    pred_rgb[~valid] = 0
                    validity_path = (preview_dir / f"{base}_validity.png").resolve()
                    _atomic_image_write(validity_path, valid.astype(np.uint8) * 255)
                overlay_rgb = _overlay_prediction(image_rgb, pred_rgb)
                _atomic_image_write(image_path, image_rgb)
                _atomic_image_write(target_path, target_rgb)
                _atomic_image_write(pred_path, pred_rgb)
                _atomic_image_write(overlay_path, overlay_rgb)
                saved.append(
                    {
                        "index": len(saved),
                        "image_path": str(image_path),
                        "target_path": str(target_path),
                        "prediction_path": str(pred_path),
                        "overlay_path": str(overlay_path),
                        "z_index": z_index,
                        "validity_path": str(validity_path)
                        if isinstance(target_cpu, dict) and "valid" in target_cpu
                        else None,
                        **(loader.dataset.provenance(len(saved)) if hasattr(loader.dataset, "provenance") else {}),
                    }
                )
            if len(saved) >= preview_count:
                break
    preview_path = (preview_dir / f"epoch_{epoch:04d}.json").resolve()
    latest_path = (preview_dir / "latest.json").resolve()
    payload = {"epoch": epoch, "task": task, "items": saved}
    write_json(preview_path, payload)
    write_json(latest_path, payload)
    return {"preview_path": str(preview_path), "latest_preview_path": str(latest_path)}


def _target_to_numpy(target: torch.Tensor | dict[str, torch.Tensor]) -> np.ndarray | dict[str, np.ndarray]:
    if isinstance(target, dict):
        return {key: value.detach().cpu().numpy() for key, value in target.items()}
    return target.detach().cpu().numpy()


def _normalize_uint8(array: np.ndarray) -> np.ndarray:
    arr = array.astype(np.float32, copy=False)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def _preview_z_index(image: np.ndarray) -> int | None:
    return int(image.shape[1] // 2) if image.ndim == 4 else None


def _slice_for_preview(array: np.ndarray, z_index: int | None) -> np.ndarray:
    if z_index is not None and array.ndim >= 3:
        return array[z_index]
    return array


def _image_preview_rgb(image: np.ndarray, z_index: int | None = None) -> np.ndarray:
    if z_index is not None:
        image = image[:, z_index]
    if image.shape[0] >= 3:
        channels = [_normalize_uint8(image[channel]) for channel in range(3)]
        return np.stack(channels, axis=-1)
    gray = _normalize_uint8(image[0])
    return np.stack([gray, gray, gray], axis=-1)


def _label_to_rgb(labels: np.ndarray) -> np.ndarray:
    labels = labels.astype(np.int64, copy=False)
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    nonzero = labels != 0
    rgb[..., 0] = ((labels * 37) % 255).astype(np.uint8)
    rgb[..., 1] = ((labels * 73) % 255).astype(np.uint8)
    rgb[..., 2] = ((labels * 109) % 255).astype(np.uint8)
    rgb[~nonzero] = 0
    return rgb


def _predictions_to_visual_targets(task: str, logits: torch.Tensor, validity: torch.Tensor | None = None) -> np.ndarray:
    if task == "multiclass_semantic":
        return torch.argmax(logits, dim=1).numpy()
    probabilities = torch.sigmoid(logits).numpy()
    if validity is not None:
        probabilities = np.where(validity.detach().cpu().numpy(), probabilities, 0)
    if task == "instance_friendly":
        predictions = []
        for item in probabilities:
            processed = postprocess_instance(
                item[0], item[1], item[2] if item.shape[0] >= 3 else None, threshold=0.5, min_object_size=0
            )
            predictions.append(processed["labels"])
        return np.stack(predictions, axis=0)
    return (probabilities[:, 0] >= 0.5).astype(np.uint8)


def _target_preview_rgb(
    task: str, target: np.ndarray | dict[str, np.ndarray], index: int, z_index: int | None = None
) -> np.ndarray:
    if isinstance(target, dict) and "semantic" in target:
        target = target["semantic"]
    if isinstance(target, dict):
        instances = target.get("instances")
        if instances is not None:
            return _label_to_rgb(_slice_for_preview(instances[index, 0], z_index))
        foreground = target.get("foreground")
        if foreground is None:
            raise ValueError("Instance preview target is missing foreground")
        return _label_to_rgb(_slice_for_preview((foreground[index, 0] > 0.5).astype(np.uint8), z_index))
    if task == "binary_semantic":
        return _label_to_rgb(_slice_for_preview((target[index, 0] > 0.5).astype(np.uint8), z_index))
    return _label_to_rgb(_slice_for_preview(target[index], z_index))


def _prediction_preview_rgb(task: str, prediction: np.ndarray, z_index: int | None = None) -> np.ndarray:
    prediction = _slice_for_preview(prediction, z_index)
    if task == "binary_semantic":
        return _label_to_rgb(prediction.astype(np.uint8))
    return _label_to_rgb(prediction.astype(np.uint32))


def _overlay_prediction(image_rgb: np.ndarray, prediction_rgb: np.ndarray) -> np.ndarray:
    mask = np.any(prediction_rgb != 0, axis=-1)
    overlay = image_rgb.copy()
    overlay[mask] = np.clip(0.6 * overlay[mask] + 0.4 * prediction_rgb[mask], 0, 255).astype(np.uint8)
    return overlay


def train(config: dict[str, Any] | TrainingConfig, task: Any = None) -> dict[str, Any]:
    try:
        if CallbackDispatcher(task).cancel_requested():
            raise PlanningCancelled()
        with image_reading_session(CallbackDispatcher(task).emit):
            return _train(config, task)
    except TrainingStopped as exc:
        return exc.result
    except PlanningCancelled:
        payload: dict[str, Any] = {"cancelled": True, "phase": "planning"}
        CallbackDispatcher(task).emit("cancelled", message="Training cancelled during dataset planning", **payload)
        return payload
    except Exception as exc:
        CallbackDispatcher(task).emit("error", message=str(exc), error_class=type(exc).__name__)
        raise
    finally:
        output = config.output_dir if isinstance(config, TrainingConfig) else config.get("output_dir")
        logger = logging.Logger.manager.loggerDict.get(f"jdll_unet.training.{output}")
        if isinstance(logger, logging.Logger):
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)


def _train(config: dict[str, Any] | TrainingConfig, task: Any = None) -> dict[str, Any]:
    source_model: SourceModel | None = None
    if isinstance(config, TrainingConfig):
        config = config.request_dict()
    resolved_request: dict[str, Any] | TrainingConfig = config
    if isinstance(config, dict) and str(config.get("starting_point", "scratch")) in {"fine_tune", "finetune"}:
        if config.get("base_model") is None:
            raise ModelLoadError("base_model is required when starting_point is fine_tune")
        source_model = resolve_source_model(Path(config["base_model"]))
        requested_architecture = config.get("architecture")
        if requested_architecture not in (None, AUTO) and str(requested_architecture) != source_model.architecture.name:
            raise ModelLoadError(
                f"Fine-tuning architecture {requested_architecture!r} disagrees with source architecture "
                f"{source_model.architecture.name!r}"
            )
        resolved = dict(config)
        resolved["architecture"] = source_model.architecture.name
        if resolved.get("context_slices", AUTO) == AUTO:
            resolved["context_slices"] = source_model.architecture.context_slices
        for key, inherited_value in (
            ("deep_supervision", source_model.architecture.deep_supervision),
            ("model_normalization", source_model.architecture.normalization),
        ):
            if key in resolved and resolved[key] not in (None, AUTO, inherited_value):
                raise ModelLoadError(f"Fine-tuning must preserve source {key}={inherited_value!r}")
            resolved[key] = inherited_value
        if resolved.get("normalization", AUTO) == AUTO:
            resolved["normalization"] = source_model.model_config.get("normalization")
        for key in ("context", "spacing", "instance_scale_normalization"):
            if source_model.architecture.dimensions == "2d" and key in {"context", "spacing"}:
                continue
            saved = source_model.model_config.get("training", {}).get(key)
            if isinstance(saved, dict):
                requested = resolved.get(key)
                if requested is None or requested == AUTO:
                    resolved[key] = saved
                elif isinstance(requested, dict):
                    resolved[key] = {**saved, **{k: v for k, v in requested.items() if v != AUTO}}
        resolved_request = resolved
    train_config = parse_training_config(
        resolved_request, source_architecture=source_model.architecture if source_model else None
    )
    if source_model is None and train_config.starting_point in {"fine_tune", "finetune"}:
        assert train_config.base_model is not None
        source_model = resolve_source_model(train_config.base_model)
    output_dir = train_config.output_dir
    callbacks = CallbackDispatcher(task)
    logger = _setup_logging(output_dir)
    _set_seed(train_config.seed)
    device = resolve_device(train_config.device)
    logger.info("Starting training on device=%s", device)
    if source_model is not None:
        message = f"Fine-tuning from base LR {source_model.base_learning_rate:g}: backbone LR {source_model.base_learning_rate * 0.1:g}; adapted layers use {source_model.base_learning_rate:g} when present."
        if source_model.learning_rate_provenance == "fallback_default":
            message = "Original scratch LR could not be recovered; using fallback base LR 0.001. " + message
        logger.info(message)
        callbacks.emit(
            "warning" if source_model.learning_rate_provenance == "fallback_default" else "source_model_resolved",
            message=message,
            base_learning_rate=source_model.base_learning_rate,
            learning_rate_provenance=source_model.learning_rate_provenance,
            recovery_sources=source_model.recovery_sources,
        )

    architecture_probe = (
        source_model.architecture
        if source_model is not None
        else architecture_defaults(train_config.architecture, normalization=train_config.model_normalization)
    )

    def emit_plan(event_type: str, **payload: Any) -> bool:
        logger.info("%s: %s", event_type, payload.get("message", ""))
        return callbacks.emit(event_type, **payload)

    current_read_session().emit = emit_plan

    active_training: dict[str, Any] = {}

    def check_cancel() -> None:
        if callbacks.cancel_requested():
            if active_training:
                raise TrainingStopped(
                    _cancel_training(
                        callbacks,
                        output_dir,
                        model,
                        optimizer,
                        lr_scheduler,
                        active_training["epoch"],
                        active_training["step"],
                        detected_task,
                        arch,
                        model_config,
                    )
                )
            raise PlanningCancelled()

    geometry = resolve_training_geometry(
        train_config,
        architecture_probe,
        inherited=source_model is not None,
        device=device,
        available_memory=_available_memory_bytes(device),
        emit=emit_plan,
        check_cancel=check_cancel,
    )
    train_pairs, val_pairs = geometry.train, geometry.val
    dimensions = geometry.architecture.dimensions
    detected_task = geometry.task
    info = geometry.info
    dataset_plan = geometry.spacing
    case_spacings = {case.case: case.spacing for case in dataset_plan.cases}
    for case in dataset_plan.cases:
        write_json(
            output_dir / "resolved_spacings" / f"{case.case}.json",
            {"spacing": case.spacing, "source": case.source, "original_spacing": case.original_spacing},
        )
    _nonempty_train_pairs, empty_train_pairs = partition_empty_pairs(train_pairs, dimensions)
    _nonempty_val_pairs, empty_val_pairs = partition_empty_pairs(val_pairs, dimensions)
    source_input_channels = info.input_channels
    resolved_context_slices = geometry.architecture.context_slices
    output_channels = geometry.architecture.output_channels
    label_values = [1] if detected_task == "binary_semantic" else info.label_values
    deep_supervision = geometry.architecture.deep_supervision
    preferred_patch = geometry.memory.preferred_patch
    memory_plan = geometry.memory
    patch_size = memory_plan.resolved_patch
    batch_size = memory_plan.microbatch_cap
    microbatch_size = memory_plan.resolved_microbatch
    assert isinstance(patch_size, tuple)
    dataset_fingerprint = dataset_plan.to_dict()
    if detected_task in {"binary_semantic", "multiclass_semantic"}:

        def diagnostic_masks():
            for pair in train_pairs:
                check_cancel()
                mask = load_domain_mask(pair, dimensions=dimensions)
                if dimensions == "3d" and dataset_plan.target_spacing is not None:
                    mask = resample_mask(mask, case_spacings[pair.stem], dataset_plan.target_spacing)
                yield mask

        dataset_fingerprint["semantic_scale_diagnostics"] = semantic_scale_diagnostics(
            diagnostic_masks(),
            dimensions=dimensions,
            patch_size=patch_size,
            label_values=label_values,
        )
    write_json(output_dir / "dataset_fingerprint.json", dataset_fingerprint)
    scale_cfg = train_config.instance_scale_normalization
    instance_scale_enabled = detected_task == "instance_friendly" and scale_cfg.enabled
    train_instance_sizes: dict[str, float] = {}
    val_instance_sizes: dict[str, float] = {}
    training_scale_estimates: list[InstanceSizeEstimate] = []
    validation_scale_estimates: list[InstanceSizeEstimate] = []
    fallback_instance_size: float | None = None
    repaired_instance_components = 0
    if instance_scale_enabled:
        for split_pairs_, destination, estimates, seed_offset in (
            (train_pairs, train_instance_sizes, training_scale_estimates, 0),
            (val_pairs, val_instance_sizes, validation_scale_estimates, 100_000),
        ):
            for pair in split_pairs_:
                check_cancel()
                estimate, repairs = geometry.instance_estimates[(pair.stem, pair.region)]
                repaired_instance_components += repairs
                if estimate is not None:
                    destination[pair.stem] = estimate.median_diameter_px
                    estimates.append(estimate)
                    logger.info(
                        "Instance size split=%s image=%s sampled=%s available=%s median_diameter_px=%.3f",
                        "training" if seed_offset == 0 else "validation",
                        pair.image.name,
                        estimate.sampled_instances,
                        estimate.available_instances,
                        estimate.median_diameter_px,
                    )
                else:
                    logger.warning(
                        "No valid instances for scale estimation split=%s image=%s; training median fallback will be used",
                        "training" if seed_offset == 0 else "validation",
                        pair.image.name,
                    )
        if not training_scale_estimates:
            raise DatasetError(
                "Instance scale normalization could not measure any valid training instances; "
                "check masks or disable border exclusion"
            )
        fallback_instance_size = float(
            np.median([estimate.median_diameter_px for estimate in training_scale_estimates])
        )
        target_extent = (
            min(size * spacing for size, spacing in zip(patch_size, dataset_plan.target_spacing, strict=True))
            if dimensions == "3d" and dataset_plan.target_spacing is not None
            else min(patch_size)
        )
        target_diameter = float(scale_cfg.target_object_fraction * target_extent)
        canonical_scales = np.asarray(
            [target_diameter / estimate.median_diameter_px for estimate in training_scale_estimates], dtype=np.float64
        )
        dataset_statistics = {
            "instance_scale_statistics": {
                "training": aggregate_instance_statistics(training_scale_estimates),
                "validation": aggregate_instance_statistics(validation_scale_estimates),
                "training_images_without_valid_instances": len(train_pairs) - len(train_instance_sizes),
                "validation_images_without_valid_instances": len(val_pairs) - len(val_instance_sizes),
                "disconnected_instance_components_relabelled": repaired_instance_components,
                "canonical_scale": {
                    "median": float(np.median(canonical_scales)),
                    "minimum": float(canonical_scales.min()),
                    "maximum": float(canonical_scales.max()),
                    "images_below_minimum_clamp": int(
                        np.count_nonzero(canonical_scales < scale_cfg.min_effective_scale)
                    ),
                    "images_above_maximum_clamp": int(
                        np.count_nonzero(canonical_scales > scale_cfg.max_effective_scale)
                    ),
                },
            }
        }
        write_json(output_dir / "dataset_statistics.json", dataset_statistics)
        logger.info(
            "Instance scale normalization enabled: target_diameter_px=%.3f training_median_px=%.3f "
            "jitter=%s effective_scale=[%.3f, %.3f]",
            target_diameter,
            fallback_instance_size,
            scale_cfg.training_scale_jitter,
            scale_cfg.min_effective_scale,
            scale_cfg.max_effective_scale,
        )
    accumulation_steps = math.ceil(train_config.effective_batch_size / microbatch_size)
    resolved_effective_batch = microbatch_size * accumulation_steps
    steps_per_epoch = (
        max(
            train_config.minimum_steps_per_epoch,
            math.ceil(train_config.expected_patches_per_case * len(train_pairs) / resolved_effective_batch),
        )
        if train_config.steps_per_epoch == AUTO
        else int(train_config.steps_per_epoch)
    )
    if source_model is not None:
        backbone_learning_rate, adapted_learning_rate = resolve_finetune_learning_rates(source_model.base_learning_rate)
        learning_rate = backbone_learning_rate
    else:
        learning_rate = (
            default_learning_rate(train_config.optimizer)
            if train_config.learning_rate == AUTO
            else float(train_config.learning_rate)
        )
        backbone_learning_rate = learning_rate
        adapted_learning_rate = None
    base_learning_rate = source_model.base_learning_rate if source_model is not None else learning_rate
    foreground_oversampling = (
        True if train_config.foreground_oversampling == AUTO else bool(train_config.foreground_oversampling)
    )
    foreground_probability = (
        default_foreground_probability(detected_task, train_config.architecture)
        if train_config.foreground_probability == AUTO
        else float(train_config.foreground_probability)
    )
    augmentation_profile = (
        default_augmentation_profile(train_config.architecture, device)
        if train_config.augmentation_profile == AUTO
        else train_config.augmentation_profile
    )
    mixed_precision = default_mixed_precision(train_config.mixed_precision, device)
    progress_update_interval = (
        default_progress_update_interval(device)
        if train_config.progress_update_interval == AUTO
        else int(train_config.progress_update_interval)
    )
    log_update_interval = (
        default_log_update_interval(device)
        if train_config.log_update_interval == AUTO
        else int(train_config.log_update_interval)
    )
    effective_loss_weights, target_sparsity = _resolve_loss_weights(
        train_config, train_pairs, detected_task, dimensions
    )
    logger.info("loss_weights=%s target_sparsity=%s", effective_loss_weights, target_sparsity)

    arch = geometry.architecture
    model = build_unet(arch).to(device)
    finetune_report: FineTuneReport | None = None
    adapted_parameter_names: set[str] = set()
    if source_model is not None:
        finetune_report, adapted_parameter_names = initialize_finetune_model(
            model,
            source_model,
            target_task=detected_task,
            target_label_values=label_values,
            backbone_learning_rate=backbone_learning_rate,
            adapted_learning_rate=cast(float, adapted_learning_rate),
        )
        logger.info(
            "Initialized fine-tuning from %s: input=%s output=%s adapted=%s reinitialized=%s",
            source_model.checkpoint_path,
            finetune_report.input_adaptation,
            finetune_report.output_adaptation,
            finetune_report.adapted_tensors,
            finetune_report.reinitialized_tensors,
        )

    resolved_context_policy = geometry.context_policy
    resolved_context_spacing = geometry.context_spacing
    train_dataset = make_dataset(
        train_pairs,
        detected_task,
        label_values=label_values,
        normalization=train_config.normalization,
        profile=augmentation_profile,
        patch_size=patch_size,
        foreground_oversampling=foreground_oversampling,
        foreground_probability=foreground_probability,
        augmentation_overrides=train_config.augmentation,
        training=True,
        dimensions=dimensions,
        seed=train_config.seed,
        instance_sizes=train_instance_sizes,
        fallback_instance_size=fallback_instance_size,
        context_slices=resolved_context_slices,
        context_stride_policy=resolved_context_policy,
        context_stride=train_config.context.stride,
        context_target_spacing=resolved_context_spacing,
        case_spacings=case_spacings,
        target_spacing=dataset_plan.target_spacing,
        sample_count=steps_per_epoch * microbatch_size * accumulation_steps,
        max_empty_plane_fraction=train_config.max_empty_plane_fraction,
    )
    train_dataset.augmentation.skip_empty_patches = train_config.skip_empty_patches
    train_dataset.augmentation.empty_patch_max_retries = train_config.empty_patch_max_retries
    train_dataset.augmentation.include_empty_patches_after_max_retries = (
        train_config.include_empty_patches_after_max_retries and not train_config.skip_empty_patches
    )
    train_dataset.augmentation.max_padding_ratio = train_config.max_padding_ratio
    if instance_scale_enabled:
        train_dataset.augmentation.instance_scale_enabled = True
        train_dataset.augmentation.target_object_diameter_px = target_diameter
        train_dataset.augmentation.training_scale_jitter = scale_cfg.training_scale_jitter
        train_dataset.augmentation.min_effective_scale = scale_cfg.min_effective_scale
        train_dataset.augmentation.max_effective_scale = scale_cfg.max_effective_scale
    val_dataset = make_dataset(
        val_pairs,
        detected_task,
        label_values=label_values,
        normalization=train_config.normalization,
        profile="fast",
        patch_size=patch_size,
        foreground_oversampling=False,
        foreground_probability=0.0,
        augmentation_overrides={},
        training=False,
        dimensions=dimensions,
        seed=train_config.seed + 10_000,
        instance_sizes=val_instance_sizes,
        fallback_instance_size=fallback_instance_size,
        context_slices=resolved_context_slices,
        context_stride_policy=resolved_context_policy,
        context_stride=train_config.context.stride,
        context_target_spacing=resolved_context_spacing,
        case_spacings=case_spacings,
        target_spacing=dataset_plan.target_spacing,
    )
    if instance_scale_enabled:
        val_dataset.augmentation.instance_scale_enabled = True
        val_dataset.augmentation.target_object_diameter_px = target_diameter
        val_dataset.augmentation.min_effective_scale = scale_cfg.min_effective_scale
        val_dataset.augmentation.max_effective_scale = scale_cfg.max_effective_scale
    val_dataset.augmentation.max_padding_ratio = train_config.max_padding_ratio
    train_dataset.set_epoch(1)
    resolved_dataset_plan = geometry.to_dict()
    resolved_dataset_plan.update(
        padding_policy={
            "max_padding_ratio_per_side": train_config.max_padding_ratio,
            "spatial_validity": "real_support_only",
            "infeasible_transform_fallback": "bounded_scale_clamp_or_identity_spatial_transform",
        },
        sampling_policy={
            "unit": "eligible_planes" if dimensions in {"2d", "2.5d"} else "volume_patches",
            "max_empty_plane_fraction": train_config.max_empty_plane_fraction,
            "seed": train_config.seed,
            "epoch_sample_budget": len(train_dataset),
            "pool_size": train_dataset.pool_size,
        },
        epoch_sampling=train_dataset.sampling_summary,
        training_instance_sizes=train_instance_sizes,
        validation_instance_sizes=val_instance_sizes,
    )
    write_json(output_dir / "dataset_plan.json", resolved_dataset_plan)
    callbacks.emit(
        "dataset_summary",
        message=f"{dimensions} training: using {len(train_pairs)} training sources, {len(val_pairs)} validation domains, and {train_dataset.pool_size} eligible training entries.",
        training_source_count=len(train_pairs),
        validation_source_count=len(val_pairs),
        pool_size=train_dataset.pool_size,
        epoch_sample_budget=len(train_dataset),
        dataset_plan_path=str(output_dir / "dataset_plan.json"),
        source_counts={
            split: {
                kind: sum(
                    record.get("source_kind") == kind and record.get("requested_split") == split
                    for record in geometry.records
                )
                for kind in ("image_2d", "singleton_stack", "volume")
            }
            for split in ("train", "val")
        },
        validation_pool_size=len(val_dataset),
        light_validation_sample_budget=min(len(val_dataset), train_config.validation.light_steps * microbatch_size),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=microbatch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=microbatch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=device.type == "cuda",
    )
    optimizer_cls = torch.optim.AdamW if train_config.optimizer == "adamw" else torch.optim.Adam
    if adapted_parameter_names:
        adapted_parameters = [
            parameter for name, parameter in model.named_parameters() if name in adapted_parameter_names
        ]
        backbone_parameters = [
            parameter for name, parameter in model.named_parameters() if name not in adapted_parameter_names
        ]
        optimizer = optimizer_cls(
            [
                {"params": backbone_parameters, "lr": backbone_learning_rate, "group": "backbone"},
                {
                    "params": adapted_parameters,
                    "lr": cast(float, adapted_learning_rate),
                    "group": "adapted_layers",
                },
            ],
            weight_decay=train_config.weight_decay,
        )
    else:
        optimizer = optimizer_cls(
            [{"params": list(model.parameters()), "lr": backbone_learning_rate, "group": "backbone"}],
            weight_decay=train_config.weight_decay,
        )
    total_steps = steps_per_epoch * train_config.epochs
    lr_scheduler = LearningRateScheduler(
        optimizer, train_config.lr_scheduler, total_steps=total_steps, total_epochs=train_config.epochs
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)
    except AttributeError:  # pragma: no cover - older torch fallback
        scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)
    model_config = model_folder_config(
        train_config,
        detected_task,
        arch,
        input_axes=("czyx" if source_input_channels > 1 else "zyx")
        if dimensions in {"3d", "2.5d"}
        else ("cyx" if source_input_channels > 1 else "yx"),
        output_axes="zyx" if dimensions in {"3d", "2.5d"} else "yx",
        label_values=label_values,
    )
    model_config["training"].update(
        {
            "resolved_device": device.type,
            "patch_size": list(patch_size),
            "preferred_patch_size": list(preferred_patch),
            "batch_size": batch_size,
            "microbatch_size": microbatch_size,
            "accumulation_steps": accumulation_steps,
            "effective_batch_size": resolved_effective_batch,
            "steps_per_epoch": steps_per_epoch,
            "learning_rate": learning_rate,
            "base_learning_rate": base_learning_rate,
            "learning_rate_provenance": source_model.learning_rate_provenance
            if source_model
            else "scratch_initial_learning_rate",
            "starting_point": "fine_tune" if source_model is not None else "scratch",
            "source_model": str(source_model.requested_path) if source_model is not None else None,
            "source_checkpoint": str(source_model.checkpoint_path) if source_model is not None else None,
            "source_learning_rate": source_model.learning_rate if source_model is not None else None,
            "backbone_learning_rate": backbone_learning_rate,
            "adapted_layers_learning_rate": (
                finetune_report.adapted_layers_learning_rate if finetune_report is not None else None
            ),
            "fine_tuning_initialization": finetune_report.to_dict() if finetune_report is not None else None,
            "model_normalization": train_config.model_normalization,
            "foreground_oversampling": foreground_oversampling,
            "foreground_probability": foreground_probability,
            "skip_empty_images": train_config.skip_empty_images,
            "skip_empty_patches": train_config.skip_empty_patches,
            "empty_patch_max_retries": train_config.empty_patch_max_retries,
            "include_empty_patches_after_max_retries": train_config.include_empty_patches_after_max_retries,
            "empty_training_images": len(empty_train_pairs),
            "empty_validation_images": len(empty_val_pairs),
            "augmentation_profile": augmentation_profile,
            "mixed_precision": mixed_precision,
            "deep_supervision": deep_supervision,
            "effective_loss_weights": effective_loss_weights,
            "target_sparsity": target_sparsity,
            "lr_scheduler": lr_scheduler.config_dict(),
            "task": detected_task,
            "axes": model_config["input_axes"],
            "device": str(device),
            "input_channels": source_input_channels,
            "output_classes": output_channels,
            "context_slices": resolved_context_slices if dimensions == "2.5d" else None,
            "context": {
                "stride_policy": resolved_context_policy,
                "stride": train_config.context.stride,
                "spacing": resolved_context_spacing,
            },
            "spacing": {**asdict(train_config.spacing), "target_spacing": dataset_plan.target_spacing},
            "progress_update_interval": progress_update_interval,
            "log_update_interval": log_update_interval,
            "augmentation": asdict(train_dataset.augmentation),
            "instance_scale_normalization": {**asdict(scale_cfg), "enabled": instance_scale_enabled},
            "target_object_size": target_diameter if instance_scale_enabled else None,
            "dataset_plan_path": str(output_dir / "dataset_plan.json"),
        }
    )
    write_json(output_dir / "config.json", model_config)
    model_metadata = {
        "format_version": 1,
        "architecture": arch.name,
        "preset_reference_memory_gb": arch.reference_memory_gb,
        "base_learning_rate": base_learning_rate,
        "source_learning_rate": source_model.learning_rate if source_model else None,
        "backbone_learning_rate": backbone_learning_rate,
        "output_channels": (
            ["foreground", "boundary", "distance"] if detected_task == "instance_friendly" else ["logits"]
        ),
        "dataset_fingerprint_path": str(output_dir / "dataset_fingerprint.json"),
        "dataset_plan": dataset_plan.to_dict(),
        "semantic_scale_diagnostics": dataset_fingerprint.get("semantic_scale_diagnostics"),
        "resolved_context": {
            "stride_policy": resolved_context_policy,
            "target_spacing": resolved_context_spacing,
        },
        "runtime_plan": {
            "microbatch_size": microbatch_size,
            "accumulation_steps": accumulation_steps,
            "effective_batch_size": resolved_effective_batch,
            "memory": memory_plan.to_dict(),
        },
        "fine_tuning_initialization": finetune_report.to_dict() if finetune_report is not None else None,
        "instance_scale": (
            {
                "target_object_size": target_diameter,
                "size_unit": "physical" if dimensions == "3d" else "pixels",
                "measure": scale_cfg.object_size_measure,
            }
            if instance_scale_enabled
            else None
        ),
    }
    write_json(output_dir / "model_metadata.json", model_metadata)
    callbacks.emit(
        "training_plan",
        message="UNet training plan resolved",
        starting_point="fine_tune" if source_model is not None else "scratch",
        source_model=str(source_model.requested_path) if source_model is not None else None,
        source_architecture=source_model.architecture.name if source_model is not None else None,
        architecture=arch.name,
        source_input_channels=source_model.architecture.input_channels if source_model is not None else None,
        target_input_channels=arch.input_channels,
        source_output_channels=source_model.architecture.output_channels if source_model is not None else None,
        target_output_channels=arch.output_channels,
        input_adaptation=finetune_report.input_adaptation if finetune_report is not None else None,
        output_adaptation=finetune_report.output_adaptation if finetune_report is not None else None,
        source_learning_rate=source_model.learning_rate if source_model is not None else None,
        backbone_learning_rate=backbone_learning_rate,
        base_learning_rate=base_learning_rate,
        learning_rate_provenance=source_model.learning_rate_provenance
        if source_model
        else "scratch_initial_learning_rate",
        dataset_plan_path=str(output_dir / "dataset_plan.json"),
        config_path=str(output_dir / "config.json"),
        max_padding_ratio=train_config.max_padding_ratio,
        max_empty_plane_fraction=train_config.max_empty_plane_fraction,
        adapted_layers_learning_rate=(
            finetune_report.adapted_layers_learning_rate if finetune_report is not None else None
        ),
        dimensions=dimensions,
        patch_size=list(patch_size),
        preferred_patch_size=list(preferred_patch),
        context_slices=resolved_context_slices if dimensions == "2.5d" else None,
        context_stride_policy=resolved_context_policy if dimensions == "2.5d" else None,
        deep_supervision=deep_supervision,
        microbatch_size=microbatch_size,
        accumulation_steps=accumulation_steps,
        effective_batch_size=resolved_effective_batch,
        memory_plan=memory_plan.to_dict(),
        steps_per_epoch=steps_per_epoch,
        augmentation_profile=augmentation_profile,
    )

    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    full_validation_best = -float("inf")
    full_validation_bad = 0
    global_step = 0
    latest_preview_path: str | None = None
    for epoch in range(1, train_config.epochs + 1):
        active_training.update(epoch=epoch, step=global_step)
        check_cancel()
        train_dataset.set_epoch(epoch)
        write_json(
            output_dir / "sampling" / f"epoch_{epoch:04d}.json",
            {"epoch": epoch, "sources": train_dataset.sampling_summary},
        )
        model.train()
        train_losses: list[dict[str, float]] = []
        optimizer.zero_grad(set_to_none=True)
        for microstep, (images, target_batch) in enumerate(train_loader, start=1):
            if callbacks.cancel_requested():
                return _cancel_training(
                    callbacks,
                    output_dir,
                    model,
                    optimizer,
                    lr_scheduler,
                    epoch,
                    global_step,
                    detected_task,
                    arch,
                    model_config,
                )
            images = images.to(device, non_blocking=True)
            target_batch = _move_target(target_batch, device)
            with torch.autocast(device_type="cuda", enabled=mixed_precision):
                logits = model(images)
                loss, components = compute_loss(
                    detected_task,
                    logits,
                    target_batch,
                    effective_loss_weights,
                    focal_gamma=train_config.focal_gamma,
                    focal_alpha=train_config.focal_alpha,
                )
            scaler.scale(loss / accumulation_steps).backward()
            component_floats = _tensor_losses_to_float(components)
            component_floats["total_loss"] = float(loss.detach().cpu().item())
            train_losses.append(component_floats)
            if microstep % accumulation_steps != 0:
                continue
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            active_training["step"] = global_step
            lr_scheduler.step_batch()
            if global_step % log_update_interval == 0:
                logger.info("step=%s/%s epoch=%s train=%s", global_step, total_steps, epoch, component_floats)
            should_emit_step = (
                global_step == 1 or global_step == total_steps or global_step % progress_update_interval == 0
            )
            if should_emit_step and not callbacks.emit(
                "progress",
                message=f"UNet training epoch {epoch}/{train_config.epochs}",
                current=global_step,
                maximum=total_steps,
                epoch=epoch,
                step=global_step,
                total_epochs=train_config.epochs,
                total_steps=total_steps,
                learning_rate=lr_scheduler.current_lr,
                losses={f"train/{key}": value for key, value in component_floats.items()},
                metrics={},
            ):
                return _cancel_training(
                    callbacks,
                    output_dir,
                    model,
                    optimizer,
                    lr_scheduler,
                    epoch,
                    global_step,
                    detected_task,
                    arch,
                    model_config,
                )

        model.eval()
        val_losses: list[dict[str, float]] = []
        val_metrics: list[dict[str, float]] = []
        with torch.inference_mode():
            for val_step, (images, target_batch) in enumerate(val_loader):
                if val_step >= train_config.validation.light_steps:
                    break
                images = images.to(device, non_blocking=True)
                target_batch = _move_target(target_batch, device)
                logits = model(images)
                loss, components = compute_loss(
                    detected_task,
                    logits,
                    target_batch,
                    effective_loss_weights,
                    focal_gamma=train_config.focal_gamma,
                    focal_alpha=train_config.focal_alpha,
                )
                losses = _tensor_losses_to_float(components)
                losses["total_loss"] = float(loss.detach().cpu().item())
                val_losses.append(losses)
                val_metrics.append(compute_metrics(detected_task, logits, target_batch))

        epoch_record: dict[str, Any] = {
            "epoch": epoch,
            "train_losses": _mean_dict(train_losses),
            "val_losses": _mean_dict(val_losses),
            "val_metrics": _mean_dict(val_metrics),
        }
        light_score = primary_metric(detected_task, epoch_record["val_metrics"])
        run_full_validation = train_config.validation.mode == "full" and (
            epoch % train_config.validation.full_every == 0 or epoch == train_config.epochs
        )
        if run_full_validation:
            full_metrics = _full_volume_validation(
                model,
                val_pairs,
                task=detected_task,
                dimensions=dimensions,
                device=device,
                patch_size=patch_size,
                normalization=train_config.normalization,
                context_slices=resolved_context_slices,
                context_policy=resolved_context_policy,
                context_fixed_stride=train_config.context.stride,
                context_target_spacing=resolved_context_spacing,
                case_spacings=case_spacings,
                target_spacing=dataset_plan.target_spacing,
                instance_sizes=val_instance_sizes,
                target_object_size=target_diameter if instance_scale_enabled else None,
                label_values=label_values,
                min_scale=scale_cfg.min_effective_scale,
                max_scale=scale_cfg.max_effective_scale,
                check_cancel=check_cancel,
            )
            epoch_record["full_validation"] = full_metrics
            score = float(full_metrics["mean_dice"])
            if score > full_validation_best:
                full_validation_best = score
                full_validation_bad = 0
            else:
                full_validation_bad += 1
        else:
            score = light_score
        selector_update = train_config.validation.mode == "light" or run_full_validation
        lr_scheduler.step_epoch(score)
        epoch_record["learning_rate"] = lr_scheduler.current_lr
        history.append(epoch_record)
        logger.info(
            "epoch=%s train=%s val=%s metrics=%s",
            epoch,
            epoch_record["train_losses"],
            epoch_record["val_losses"],
            epoch_record["val_metrics"],
        )
        if not callbacks.emit(
            "progress",
            message=f"UNet validation epoch {epoch}",
            current=global_step,
            maximum=total_steps,
            epoch=epoch,
            step=global_step,
            total_epochs=train_config.epochs,
            total_steps=total_steps,
            learning_rate=lr_scheduler.current_lr,
            losses={f"val/{key}": value for key, value in epoch_record["val_losses"].items()},
            metrics={f"val/{key}": value for key, value in epoch_record["val_metrics"].items()},
            last_checkpoint_path=str(output_dir / "weights_last.pt"),
            best_checkpoint_path=str(output_dir / "weights_best.pt")
            if score >= best_score or (output_dir / "weights_best.pt").exists()
            else None,
        ):
            return _cancel_training(
                callbacks,
                output_dir,
                model,
                optimizer,
                lr_scheduler,
                epoch,
                global_step,
                detected_task,
                arch,
                model_config,
            )

        _save_checkpoint(
            output_dir / "weights_last.pt",
            model,
            optimizer,
            lr_scheduler,
            epoch,
            detected_task,
            arch,
            epoch_record,
            model_config,
        )
        if selector_update and score >= best_score:
            best_score = score
            _save_checkpoint(
                output_dir / "weights_best.pt",
                model,
                optimizer,
                lr_scheduler,
                epoch,
                detected_task,
                arch,
                epoch_record,
                model_config,
            )
        write_json(
            output_dir / "metrics.json",
            {"history": history, "best_score": best_score if math.isfinite(best_score) else None},
        )
        preview_event = _save_previews(
            output_dir, epoch, detected_task, model, val_loader, device, train_config.preview_count
        )
        if preview_event is not None:
            latest_preview_path = preview_event["latest_preview_path"]
            callbacks.emit(
                "preview",
                message=f"UNet validation preview epoch {epoch}",
                current=global_step,
                maximum=total_steps,
                epoch=epoch,
                **preview_event,
            )
        if run_full_validation and full_validation_bad >= train_config.validation.early_stopping_patience:
            logger.info("Early stopping after %s full validations without improvement", full_validation_bad)
            break

    best_path = output_dir / "weights_best.pt"
    if not best_path.exists():
        best_path = output_dir / "weights_last.pt"
    _atomic_copy(best_path, output_dir / "model.pt")
    result = {
        "model_dir": str(output_dir),
        "model_path": str(output_dir / "model.pt"),
        "task": detected_task,
        "epochs": train_config.epochs,
        "best_score": best_score,
        "metrics": history[-1] if history else {},
        "loss_weights": effective_loss_weights,
        "target_sparsity": target_sparsity,
        "learning_rate": lr_scheduler.current_lr,
        "lr_scheduler": lr_scheduler.state_dict(),
        "metrics_path": str(output_dir / "metrics.json"),
        "config_path": str(output_dir / "config.json"),
        "dataset_statistics_path": str(output_dir / "dataset_statistics.json") if instance_scale_enabled else None,
        "dataset_fingerprint_path": str(output_dir / "dataset_fingerprint.json"),
        "model_metadata_path": str(output_dir / "model_metadata.json"),
        "dataset_plan_path": str(output_dir / "dataset_plan.json"),
        "latest_preview_path": latest_preview_path,
        "config": model_config,
    }
    callbacks.emit(
        "complete",
        message="UNet training complete",
        current=total_steps,
        maximum=total_steps,
        model_dir=result["model_dir"],
        model_path=result["model_path"],
        task=detected_task,
        epochs=train_config.epochs,
        best_score=best_score,
        learning_rate=lr_scheduler.current_lr,
        metrics_path=result["metrics_path"],
        config_path=result["config_path"],
        latest_preview_path=latest_preview_path,
    )
    logger.info("Training complete: %s", result)
    return result


def _cancel_training(
    callbacks: CallbackDispatcher,
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LearningRateScheduler,
    epoch: int,
    step: int,
    task: str,
    arch: ArchitectureConfig,
    model_config: dict[str, Any],
) -> dict[str, Any]:
    _save_checkpoint(
        output_dir / "weights_last.pt",
        model,
        optimizer,
        scheduler,
        epoch,
        task,
        arch,
        {"cancelled": True, "epoch": epoch, "step": step},
        model_config,
    )
    payload = {
        "cancelled": True,
        "epoch": epoch,
        "step": step,
        "model_dir": str(output_dir),
        "last_checkpoint_path": str(output_dir / "weights_last.pt"),
        "best_checkpoint_path": str(output_dir / "weights_best.pt")
        if (output_dir / "weights_best.pt").exists()
        else None,
    }
    callbacks.emit("cancelled", message="UNet training cancelled", current=step, maximum=step, **payload)
    return payload
