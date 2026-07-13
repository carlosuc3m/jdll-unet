"""Simple Appose-friendly PyTorch training loop."""

from __future__ import annotations

import logging
import os
import random
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import imageio.v3 as imageio
import numpy as np
import torch
from torch.utils.data import DataLoader

from .callbacks import CallbackDispatcher
from .config import (
    AUTO,
    ArchitectureConfig,
    architecture_defaults,
    default_augmentation_profile,
    default_batch_size,
    default_foreground_probability,
    default_learning_rate,
    default_log_update_interval,
    default_mixed_precision,
    default_patch_size,
    default_progress_update_interval,
    model_folder_config,
    parse_training_config,
    resolve_device,
    write_json,
)
from .dataset import inspect_dataset, make_dataset, partition_empty_pairs, split_pairs
from .errors import ConfigError, DatasetError, ModelLoadError
from .io import ImageMaskPair, discover_dataset, load_mask
from .losses import compute_loss, primary_logits
from .metrics import compute_metrics, primary_metric
from .model import build_unet
from .postprocess import postprocess_instance
from .scale import aggregate_instance_statistics, estimate_instance_size, estimate_volume_instance_size
from .schedulers import LearningRateScheduler
from .targets import boundary_target, target_output_channels
from .task_detect import detect_task_from_pairs


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


def _move_target(target: torch.Tensor | dict[str, torch.Tensor], device: torch.device):
    if isinstance(target, dict):
        return {key: value.to(device, non_blocking=True) for key, value in target.items()}
    return target.to(device, non_blocking=True)


def _mean_dict(values: list[dict[str, float]]) -> dict[str, float]:
    if not values:
        return {}
    keys = sorted(set().union(*(item.keys() for item in values)))
    return {key: float(np.mean([item[key] for item in values if key in item])) for key in keys}


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
) -> dict[str, float | int | None]:
    sampled = _sample_pairs(pairs, sample_limit)
    foreground_pixels = 0
    boundary_pixels = 0
    total_pixels = 0
    for pair in sampled:
        mask = load_mask(pair.mask)
        foreground_pixels += int(np.count_nonzero(mask))
        total_pixels += int(mask.size)
        if task == "instance_friendly":
            boundary_pixels += int(np.count_nonzero(boundary_target(mask)))

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

    stats = _estimate_target_sparsity(train_pairs, task, train_config.auto_focal_sample_limit)
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


def _load_base_model(model: torch.nn.Module, base_model: Path, device: torch.device, logger: logging.Logger) -> None:
    checkpoint = base_model / "model.pt" if base_model.is_dir() else base_model
    if not checkpoint.exists():
        raise ModelLoadError(f"Base model checkpoint does not exist: {checkpoint}")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if not isinstance(state, dict):
        raise ModelLoadError(f"Base model checkpoint {checkpoint} is not a valid JDLL UNet checkpoint")
    state_dict = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info("Loaded base model %s (missing=%s unexpected=%s)", checkpoint, missing, unexpected)


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
    saved = []
    model.eval()
    with torch.inference_mode():
        for images, target_batch in loader:
            images = images.to(device)
            logits = primary_logits(model(images)).detach().cpu()
            images_cpu = images.detach().cpu().numpy()
            target_cpu = _target_to_numpy(target_batch)
            predictions = _predictions_to_visual_targets(task, logits)
            for idx in range(images_cpu.shape[0]):
                if len(saved) >= preview_count:
                    break
                base = f"preview_{len(saved):03d}"
                image_path = (preview_dir / f"{base}_image.png").resolve()
                target_path = (preview_dir / f"{base}_target.png").resolve()
                pred_path = (preview_dir / f"{base}_prediction.png").resolve()
                overlay_path = (preview_dir / f"{base}_overlay.png").resolve()
                z_index = _preview_z_index(images_cpu[idx])
                image_rgb = _image_preview_rgb(images_cpu[idx], z_index)
                target_rgb = _target_preview_rgb(task, target_cpu, idx, z_index)
                pred_rgb = _prediction_preview_rgb(task, predictions[idx], z_index)
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


def _predictions_to_visual_targets(task: str, logits: torch.Tensor) -> np.ndarray:
    if task == "multiclass_semantic":
        return torch.argmax(logits, dim=1).numpy()
    probabilities = torch.sigmoid(logits).numpy()
    if task == "instance_friendly":
        predictions = []
        for item in probabilities:
            processed = postprocess_instance(item[0], item[1], threshold=0.5, min_object_size=0)
            predictions.append(processed["labels"])
        return np.stack(predictions, axis=0)
    return (probabilities[:, 0] >= 0.5).astype(np.uint8)


def _target_preview_rgb(task: str, target: np.ndarray | dict[str, np.ndarray], index: int, z_index: int | None = None) -> np.ndarray:
    if isinstance(target, dict):
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


def train(config: dict[str, Any], task: Any = None) -> dict[str, Any]:
    train_config = parse_training_config(config)
    output_dir = train_config.output_dir
    callbacks = CallbackDispatcher(task)
    logger = _setup_logging(output_dir)
    _set_seed(train_config.seed)
    device = resolve_device(train_config.device)
    logger.info("Starting training on device=%s", device)

    splits = discover_dataset(train_config.dataset_path)
    if splits.explicit_val:
        train_pairs, val_pairs = splits.train, splits.val
    else:
        train_pairs, val_pairs = split_pairs(splits.train, train_config.validation_fraction, train_config.seed)
    if len(train_pairs) < 3:
        warning = f"Very small training set: {len(train_pairs)} image(s). Results may be unstable."
        logger.warning(warning)
        callbacks.emit("warning", message=warning)

    detection = detect_task_from_pairs(train_pairs + val_pairs, train_config.dataset_path, requested_task=train_config.task)
    if detection.get("ambiguous"):
        raise ValueError(
            "Dataset task is ambiguous. Ask whether labels represent classes or objects and pass "
            "task='multiclass_semantic' or task='instance_friendly'."
        )
    detected_task = str(detection["task"])
    if detected_task == "unsupported":
        raise ValueError(str(detection.get("reason", "Unsupported annotation type")))

    architecture_probe = architecture_defaults(train_config.architecture, normalization=train_config.model_normalization)
    dimensions = architecture_probe.dimensions
    info = inspect_dataset(train_pairs + val_pairs, dimensions=dimensions)
    nonempty_train_pairs, empty_train_pairs = partition_empty_pairs(train_pairs, dimensions=dimensions)
    _nonempty_val_pairs, empty_val_pairs = partition_empty_pairs(val_pairs, dimensions=dimensions)
    if train_config.skip_empty_images:
        train_pairs = nonempty_train_pairs
    if not train_pairs or not nonempty_train_pairs:
        raise DatasetError("All training masks are empty; at least one foreground annotation is required")
    if empty_train_pairs:
        action = "skipped" if train_config.skip_empty_images else "retained"
        message = f"Empty training masks: {len(empty_train_pairs)} image(s) {action}."
        logger.warning(message)
        callbacks.emit("warning", message=message)
    if empty_val_pairs:
        message = f"Empty validation masks: {len(empty_val_pairs)} image(s) retained."
        logger.warning(message)
        callbacks.emit("warning", message=message)
    source_input_channels = info.input_channels if train_config.input_channels == AUTO else int(train_config.input_channels)
    input_channels = source_input_channels * train_config.context_slices if dimensions == "2.5d" else source_input_channels
    label_values = [1] if detected_task == "binary_semantic" else info.label_values
    output_channels = target_output_channels(detected_task, label_values)
    if train_config.output_classes != AUTO and detected_task == "multiclass_semantic":
        output_channels = int(train_config.output_classes)

    patch_size = (
        default_patch_size(train_config.architecture, info.image_shape[-2:] if dimensions == "2.5d" else info.image_shape)
        if train_config.patch_size == AUTO
        else train_config.patch_size
    )
    assert isinstance(patch_size, tuple)
    scale_cfg = train_config.instance_scale_normalization
    instance_scale_enabled = detected_task == "instance_friendly" and scale_cfg.enabled
    train_instance_sizes: dict[str, float] = {}
    val_instance_sizes: dict[str, float] = {}
    training_scale_estimates = []
    validation_scale_estimates = []
    fallback_instance_size: float | None = None
    repaired_instance_components = 0
    if instance_scale_enabled:
        if dimensions not in {"2d", "2.5d"}:
            raise ConfigError("Instance scale normalization currently supports 2D and 2.5D models only")
        for split_pairs_, destination, estimates, seed_offset in (
            (train_pairs, train_instance_sizes, training_scale_estimates, 0),
            (val_pairs, val_instance_sizes, validation_scale_estimates, 100_000),
        ):
            for index, pair in enumerate(split_pairs_):
                if dimensions == "2.5d":
                    estimate, repair = estimate_volume_instance_size(
                        load_mask(pair.mask, dimensions="2.5d"),
                        max_instances=scale_cfg.max_instances_per_image,
                        exclude_xy_border=scale_cfg.exclude_border_instances,
                        min_instance_area=scale_cfg.min_instance_area,
                        seed=train_config.seed + seed_offset + index,
                    )
                    repaired_instance_components += repair.repaired_components
                    if repair.repaired_components:
                        logger.warning(
                            "Canonicalized %s disconnected instance component(s) in %s",
                            repair.repaired_components,
                            pair.mask.name,
                        )
                else:
                    estimate = estimate_instance_size(
                        load_mask(pair.mask),
                        max_instances=scale_cfg.max_instances_per_image,
                        exclude_border=scale_cfg.exclude_border_instances,
                        min_instance_area=scale_cfg.min_instance_area,
                        seed=train_config.seed + seed_offset + index,
                    )
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
        target_diameter = float(scale_cfg.target_object_fraction * min(patch_size))
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
                    "images_below_minimum_clamp": int(np.count_nonzero(canonical_scales < scale_cfg.min_effective_scale)),
                    "images_above_maximum_clamp": int(np.count_nonzero(canonical_scales > scale_cfg.max_effective_scale)),
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
    batch_size = default_batch_size(train_config.architecture, device) if train_config.batch_size == AUTO else int(train_config.batch_size)
    learning_rate = (
        default_learning_rate(train_config.optimizer)
        if train_config.learning_rate == AUTO
        else float(train_config.learning_rate)
    )
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
        default_log_update_interval(device) if train_config.log_update_interval == AUTO else int(train_config.log_update_interval)
    )
    deep_supervision = (
        train_config.architecture.lower().startswith(("resenc", "residual"))
        if train_config.deep_supervision == AUTO
        else bool(train_config.deep_supervision)
    )
    effective_loss_weights, target_sparsity = _resolve_loss_weights(train_config, train_pairs, detected_task)
    logger.info("loss_weights=%s target_sparsity=%s", effective_loss_weights, target_sparsity)

    arch = architecture_defaults(
        train_config.architecture,
        input_channels=input_channels,
        output_channels=output_channels,
        normalization=train_config.model_normalization,
        deep_supervision=deep_supervision,
    )
    arch.context_slices = train_config.context_slices
    model = build_unet(arch).to(device)
    if train_config.starting_point in {"fine_tune", "finetune"} and train_config.base_model is not None:
        _load_base_model(model, train_config.base_model, device, logger)

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
        context_slices=train_config.context_slices,
    )
    train_dataset.augmentation.skip_empty_patches = train_config.skip_empty_patches
    train_dataset.augmentation.empty_patch_max_retries = train_config.empty_patch_max_retries
    train_dataset.augmentation.include_empty_patches_after_max_retries = (
        train_config.include_empty_patches_after_max_retries
    )
    if instance_scale_enabled:
        train_dataset.augmentation.instance_scale_enabled = True
        train_dataset.augmentation.target_object_diameter_px = float(scale_cfg.target_object_fraction * min(patch_size))
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
        context_slices=train_config.context_slices,
    )
    if instance_scale_enabled:
        val_dataset.augmentation.instance_scale_enabled = True
        val_dataset.augmentation.target_object_diameter_px = float(scale_cfg.target_object_fraction * min(patch_size))
        val_dataset.augmentation.min_effective_scale = scale_cfg.min_effective_scale
        val_dataset.augmentation.max_effective_scale = scale_cfg.max_effective_scale
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        pin_memory=device.type == "cuda",
    )
    optimizer_cls = torch.optim.AdamW if train_config.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_cls(model.parameters(), lr=learning_rate, weight_decay=train_config.weight_decay)
    total_steps = len(train_loader) * train_config.epochs
    lr_scheduler = LearningRateScheduler(optimizer, train_config.lr_scheduler, total_steps=total_steps)
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
            "batch_size": batch_size,
            "learning_rate": learning_rate,
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
        }
    )
    write_json(output_dir / "config.json", model_config)

    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    global_step = 0
    latest_preview_path: str | None = None
    for epoch in range(1, train_config.epochs + 1):
        model.train()
        train_losses: list[dict[str, float]] = []
        for images, target_batch in train_loader:
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
            global_step += 1
            images = images.to(device, non_blocking=True)
            target_batch = _move_target(target_batch, device)
            optimizer.zero_grad(set_to_none=True)
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
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step_batch()
            component_floats = _tensor_losses_to_float(components)
            component_floats["total_loss"] = float(loss.detach().cpu().item())
            train_losses.append(component_floats)
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
            for images, target_batch in val_loader:
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

        epoch_record = {
            "epoch": epoch,
            "train_losses": _mean_dict(train_losses),
            "val_losses": _mean_dict(val_losses),
            "val_metrics": _mean_dict(val_metrics),
        }
        score = primary_metric(detected_task, epoch_record["val_metrics"])
        lr_scheduler.step_epoch(score)
        epoch_record["learning_rate"] = lr_scheduler.current_lr
        history.append(epoch_record)
        logger.info("epoch=%s train=%s val=%s metrics=%s", epoch, epoch_record["train_losses"], epoch_record["val_losses"], epoch_record["val_metrics"])
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
            best_checkpoint_path=str(output_dir / "weights_best.pt") if score >= best_score or (output_dir / "weights_best.pt").exists() else None,
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
        if score >= best_score:
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
        write_json(output_dir / "metrics.json", {"history": history, "best_score": best_score})
        preview_event = _save_previews(output_dir, epoch, detected_task, model, val_loader, device, train_config.preview_count)
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
        "best_checkpoint_path": str(output_dir / "weights_best.pt") if (output_dir / "weights_best.pt").exists() else None,
    }
    callbacks.emit("cancelled", message="UNet training cancelled", current=step, maximum=step, **payload)
    return payload
