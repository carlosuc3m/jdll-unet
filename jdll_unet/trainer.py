"""Simple Appose-friendly PyTorch training loop."""

from __future__ import annotations

import logging
import os
import random
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import (
    AUTO,
    ArchitectureConfig,
    architecture_defaults,
    default_augmentation_profile,
    default_batch_size,
    default_foreground_probability,
    default_learning_rate,
    default_mixed_precision,
    default_patch_size,
    model_folder_config,
    parse_training_config,
    resolve_device,
    write_json,
)
from .dataset import inspect_dataset, make_dataset, split_pairs
from .errors import ModelLoadError
from .io import discover_dataset
from .losses import compute_loss
from .metrics import compute_metrics, primary_metric
from .model import build_unet
from .targets import target_output_channels
from .task_detect import detect_task_from_pairs


def _emit(task: Any, payload: dict[str, Any]) -> None:
    if task is None:
        return
    if callable(task):
        task(payload)
        return
    update = getattr(task, "update", None)
    if callable(update):
        update(payload)


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


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
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
) -> None:
    if preview_count <= 0:
        return
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    model.eval()
    with torch.inference_mode():
        for images, _target in loader:
            images = images.to(device)
            logits = model(images).detach().cpu()
            images_cpu = images.detach().cpu().numpy()
            if task == "multiclass_semantic":
                predictions = torch.softmax(logits, dim=1).numpy()
            else:
                predictions = torch.sigmoid(logits).numpy()
            for idx in range(images_cpu.shape[0]):
                if len(saved) >= preview_count:
                    break
                base = f"preview_{len(saved):03d}"
                image_path = preview_dir / f"{base}_image.npy"
                pred_path = preview_dir / f"{base}_prediction.npy"
                np.save(image_path, images_cpu[idx])
                np.save(pred_path, predictions[idx])
                saved.append({"image": image_path.name, "prediction": pred_path.name})
            if len(saved) >= preview_count:
                break
    payload = {"epoch": epoch, "items": saved}
    write_json(preview_dir / f"epoch_{epoch:04d}.json", payload)
    write_json(preview_dir / "latest.json", payload)


def train(config: dict[str, Any], task: Any = None) -> dict[str, Any]:
    train_config = parse_training_config(config)
    output_dir = train_config.output_dir
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
        _emit(task, {"type": "warning", "message": warning})

    detection = detect_task_from_pairs(train_pairs + val_pairs, train_config.dataset_path, requested_task=train_config.task)
    if detection.get("ambiguous"):
        raise ValueError(
            "Dataset task is ambiguous. Ask whether labels represent classes or objects and pass "
            "task='multiclass_semantic' or task='instance_friendly'."
        )
    detected_task = str(detection["task"])
    if detected_task == "unsupported":
        raise ValueError(str(detection.get("reason", "Unsupported annotation type")))

    info = inspect_dataset(train_pairs + val_pairs)
    input_channels = info.input_channels if train_config.input_channels == AUTO else int(train_config.input_channels)
    label_values = [1] if detected_task == "binary_semantic" else info.label_values
    output_channels = target_output_channels(detected_task, label_values)
    if train_config.output_classes != AUTO and detected_task == "multiclass_semantic":
        output_channels = int(train_config.output_classes)

    patch_size = (
        default_patch_size(train_config.architecture, info.image_shape)
        if train_config.patch_size == AUTO
        else train_config.patch_size
    )
    assert isinstance(patch_size, tuple)
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

    arch = architecture_defaults(
        train_config.architecture,
        input_channels=input_channels,
        output_channels=output_channels,
    )
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
        seed=train_config.seed,
    )
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
        seed=train_config.seed + 10_000,
    )
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
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)
    except AttributeError:  # pragma: no cover - older torch fallback
        scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)
    model_config = model_folder_config(
        train_config,
        detected_task,
        arch,
        input_axes="cyx" if input_channels > 1 else "yx",
        output_axes="yx",
        label_values=label_values,
    )
    model_config["training"].update(
        {
            "resolved_device": device.type,
            "patch_size": list(patch_size),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "foreground_oversampling": foreground_oversampling,
            "foreground_probability": foreground_probability,
            "augmentation_profile": augmentation_profile,
            "mixed_precision": mixed_precision,
        }
    )
    write_json(output_dir / "config.json", model_config)

    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    total_steps = len(train_loader) * train_config.epochs
    global_step = 0
    for epoch in range(1, train_config.epochs + 1):
        model.train()
        train_losses: list[dict[str, float]] = []
        for images, target_batch in train_loader:
            global_step += 1
            images = images.to(device, non_blocking=True)
            target_batch = _move_target(target_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=mixed_precision):
                logits = model(images)
                loss, components = compute_loss(detected_task, logits, target_batch, train_config.loss_weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            component_floats = _tensor_losses_to_float(components)
            component_floats["total_loss"] = float(loss.detach().cpu().item())
            train_losses.append(component_floats)
            _emit(
                task,
                {
                    "type": "progress",
                    "epoch": epoch,
                    "step": global_step,
                    "total_epochs": train_config.epochs,
                    "total_steps": total_steps,
                    "losses": {f"train/{key}": value for key, value in component_floats.items()},
                    "metrics": {},
                },
            )

        model.eval()
        val_losses: list[dict[str, float]] = []
        val_metrics: list[dict[str, float]] = []
        with torch.inference_mode():
            for images, target_batch in val_loader:
                images = images.to(device, non_blocking=True)
                target_batch = _move_target(target_batch, device)
                logits = model(images)
                loss, components = compute_loss(detected_task, logits, target_batch, train_config.loss_weights)
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
        history.append(epoch_record)
        score = primary_metric(detected_task, epoch_record["val_metrics"])
        logger.info("epoch=%s train=%s val=%s metrics=%s", epoch, epoch_record["train_losses"], epoch_record["val_losses"], epoch_record["val_metrics"])
        _emit(
            task,
            {
                "type": "progress",
                "epoch": epoch,
                "step": global_step,
                "total_epochs": train_config.epochs,
                "total_steps": total_steps,
                "losses": {f"val/{key}": value for key, value in epoch_record["val_losses"].items()},
                "metrics": {f"val/{key}": value for key, value in epoch_record["val_metrics"].items()},
            },
        )

        _save_checkpoint(
            output_dir / "weights_last.pt",
            model,
            optimizer,
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
                epoch,
                detected_task,
                arch,
                epoch_record,
                model_config,
            )
        write_json(output_dir / "metrics.json", {"history": history, "best_score": best_score})
        _save_previews(output_dir, epoch, detected_task, model, val_loader, device, train_config.preview_count)

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
        "config": model_config,
    }
    _emit(task, {"type": "complete", **result})
    logger.info("Training complete: %s", result)
    return result
