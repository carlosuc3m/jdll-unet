"""Model loading and tiled inference for JDLL UNet."""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .config import ArchitectureConfig, read_json, resolve_device
from .errors import InferenceError, ModelLoadError
from .io import load_image, normalize_image
from .model import build_unet
from .postprocess import postprocess_binary, postprocess_instance, postprocess_multiclass


def _model_cache_size() -> int:
    try:
        return max(1, int(os.environ.get("JDLL_UNET_MODEL_CACHE_SIZE", "2")))
    except ValueError:
        return 2


_MAX_MODEL_CACHE_SIZE = _model_cache_size()
_MODEL_CACHE: OrderedDict[tuple[str, str], tuple[torch.nn.Module, dict[str, Any]]] = OrderedDict()


def clear_model_cache() -> None:
    """Clear loaded inference models from the current Appose process."""

    _MODEL_CACHE.clear()


def _checkpoint_from_path(model_path: Path) -> Path:
    if model_path.is_dir():
        return model_path / "model.pt"
    return model_path


def _config_from_checkpoint(checkpoint: Path, state: dict[str, Any]) -> dict[str, Any]:
    folder_config = checkpoint.parent / "config.json"
    if folder_config.exists():
        return read_json(folder_config)
    if "model_config" in state:
        return state["model_config"]
    raise ModelLoadError(f"Cannot find config.json or embedded model_config for {checkpoint}")


def load_model(model_path: str | Path, device: str | torch.device = "cpu") -> tuple[torch.nn.Module, dict[str, Any]]:
    requested_device = resolve_device(str(device)) if not isinstance(device, torch.device) else device
    checkpoint = _checkpoint_from_path(Path(model_path))
    cache_key = (str(checkpoint.resolve()), str(requested_device))
    if cache_key in _MODEL_CACHE:
        _MODEL_CACHE.move_to_end(cache_key)
        return _MODEL_CACHE[cache_key]
    if not checkpoint.exists():
        raise ModelLoadError(f"Model checkpoint does not exist: {checkpoint}")
    state = torch.load(checkpoint, map_location=requested_device, weights_only=False)
    if not isinstance(state, dict):
        raise ModelLoadError(f"Checkpoint {checkpoint} does not contain a state dictionary")
    config = _config_from_checkpoint(checkpoint, state)
    arch_payload = config.get("architecture_config") or state.get("architecture_config")
    if not arch_payload:
        raise ModelLoadError(f"Missing architecture_config for {checkpoint}")
    arch = ArchitectureConfig(**arch_payload)
    model = build_unet(arch).to(requested_device)
    model.load_state_dict(state.get("state_dict", state))
    model.eval()
    _MODEL_CACHE[cache_key] = (model, config)
    while len(_MODEL_CACHE) > _MAX_MODEL_CACHE_SIZE:
        _MODEL_CACHE.popitem(last=False)
    return model, config


def _parse_tile_size(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        tile_size = (value, value)
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        tile_size = (int(value[0]), int(value[1]))
    else:
        raise InferenceError("tile_size must be an int or a length-2 sequence")
    if tile_size[0] <= 0 or tile_size[1] <= 0:
        raise InferenceError("tile_size values must be positive")
    return tile_size


def _parse_overlap(value: Any) -> float:
    overlap = float(value)
    if not 0 <= overlap < 1:
        raise InferenceError("tile_overlap must be in [0, 1)")
    return overlap


def _pad_image(image: np.ndarray, patch_size: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image.shape[-2:]
    pad_y = max(0, patch_size[0] - height)
    pad_x = max(0, patch_size[1] - width)
    if pad_y == 0 and pad_x == 0:
        return image, (height, width)
    padded = np.pad(image, ((0, 0), (0, pad_y), (0, pad_x)), mode="edge")
    return padded, (height, width)


def _starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, max(1, length - tile + 1), stride))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def tiled_predict(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    tile_size: tuple[int, int],
    overlap: float = 0.25,
) -> np.ndarray:
    image, original_shape = _pad_image(image, tile_size)
    _channels, height, width = image.shape
    stride_y = max(1, int(tile_size[0] * (1.0 - overlap)))
    stride_x = max(1, int(tile_size[1] * (1.0 - overlap)))
    ys = _starts(height, tile_size[0], stride_y)
    xs = _starts(width, tile_size[1], stride_x)
    output_channels = int(model.config.output_channels)
    accum = torch.zeros((output_channels, height, width), dtype=torch.float32, device=device)
    counts = torch.zeros((1, height, width), dtype=torch.float32, device=device)
    with torch.inference_mode():
        for y in ys:
            for x in xs:
                patch = image[:, y : y + tile_size[0], x : x + tile_size[1]]
                patch_t = torch.from_numpy(patch[None].astype(np.float32, copy=False)).to(device)
                logits = model(patch_t)[0]
                if logits.shape[-2:] != tuple(tile_size):
                    logits = F.interpolate(logits[None], size=tile_size, mode="bilinear", align_corners=False)[0]
                accum[:, y : y + tile_size[0], x : x + tile_size[1]] += logits
                counts[:, y : y + tile_size[0], x : x + tile_size[1]] += 1.0
    logits = (accum / counts.clamp_min(1.0)).detach().cpu().numpy()
    return logits[:, : original_shape[0], : original_shape[1]]


def _load_input(inputs: dict[str, Any] | np.ndarray) -> np.ndarray:
    if isinstance(inputs, np.ndarray):
        image = inputs
        if image.ndim == 2:
            out = image[None].astype(np.float32, copy=False)
        elif image.ndim == 3 and image.shape[-1] in {1, 3, 4}:
            out = np.moveaxis(image[..., :3], -1, 0).astype(np.float32, copy=False)
        elif image.ndim == 3:
            out = image.astype(np.float32, copy=False)
        else:
            raise InferenceError(f"Unsupported input image shape: {image.shape}")
        if not np.all(np.isfinite(out)):
            raise InferenceError("Inference input contains non-finite values")
        return np.ascontiguousarray(out)
    if "image" in inputs:
        return _load_input(np.asarray(inputs["image"]))
    if "image_path" in inputs:
        return load_image(inputs["image_path"])
    raise InferenceError("Inference inputs must contain 'image' or 'image_path'")


def _validate_input_channels(image: np.ndarray, model_config: dict[str, Any]) -> None:
    expected = int(model_config.get("input_channels", image.shape[0]))
    if image.shape[0] != expected:
        raise InferenceError(f"Model expects {expected} input channel(s), got {image.shape[0]}")


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))


def infer(config: dict[str, Any], inputs: dict[str, Any] | np.ndarray, task: Any = None) -> dict[str, Any]:
    model_path = config.get("model_path") or config.get("model_folder") or config.get("checkpoint")
    if model_path is None:
        raise InferenceError("Inference config requires model_path, model_folder, or checkpoint")
    device = resolve_device(str(config.get("device", "cpu")))
    model, model_config = load_model(model_path, device)
    image = normalize_image(_load_input(inputs), model_config.get("normalization"))
    _validate_input_channels(image, model_config)
    task_name = str(model_config["task"])
    train_cfg = model_config.get("training", {})
    tile_size = config.get("tile_size") or train_cfg.get("patch_size") or [128, 128]
    tile_size = _parse_tile_size(tile_size)
    overlap = _parse_overlap(config.get("tile_overlap", 0.25))
    logits = tiled_predict(model, image, device, tile_size=tile_size, overlap=overlap)

    post_cfg = dict(model_config.get("postprocessing", {}))
    post_overrides = config.get("postprocessing", {})
    if not isinstance(post_overrides, dict):
        raise InferenceError("postprocessing overrides must be a mapping")
    post_cfg.update(post_overrides)
    if task_name == "binary_semantic":
        probability = _sigmoid(logits[0])
        outputs = {"foreground_probability": probability}
        outputs.update(postprocess_binary(probability, **post_cfg))
    elif task_name == "multiclass_semantic":
        exp = np.exp(logits - logits.max(axis=0, keepdims=True))
        probabilities = exp / exp.sum(axis=0, keepdims=True)
        outputs = {"probabilities": probabilities}
        outputs.update(postprocess_multiclass(probabilities, min_object_size=int(post_cfg.get("min_object_size", 0))))
    elif task_name == "instance_friendly":
        probabilities = _sigmoid(logits)
        outputs = {
            "foreground_probability": probabilities[0],
            "boundary_probability": probabilities[1],
        }
        outputs.update(
            postprocess_instance(
                probabilities[0],
                probabilities[1],
                threshold=float(post_cfg.get("threshold", 0.5)),
                min_object_size=int(post_cfg.get("min_object_size", 0)),
            )
        )
    else:
        raise ModelLoadError(f"Unsupported model task: {task_name}")

    metadata = {
        "task": task_name,
        "model_path": str(model_path),
        "input_shape": list(image.shape),
        "output_keys": sorted(outputs),
    }
    if task is not None:
        update = getattr(task, "update", None)
        if callable(update):
            update({"type": "complete", **metadata})
        elif callable(task):
            task({"type": "complete", **metadata})
    return {"metadata": metadata, "outputs": outputs}
