"""Model loading and tiled inference for JDLL UNet."""

from __future__ import annotations

import os
import warnings
from collections import OrderedDict
from itertools import product
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F

from .callbacks import CallbackDispatcher
from .config import ArchitectureConfig, read_json, resolve_device
from .errors import InferenceError, ModelLoadError
from .io import load_image, normalize_image
from .losses import primary_logits
from .model import build_unet
from .planning import read_spacing, resample_image_mask, resolve_context_stride, restore_continuous_maps
from .postprocess import postprocess_binary, postprocess_instance, postprocess_multiclass
from .scale import resize_2d_channels
from .semantic_scale import compare_semantic_region_fraction


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
        config = read_json(folder_config)
        metadata_path = checkpoint.parent / "model_metadata.json"
        if metadata_path.exists():
            config["_model_metadata"] = read_json(metadata_path)
        return config
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


def _parse_tile_size(value: Any, dimensions: str) -> tuple[int, ...]:
    expected = 3 if dimensions == "3d" else 2
    if isinstance(value, int):
        tile_size = tuple([value] * expected)
    elif isinstance(value, (list, tuple)) and len(value) == expected:
        tile_size = tuple(int(item) for item in value)
    else:
        raise InferenceError(f"tile_size must be an int or a length-{expected} sequence")
    if any(item <= 0 for item in tile_size):
        raise InferenceError("tile_size values must be positive")
    return tile_size


def _parse_overlap(value: Any) -> float:
    overlap = float(value)
    if not 0 <= overlap < 1:
        raise InferenceError("tile_overlap must be in [0, 1)")
    return overlap


def _pad_image(image: np.ndarray, patch_size: tuple[int, ...]) -> tuple[np.ndarray, tuple[int, ...]]:
    spatial_shape = tuple(int(item) for item in image.shape[1:])
    pads = [max(0, patch - size) for patch, size in zip(patch_size, spatial_shape, strict=True)]
    if all(pad == 0 for pad in pads):
        return image, spatial_shape
    padded = np.pad(image, ((0, 0), *((0, pad) for pad in pads)), mode="edge")
    return padded, spatial_shape


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
    tile_size: tuple[int, ...],
    overlap: float = 0.25,
) -> np.ndarray:
    image, original_shape = _pad_image(image, tile_size)
    spatial_shape = tuple(int(item) for item in image.shape[1:])
    strides = tuple(max(1, int(tile * (1.0 - overlap))) for tile in tile_size)
    starts_by_axis = [
        _starts(length, tile, stride) for length, tile, stride in zip(spatial_shape, tile_size, strides, strict=True)
    ]
    output_channels = int(model.config.output_channels)
    accum = torch.zeros((output_channels, *spatial_shape), dtype=torch.float32, device=device)
    counts = torch.zeros((1, *spatial_shape), dtype=torch.float32, device=device)
    with torch.inference_mode():
        for starts in product(*starts_by_axis):
            spatial_slices = tuple(slice(start, start + tile) for start, tile in zip(starts, tile_size, strict=True))
            patch = image[(slice(None), *spatial_slices)]
            patch_t = torch.from_numpy(patch[None].astype(np.float32, copy=False)).to(device)
            patch_logits = primary_logits(model(patch_t))[0]
            if patch_logits.shape[1:] != tuple(tile_size):
                mode = "trilinear" if len(tile_size) == 3 else "bilinear"
                patch_logits = F.interpolate(patch_logits[None], size=tile_size, mode=mode, align_corners=False)[0]
            accum[(slice(None), *spatial_slices)] += patch_logits
            counts[(slice(None), *spatial_slices)] += 1.0
    logits_array = (accum / counts.clamp_min(1.0)).detach().cpu().numpy()
    crop = tuple(slice(0, size) for size in original_shape)
    return logits_array[(slice(None), *crop)]


def _load_input(inputs: dict[str, Any] | np.ndarray, dimensions: str) -> np.ndarray:
    if isinstance(inputs, np.ndarray):
        image = inputs
        if dimensions in {"3d", "2.5d"}:
            if image.ndim == 3:
                if image.shape[-1] in {1, 3, 4} and image.shape[0] not in {1, 3, 4}:
                    raise InferenceError(f"3D model input looks like a 2D RGB image, got shape {image.shape}")
                out = image[None].astype(np.float32, copy=False)
            elif image.ndim == 4:
                out = image.astype(np.float32, copy=False)
            else:
                raise InferenceError(f"Unsupported 3D input image shape: {image.shape}")
        elif image.ndim == 2:
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
        return _load_input(np.asarray(inputs["image"]), dimensions)
    if "image_path" in inputs:
        return load_image(inputs["image_path"], dimensions=dimensions)
    raise InferenceError("Inference inputs must contain 'image' or 'image_path'")


def _validate_input_channels(image: np.ndarray, model_config: dict[str, Any]) -> None:
    expected = int(model_config.get("input_channels", image.shape[0]))
    dimensions = (model_config.get("architecture_config") or {}).get("dimensions", "2d")
    context_slices = int((model_config.get("architecture_config") or {}).get("context_slices", 3))
    expected_source = expected // context_slices if dimensions == "2.5d" else expected
    if image.shape[0] != expected_source:
        raise InferenceError(f"Model expects {expected} input channel(s), got {image.shape[0]}")
    expected_rank = 4 if dimensions in {"3d", "2.5d"} else 3
    if image.ndim != expected_rank:
        raise InferenceError(f"Model expects {dimensions} input rank {expected_rank}, got shape {image.shape}")


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))


def _context_stack(volume: np.ndarray, center_z: int, context_slices: int, stride: int = 1) -> np.ndarray:
    radius = context_slices // 2
    channels: list[np.ndarray] = []
    for modality in range(volume.shape[0]):
        for z in range(center_z - radius * stride, center_z + radius * stride + 1, stride):
            channels.append(
                volume[modality, z] if 0 <= z < volume.shape[1] else np.zeros(volume.shape[2:], dtype=volume.dtype)
            )
    return np.ascontiguousarray(np.stack(channels))


def _predict_25d(
    model: torch.nn.Module,
    volume: np.ndarray,
    device: torch.device,
    tile_size: tuple[int, int],
    overlap: float,
    context_slices: int,
    context_stride: int = 1,
) -> np.ndarray:
    predictions = [
        tiled_predict(model, _context_stack(volume, z, context_slices, context_stride), device, tile_size, overlap)
        for z in range(volume.shape[1])
    ]
    return np.ascontiguousarray(np.stack(predictions, axis=1))


def infer(config: dict[str, Any], inputs: dict[str, Any] | np.ndarray, task: Any = None) -> dict[str, Any]:
    model_path = config.get("model_path") or config.get("model_folder") or config.get("checkpoint")
    if model_path is None:
        raise InferenceError("Inference config requires model_path, model_folder, or checkpoint")
    device = resolve_device(str(config.get("device", "cpu")))
    model, model_config = load_model(model_path, device)
    dimensions = str((model_config.get("architecture_config") or {}).get("dimensions", "2d"))
    image = normalize_image(_load_input(inputs, dimensions), model_config.get("normalization"))
    _validate_input_channels(image, model_config)
    original_spatial_shape = tuple(int(item) for item in image.shape[1:])
    task_name = str(model_config["task"])
    train_cfg = model_config.get("training", {})
    metadata = model_config.get("_model_metadata", {})
    dataset_plan = metadata.get("dataset_plan", {})
    spacing_cfg = train_cfg.get("spacing", {})
    input_spacing = None
    if isinstance(inputs, dict):
        supplied_spacing = inputs.get("spacing", config.get("spacing"))
        if supplied_spacing is not None:
            input_spacing = cast(tuple[float, float, float], tuple(float(value) for value in supplied_spacing))
        elif inputs.get("image_path") is not None and dimensions in {"2.5d", "3d"}:
            input_spacing, _spacing_source = read_spacing(Path(inputs["image_path"]))
    if input_spacing is None and dimensions in {"2.5d", "3d"}:
        input_spacing = cast(
            tuple[float, float, float], tuple(float(value) for value in spacing_cfg.get("default_spacing", (1, 1, 1)))
        )
    target_spacing = (
        cast(tuple[float, float, float], tuple(dataset_plan["target_spacing"]))
        if dataset_plan.get("target_spacing")
        else None
    )
    if dimensions == "3d" and input_spacing is not None and target_spacing is not None:
        dummy_mask = np.zeros(image.shape[1:], dtype=np.uint8)
        image, _ = resample_image_mask(image, dummy_mask, input_spacing, target_spacing)
    tile_size = (
        config.get("tile_size") or train_cfg.get("patch_size") or ([16, 96, 96] if dimensions == "3d" else [128, 128])
    )
    tile_size = _parse_tile_size(tile_size, dimensions)
    semantic_scale_comparison = None
    semantic_scale_factor = 1.0
    if task_name in {"binary_semantic", "multiclass_semantic"}:
        supplied_fraction = config.get("semantic_region_fraction")
        semantic_values = {
            "semantic_region_area": config.get("semantic_region_area"),
            "semantic_region_volume": config.get("semantic_region_volume"),
            "semantic_region_size": config.get("semantic_region_size"),
            "object_size": config.get("object_size"),
        }
        if isinstance(inputs, dict):
            supplied_fraction = inputs.get("semantic_region_fraction", supplied_fraction)
            semantic_values = {
                key: inputs.get(key, value) for key, value in semantic_values.items()
            }
        provided_measures = {key: value for key, value in semantic_values.items() if value is not None}
        if len(provided_measures) > 1 or (supplied_fraction is not None and provided_measures):
            raise InferenceError("Provide exactly one semantic region fraction, area, volume, size, or object_size")
        if dimensions == "3d" and semantic_values["semantic_region_area"] is not None:
            raise InferenceError("Use semantic_region_volume rather than semantic_region_area for 3D inference")
        if dimensions != "3d" and semantic_values["semantic_region_volume"] is not None:
            raise InferenceError("Use semantic_region_area rather than semantic_region_volume for 2D/2.5D inference")
        supplied_measure = next(iter(provided_measures.values()), None)
        if supplied_measure is not None:
            try:
                diagnostic_patch = (metadata.get("semantic_scale_diagnostics") or {}).get("patch_size", tile_size)
                supplied_fraction = float(supplied_measure) / float(np.prod(diagnostic_patch))
            except (TypeError, ValueError) as exc:
                raise InferenceError("semantic region size must be a positive number") from exc
        if supplied_fraction is not None:
            try:
                semantic_scale_comparison = compare_semantic_region_fraction(
                    float(supplied_fraction), metadata.get("semantic_scale_diagnostics") or {}
                )
            except (TypeError, ValueError) as exc:
                raise InferenceError(str(exc)) from exc
            if semantic_scale_comparison["status"] == "training_distribution_unavailable":
                raise InferenceError(
                    "This model does not contain semantic scale diagnostics; retrain it before using semantic size normalization"
                )
            scale_power = 1 / (3 if dimensions == "3d" else 2)
            requested_scale = (1 / float(semantic_scale_comparison["ratio_to_training_median"])) ** scale_power
            try:
                minimum_scale = float(config.get("semantic_scale_min_factor", 0.25))
                maximum_scale = float(config.get("semantic_scale_max_factor", 4.0))
            except (TypeError, ValueError) as exc:
                raise InferenceError("semantic scale factor limits must be positive numbers") from exc
            if minimum_scale <= 0 or maximum_scale < minimum_scale:
                raise InferenceError("semantic scale factor limits must be positive and ordered")
            semantic_scale_factor = float(np.clip(requested_scale, minimum_scale, maximum_scale))
            semantic_scale_comparison["requested_scale_factor"] = requested_scale
            semantic_scale_comparison["applied_scale_factor"] = semantic_scale_factor
            semantic_scale_comparison["scale_was_clamped"] = not np.isclose(requested_scale, semantic_scale_factor)
            if semantic_scale_comparison.get("warning"):
                message = (
                    "Approximate semantic region fraction is "
                    f"{semantic_scale_comparison['status'].replace('_', ' ')} "
                    f"(ratio to training median: {semantic_scale_comparison['ratio_to_training_median']:.3g}). "
                    f"Applied semantic scale factor {semantic_scale_factor:.3g}."
                )
                warnings.warn(message, RuntimeWarning, stacklevel=2)
                CallbackDispatcher(task).emit("warning", message=message, **semantic_scale_comparison)
            scale_source_shape = image.shape[-2:] if dimensions == "2.5d" else image.shape[1:]
            scaled_shape = tuple(max(1, int(round(size * semantic_scale_factor))) for size in scale_source_shape)
            if dimensions == "2.5d":
                flattened = image.reshape(image.shape[0] * image.shape[1], *image.shape[2:])
                image = resize_2d_channels(flattened, cast(tuple[int, int], scaled_shape)).reshape(
                    image.shape[0], image.shape[1], *scaled_shape
                )
            elif dimensions == "2d":
                image = np.ascontiguousarray(resize_2d_channels(image, cast(tuple[int, int], scaled_shape)))
            else:
                image_t = torch.from_numpy(np.ascontiguousarray(image[None]))
                image = F.interpolate(image_t, size=scaled_shape, mode="trilinear", align_corners=False)[0].numpy()
    overlap = _parse_overlap(config.get("tile_overlap", 0.25))
    inference_scale = 1.0
    scale_cfg = train_cfg.get("instance_scale_normalization", {})
    instance_scale_enabled = task_name == "instance_friendly" and bool(scale_cfg.get("enabled", False))
    if instance_scale_enabled:
        object_size = config.get("object_size", config.get("object_diameter"))
        if object_size is None and isinstance(inputs, dict):
            object_size = inputs.get("object_size", inputs.get("object_diameter"))
        if object_size is None:
            raise InferenceError("Scale-normalized instance inference requires approximate object_size in pixels")
        try:
            object_size = float(object_size)
        except (TypeError, ValueError) as exc:
            raise InferenceError("object_size must be a positive number in input pixels") from exc
        if object_size <= 0 or not np.isfinite(object_size):
            raise InferenceError("object_size must be a positive finite number in input pixels")
        training_patch = _parse_tile_size(train_cfg.get("patch_size", tile_size), dimensions)
        scale_metadata = metadata.get("instance_scale") or {}
        target_diameter = float(
            scale_metadata.get(
                "target_object_size", float(scale_cfg.get("target_object_fraction", 0.25)) * min(training_patch)
            )
        )
        inference_scale = float(
            np.clip(
                target_diameter / object_size,
                float(scale_cfg.get("min_effective_scale", 0.25)),
                float(scale_cfg.get("max_effective_scale", 4.0)),
            )
        )
        scale_source_shape = original_spatial_shape[-2:] if dimensions == "2.5d" else tuple(image.shape[1:])
        scaled_shape = tuple(max(1, int(round(size * inference_scale))) for size in scale_source_shape)
        if dimensions == "2.5d":
            flattened = image.reshape(image.shape[0] * image.shape[1], *image.shape[2:])
            image = resize_2d_channels(flattened, cast(tuple[int, int], scaled_shape)).reshape(
                image.shape[0], image.shape[1], *scaled_shape
            )
        elif dimensions == "2d":
            image = np.ascontiguousarray(resize_2d_channels(image, cast(tuple[int, int], scaled_shape)))
        else:
            image_t = torch.from_numpy(np.ascontiguousarray(image[None]))
            image = F.interpolate(image_t, size=scaled_shape, mode="trilinear", align_corners=False)[0].numpy()
    context_slices = int((model_config.get("architecture_config") or {}).get("context_slices", 3))
    context_cfg = train_cfg.get("context", {})
    resolved_context = metadata.get("resolved_context", {})
    context_policy = str(resolved_context.get("stride_policy", context_cfg.get("stride_policy", "adjacent")))
    target_context_spacing = resolved_context.get("target_spacing")
    context_stride = resolve_context_stride(
        context_policy,
        fixed_stride=int(context_cfg.get("stride", 1)),
        target_spacing=float(target_context_spacing) if target_context_spacing is not None else None,
        z_spacing=float(input_spacing[0]) if input_spacing is not None else 1.0,
    )
    logits = (
        _predict_25d(model, image, device, cast(tuple[int, int], tile_size), overlap, context_slices, context_stride)
        if dimensions == "2.5d"
        else tiled_predict(model, image, device, tile_size=tile_size, overlap=overlap)
    )
    if dimensions == "3d" and logits.shape[1:] != original_spatial_shape:
        logits = restore_continuous_maps(logits, original_spatial_shape)
    elif semantic_scale_factor != 1.0 and logits.shape[1:] != original_spatial_shape:
        if dimensions == "2.5d":
            channels, depth = logits.shape[:2]
            restored = resize_2d_channels(
                logits.reshape(channels * depth, *logits.shape[2:]),
                cast(tuple[int, int], original_spatial_shape[-2:]),
            )
            logits = restored.reshape(channels, depth, *original_spatial_shape[-2:])
        else:
            logits = resize_2d_channels(logits, cast(tuple[int, int], original_spatial_shape))

    post_cfg = dict(model_config.get("postprocessing", {}))
    post_overrides = config.get("postprocessing", {})
    if not isinstance(post_overrides, dict):
        raise InferenceError("postprocessing overrides must be a mapping")
    post_cfg.update(post_overrides)
    if task_name == "binary_semantic":
        probability = _sigmoid(logits[0])
        outputs = {"foreground_probability": probability}
        outputs.update(
            postprocess_binary(
                probability,
                threshold=float(post_cfg.get("threshold", 0.5)),
                min_object_size=int(post_cfg.get("min_object_size", 0)),
                fill_holes=bool(post_cfg.get("fill_holes", False)),
                connected_components=bool(post_cfg.get("connected_components", True)),
            )
        )
    elif task_name == "multiclass_semantic":
        exp = np.exp(logits - logits.max(axis=0, keepdims=True))
        probabilities = exp / exp.sum(axis=0, keepdims=True)
        outputs = {"probabilities": probabilities}
        outputs.update(postprocess_multiclass(probabilities, min_object_size=int(post_cfg.get("min_object_size", 0))))
    elif task_name == "instance_friendly":
        probabilities = _sigmoid(logits)
        if instance_scale_enabled and probabilities.shape[1:] != original_spatial_shape:
            if dimensions == "2.5d":
                channels, depth = probabilities.shape[:2]
                restored = resize_2d_channels(
                    probabilities.reshape(channels * depth, *probabilities.shape[2:]),
                    cast(tuple[int, int], original_spatial_shape[1:]),
                )
                probabilities = restored.reshape(channels, depth, *original_spatial_shape[1:])
            else:
                probabilities = resize_2d_channels(probabilities, cast(tuple[int, int], original_spatial_shape))
        outputs = {
            "foreground_probability": probabilities[0],
            "boundary_probability": probabilities[1],
            "distance_probability": probabilities[2],
        }
        outputs.update(
            postprocess_instance(
                probabilities[0],
                probabilities[1],
                probabilities[2],
                threshold=float(post_cfg.get("threshold", 0.5)),
                min_object_size=int(post_cfg.get("min_object_size", 0)),
                method=str(post_cfg.get("method", "distance_boundary_watershed")),
                seed_distance_threshold=float(post_cfg.get("seed_distance_threshold", 0.35)),
                seed_boundary_threshold=float(post_cfg.get("seed_boundary_threshold", 0.5)),
                seed_h=float(post_cfg.get("seed_h", 0.1)),
                min_seed_size=int(post_cfg.get("min_seed_size", 3)),
                boundary_weight=float(post_cfg.get("boundary_weight", 1.0)),
                connectivity=str(post_cfg.get("connectivity", "face")),
                spacing=tuple(input_spacing) if input_spacing is not None else None,
                min_object_size_physical=post_cfg.get("min_object_size_physical"),
                min_seed_size_physical=post_cfg.get("min_seed_size_physical"),
            )
        )
    else:
        raise ModelLoadError(f"Unsupported model task: {task_name}")

    metadata = {
        "task": task_name,
        "model_path": str(model_path),
        "input_shape": list(image.shape),
        "original_input_shape": [int(image.shape[0]), *original_spatial_shape],
        "instance_scale_factor": inference_scale,
        "input_spacing": list(input_spacing) if input_spacing is not None else None,
        "target_spacing": list(target_spacing) if target_spacing is not None else None,
        "context_stride": context_stride if dimensions == "2.5d" else None,
        "semantic_scale_comparison": semantic_scale_comparison,
        "semantic_scale_factor": semantic_scale_factor,
        "output_keys": sorted(outputs),
    }
    CallbackDispatcher(task).emit("complete", message="UNet inference complete", **metadata)
    return {"metadata": metadata, "outputs": outputs}
