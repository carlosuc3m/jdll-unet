"""Sequential enlarged previews reusing regular-validation anchor predictions."""

from __future__ import annotations

import itertools
import json
import math
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import expit

from .annotations import _storage_dtype
from .config import PostprocessingConfig, write_json
from .device_ops import autocast_context, interpolate
from .errors import TrainingError
from .geometry import available_host_memory as available_ram
from .losses import primary_logits
from .postprocess import _remove_small, postprocess_binary, postprocess_instance
from .validation_sampling import PlannedValidationDataset


def importance_map(patch: tuple[int, ...], mode: str) -> np.ndarray:
    importance = np.ones(patch, dtype=np.float32)
    if mode == "gaussian":
        for axis, length in enumerate(patch):
            coordinate = (np.arange(length, dtype=np.float32) - (length - 1) / 2) / (length / 8)
            shape = [1] * len(patch)
            shape[axis] = length
            importance *= np.exp(-0.5 * coordinate**2).reshape(shape)
        importance /= importance.max()
        np.maximum(importance, 1e-6, out=importance)
    return importance


def predict_tile(model: torch.nn.Module, image: np.ndarray, device: torch.device,
                 dtype: torch.dtype = torch.float32) -> np.ndarray:
    with torch.inference_mode(), autocast_context(device, dtype):
        logits = primary_logits(model(torch.from_numpy(image[None]).to(device)))[0]
        if tuple(logits.shape[1:]) != tuple(image.shape[1:]):
            logits = interpolate(logits[None], size=image.shape[1:],
                      mode="trilinear" if image.ndim == 4 else "bilinear", align_corners=False)[0]
        result = logits.detach().float().cpu().numpy()
    if not np.isfinite(result).all():
        raise TrainingError("Non-finite validation tile prediction")
    return result


@dataclass
class PreviewAnchor:
    sample: int
    image: np.ndarray
    target: np.ndarray
    valid: np.ndarray
    logits: np.ndarray

    @property
    def nbytes(self) -> int:
        return sum(value.nbytes for value in (self.image, self.target, self.valid, self.logits))


def preview_layout(data: PlannedValidationDataset, sample: int) -> tuple[list[tuple[int, ...]], tuple[int, ...], tuple[int, ...]]:
    domain_index, cell = data.samples[sample]
    domain = data.domains[domain_index]
    anchor = domain.origin(cell, data.patch)
    positions = []
    for axis, (start, n, p) in enumerate(zip(anchor, domain.shape, data.patch, strict=True)):
        values = [start]
        if axis >= len(data.patch) - 2 and n > p:
            stride = max(1, int(p * (1 - data.options.tile_overlap)))
            neighbor = min(start + stride, n - p)
            if neighbor == start:
                neighbor = max(0, start - stride)
            if neighbor != start:
                values.append(neighbor)
        positions.append(tuple(sorted(values)))
    tiles = list(itertools.product(*positions))
    origin = tuple(min(v) for v in positions)
    shape = tuple(max(v) - min(v) + p for v, p in zip(positions, data.patch, strict=True))
    return tiles, origin, shape


def preview_workspace(anchor: PreviewAnchor, shape: tuple[int, ...], task: str) -> int:
    voxels = math.prod(shape)
    # Include native reconstruction work arrays (EDT/h-maxima/labels), incoming
    # tile buffers, and PNG slices. This is a conservative estimate, not an RSS cap.
    reconstruction = 96 if task == "instance_friendly" else 24
    return voxels * (4 * anchor.image.shape[0] + 4 * anchor.logits.shape[0] + anchor.target.dtype.itemsize
                     + 5 + reconstruction) + 2 * anchor.nbytes


def source_crop_workspace(data: PlannedValidationDataset, domain_index: int) -> int:
    domain = data.domains[domain_index]
    pair = data.base.pairs[domain.pair]
    native = pair.domain_shape[-len(data.patch):] or domain.base_shape
    lengths = [min(n, math.ceil(p * n / m) + 2) for n, p, m in zip(native, data.patch, domain.shape, strict=True)]
    channels = max(1, pair.image_channels) * (data.base.context_slices if data.dimensions == "2.5d" else 1)
    return math.prod(lengths) * (8 * channels + 16)


def reconstruct(task: str, logits: np.ndarray, valid: np.ndarray, config: PostprocessingConfig,
                spacing: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray | None]:
    if task == "multiclass_semantic":
        prediction = logits.argmax(0).astype(_storage_dtype(logits.shape[0] - 1))
        prediction[~valid] = 0
        if config.min_object_size > 0:
            for cls in range(1, logits.shape[0]):
                current = prediction == cls
                prediction[current & ~_remove_small(current, config.min_object_size)] = 0
        return prediction, None
    # Reuse the stitching buffer for probabilities, rather than keeping both.
    expit(logits, out=logits)
    logits[:, ~valid] = 0
    logits[0, ~valid] = -1
    if task == "instance_friendly":
        from dataclasses import asdict
        prediction = postprocess_instance(logits[0], logits[1], logits[2], spacing=spacing, **asdict(config))["labels"]
    else:
        prediction = postprocess_binary(logits[0], threshold=config.threshold, min_object_size=config.min_object_size,
                                        fill_holes=config.fill_holes, connected_components=False)["mask"]
    prediction[~valid] = 0
    logits[:, ~valid] = 0
    return prediction, logits


def _save_array(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_pngs(folder: Path, image: np.ndarray, target: np.ndarray, prediction: np.ndarray,
               valid: np.ndarray, data: PlannedValidationDataset) -> dict[str, Any]:
    from .targets import multiclass_target
    from .trainer import _atomic_image_write, _image_preview_rgb, _label_to_rgb, _overlay_prediction
    z = image.shape[1] // 2 if data.dimensions == "3d" else None
    if data.dimensions == "2.5d":
        context = data.base.context_slices
        image = image[context // 2::context]
    image_rgb = _image_preview_rgb(image, z)
    target_plane = target[z] if z is not None else target
    if data.base.task == "multiclass_semantic":
        target_plane = multiclass_target(target_plane, data.base.label_values)
    elif data.base.task == "binary_semantic":
        target_plane = (target_plane != 0).astype(np.uint8)
    target_rgb = _label_to_rgb(target_plane)
    prediction_rgb = _label_to_rgb(prediction[z] if z is not None else prediction)
    support = valid[z] if z is not None else valid
    for rgb in (image_rgb, target_rgb, prediction_rgb):
        rgb[~support] = 0
    arrays = {"image": image_rgb, "target": target_rgb, "prediction": prediction_rgb,
              "overlay": _overlay_prediction(image_rgb, prediction_rgb), "validity": support.astype(np.uint8) * 255}
    result: dict[str, Any] = {"z_index": z}
    for key, array in arrays.items():
        path = folder / (key + ".png")
        _atomic_image_write(path, array)
        result[key + "_path"] = str(path.resolve())
    return result


def save_enlarged_previews(data: PlannedValidationDataset, anchors: list[PreviewAnchor], model: torch.nn.Module,
                          device: torch.device, output: Path, epoch: int, postprocessing: PostprocessingConfig,
                          *, emit: Callable[..., Any], check_cancel: Callable[[], None],
                          dtype: torch.dtype = torch.float32) -> dict[str, str] | None:
    if not anchors:
        return None
    root = output / "previews"
    folder = root / f"volume_epoch_{epoch:04d}"
    if folder.exists():
        raise TrainingError(f"Refusing to overwrite published volumetric preview assets: {folder}")
    folder.mkdir(parents=True)
    items: list[dict[str, Any]] = []
    total_extra = 0
    retained = sum(anchor.nbytes for anchor in anchors)
    budget = int(data.options.preview_max_bytes)
    importance = importance_map(data.patch, data.options.tile_blending)
    manifest_written = False
    try:
        while anchors:
            check_cancel()
            anchor = anchors.pop(0)
            domain_index, cell = data.samples[anchor.sample]
            anchor_origin = data.domains[domain_index].origin(cell, data.patch)
            tiles, origin, shape = preview_layout(data, anchor.sample)
            available = available_ram()
            native_workspace = source_crop_workspace(data, domain_index)
            while True:
                working = retained + preview_workspace(anchor, shape, data.base.task)
                fits = working <= budget and (available is None or working + native_workspace <= available // 2)
                if fits or len(tiles) == 1:
                    break
                # Reduce one XY direction, preserving the anchor and model scale.
                axis = next(axis for axis in reversed(range(len(data.patch))) if len({t[axis] for t in tiles}) > 1)
                tiles = [t for t in tiles if t[axis] == anchor_origin[axis]]
                origin = tuple(min(t[a] for t in tiles) for a in range(len(data.patch)))
                shape = tuple(max(t[a] for t in tiles) - origin[a] + p for a, p in enumerate(data.patch))
            if not fits:
                emit("warning", message="Skipping a validation preview because its estimated workspace exceeds the memory budget.",
                     epoch=epoch, sample=anchor.sample, estimated_bytes=working, preview_max_bytes=budget,
                     source_crop_estimated_bytes=native_workspace)
                retained -= anchor.nbytes
                del anchor
                continue
            channels = anchor.logits.shape[0]
            stitched = np.zeros((channels, *shape), dtype=np.float32)
            counts = np.zeros(shape, dtype=np.float32)
            image = np.zeros((anchor.image.shape[0], *shape), dtype=np.float32)
            target = np.zeros(shape, dtype=anchor.target.dtype)
            valid = np.zeros(shape, dtype=bool)
            extra = 0
            for tile in tiles:
                check_cancel()
                cached = anchor if tile == anchor_origin else next(
                    (other for other in anchors if data.samples[other.sample][0] == domain_index
                     and data.domains[domain_index].origin(data.samples[other.sample][1], data.patch) == tile), None)
                if cached is not None:
                    patch_image, patch_target, patch_valid, prediction = cached.image, cached.target, cached.valid, cached.logits
                else:
                    incoming_image, patch_target, patch_valid = data.read_tile(domain_index, tile)
                    assert incoming_image is not None
                    patch_image = incoming_image
                    del incoming_image
                    prediction = predict_tile(model, patch_image, device, dtype)
                    extra += 1
                selection = tuple(slice(start - base, start - base + size)
                                  for start, base, size in zip(tile, origin, data.patch, strict=True))
                stitched[(slice(None), *selection)] += prediction * importance
                counts[selection] += importance
                image[(slice(None), *selection)] = patch_image
                target[selection] = patch_target
                valid[selection] = patch_valid
                del patch_image, patch_target, patch_valid, prediction, cached
            stitched /= counts[None]
            del counts
            metadata = data.provenance(domain_index, origin, shape)
            prediction, probability = reconstruct(data.base.task, stitched, valid, postprocessing,
                                                  tuple(metadata["spacing"]))
            destination = folder / f"preview_{len(items):03d}"
            destination.mkdir()
            assets = {}
            arrays = {"image": image, "target": target, "prediction": prediction, "validity": valid}
            if probability is not None:
                arrays["probabilities"] = probability
            for name, array in arrays.items():
                check_cancel()
                path = destination / f"{name}.npy"
                _save_array(path, array)
                assets[name] = {"path": str(path.resolve()), "dtype": str(array.dtype), "shape": list(array.shape),
                                "axes": ("C" if name in {"image", "probabilities"} else "") + metadata["axes"]}
            item = {"index": len(items), "sample_index": anchor.sample, **metadata,
                    "assets": assets, "scope": "enlarged_validation_patch", "additional_tiles": extra,
                    "tile_origins": tiles, "tile_layout": [len({t[a] for t in tiles}) for a in range(len(shape))],
                    "tile_overlap": data.options.tile_overlap, "tile_blending": data.options.tile_blending,
                    "target_label_encoding": "prepared_source_ids" if data.base.task == "instance_friendly" else "source_ids",
                    "prediction_label_encoding": "class_index" if data.base.task == "multiclass_semantic" else "instance_id" if data.base.task == "instance_friendly" else "binary",
                    "class_index_to_source_label": [0, *(data.base.label_values or []),
                                                    *([None] * max(0, channels - len(data.base.label_values or []) - 1))]
                    if data.base.task == "multiclass_semantic" else None,
                    "prediction_channels": ["foreground", "boundary", "distance"] if data.base.task == "instance_friendly"
                    else ["foreground"] if data.base.task == "binary_semantic" else [f"class_{i}" for i in range(channels)],
                    "estimated_workspace_bytes": working,
                    **_save_pngs(destination, image, target, prediction, valid, data)}
            write_json(destination / "geometry.json", item)
            items.append(item)
            total_extra += extra
            emit("validation_previews", status="saved", epoch=epoch, completed=len(items),
                 additional_tiles=total_extra, message=f"Saved enlarged validation preview {len(items)} at {destination}.")
            retained -= anchor.nbytes
            del anchor, image, target, valid, prediction, probability, stitched, arrays
        if not items:
            shutil.rmtree(folder)
            return None
        payload = {"format_version": 2, "epoch": epoch, "task": data.base.task,
                   "dimensions": data.dimensions, "scope": "enlarged_validation_patches", "items": items,
                   "actual_preview_count": len(items), "additional_tiles": total_extra,
                   "preview_max_bytes": budget, "owned_asset_directory": str(folder.resolve())}
        manifest = root / f"epoch_{epoch:04d}.json"
        latest = root / "latest.json"
        write_json(manifest, payload)
        manifest_written = True
        write_json(latest, payload)
        _retain_previews(root, keep=2)
        return {"preview_path": str(manifest.resolve()), "latest_preview_path": str(latest.resolve())}
    except BaseException:
        # A failed publication must leave the previous latest manifest usable.
        latest = root / "latest.json"
        published = latest.exists() and json.loads(latest.read_text()).get("epoch") == epoch
        if not published:
            if manifest_written:
                (root / f"epoch_{epoch:04d}.json").unlink(missing_ok=True)
            shutil.rmtree(folder, ignore_errors=True)
        raise


def _retain_previews(root: Path, keep: int) -> None:
    owned = []
    for path in sorted(root.glob("epoch_*.json")):
        payload = json.loads(path.read_text())
        folder = Path(payload.get("owned_asset_directory", ""))
        if payload.get("format_version") == 2 and folder.parent.resolve() == root.resolve() and folder.name.startswith("volume_epoch_"):
            owned.append((path, folder))
    for manifest, folder in owned[:-keep]:
        shutil.rmtree(folder, ignore_errors=True)
        manifest.unlink(missing_ok=True)
