"""Evaluate a saved 3D training run without rebuilding its dataset analysis.

All native voxels contribute. Probability histograms bound memory; the ROC-AUC
quantization error is bounded by the positive/negative pairs sharing a bin.
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F

from jdll_unet.callbacks import CallbackDispatcher
from jdll_unet.infer import _InferenceProgress, _sigmoid, _tile_layout, load_model, tiled_predict
from jdll_unet.io import load_image, normalize_image
from jdll_unet.planning import resample_image_mask, restore_continuous_maps


class ProbabilityHistogram:
    def __init__(self, bins: int = 2**20) -> None:
        if bins < 2 or bins % 2:
            raise ValueError("bins must be an even integer of at least two")
        self.positive = np.zeros(bins, dtype=np.uint64)
        self.negative = np.zeros(bins, dtype=np.uint64)

    def update(self, probabilities: np.ndarray, foreground: np.ndarray) -> None:
        if probabilities.shape != foreground.shape:
            raise ValueError("Prediction and target shapes differ")
        if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("Probabilities must be finite and in [0, 1]")
        bins = len(self.positive)
        indices = np.minimum((probabilities.ravel() * bins).astype(np.int64), bins - 1)
        truth = foreground.ravel().astype(bool, copy=False)
        self.positive += np.bincount(indices[truth], minlength=bins).astype(np.uint64)
        self.negative += np.bincount(indices[~truth], minlength=bins).astype(np.uint64)

    def metrics(self) -> dict:
        pos, neg = self.positive.astype(np.float64), self.negative.astype(np.float64)
        p, n = float(pos.sum()), float(neg.sum())
        result = {
            "positive_voxels": int(p),
            "negative_voxels": int(n),
            "foreground_fraction": p / (p + n) if p + n else None,
            "histogram_bins": len(pos),
        }
        if not p or not n:
            return {
                **result,
                "roc_auc": None,
                "roc_auc_error_bound": None,
                "average_precision": None,
                "pr_auc_trapezoidal": None,
                "dice_at_0_5": None,
                "undefined_reason": "Both foreground and background are required",
            }
        neg_below = np.cumsum(neg) - neg
        auc = float(np.dot(pos, neg_below + 0.5 * neg) / (p * n))
        error = float(0.5 * np.dot(pos, neg) / (p * n))
        tp, fp = np.cumsum(pos[::-1]), np.cumsum(neg[::-1])
        precision = np.divide(tp, tp + fp, out=np.ones_like(tp), where=tp + fp > 0)
        recall_delta = pos[::-1] / p
        ap = float(np.dot(recall_delta, precision))
        pr_auc = float(np.dot(recall_delta, (precision + np.r_[1.0, precision[:-1]]) * 0.5))
        true_positive = float(pos[len(pos) // 2 :].sum())
        predicted_positive = true_positive + float(neg[len(neg) // 2 :].sum())
        return {
            **result,
            "roc_auc": auc,
            "roc_auc_error_bound": error,
            "roc_auc_lower_bound": max(0.0, auc - error),
            "roc_auc_upper_bound": min(1.0, auc + error),
            "average_precision": ap,
            "pr_auc_trapezoidal": pr_auc,
            "dice_at_0_5": 2 * true_positive / (p + predicted_positive),
            "recall_at_0_5": true_positive / p,
        }


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def evaluate(run_dir: Path, bins: int, overlap: float) -> None:
    model_dir = run_dir / "model"
    output = run_dir / "full-volume-auc"
    output.mkdir(exist_ok=False)
    started = time.monotonic()

    def emit(phase: str, **info: object) -> None:
        event = {
            "phase": phase,
            "seconds": time.monotonic() - started,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            **info,
        }
        print(json.dumps(event), flush=True)
        write_json(output / "state.json", event)
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")

    checkpoint = model_dir / "weights_last.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = state["model_config"]
    if config["architecture_config"]["dimensions"] != "3d" or config["task"] != "instance_friendly":
        raise ValueError("This evaluation harness requires a 3D instance-friendly model")
    exported = model_dir / "model.pt"
    if exported.exists():
        raise FileExistsError(f"Refusing to overwrite {exported}")
    state.pop("optimizer_state_dict", None)
    state.pop("scheduler_state_dict", None)
    torch.save(state, exported.with_suffix(".pt.tmp"))
    exported.with_suffix(".pt.tmp").replace(exported)
    checkpoint_info = {"epoch": state["epoch"], "step": state.get("metrics", {}).get("step")}
    del state
    gc.collect()
    plan = json.loads((model_dir / "dataset_plan.json").read_text())
    metadata = json.loads((model_dir / "model_metadata.json").read_text())
    spacings = {item["case"]: tuple(item["spacing"]) for item in plan["spacing"]["cases"]}
    target_spacing = tuple(metadata["dataset_plan"]["target_spacing"])
    scale_cfg = config["training"]["instance_scale_normalization"]
    target_size = metadata["instance_scale"]["target_object_size"]
    patch_size = tuple(config["training"]["patch_size"])
    torch.set_num_threads(4)
    device = torch.device("cuda")
    model, _ = load_model(exported, device)
    pooled = ProbabilityHistogram(bins)
    report = {
        "checkpoint": str(exported),
        "checkpoint_info": checkpoint_info,
        "task": "foreground_vs_background",
        "positive_definition": "original_mask > 0",
        "evaluation_grid": "all native voxels; no sampling or background exclusion",
        "object_size_source": "saved mask-derived validation estimates; nominal scale, no jitter",
        "aggregation": "per-case, unweighted case mean, and pooled voxel-weighted",
        "auc_method": "probability histogram; ROC error bound reported; PR metrics are quantized estimates",
        "tile_overlap": overlap,
        "cases": [],
    }
    emit("model_loaded", checkpoint=str(exported), **checkpoint_info)
    for domain in plan["validation_domains"]:
        case_started = time.monotonic()
        stem = domain["stem"]
        native_shape = tuple(domain["spatial_shape"])
        if domain["mask_axes"] != "ZYX" or domain["region"] != [[0, v] for v in native_shape]:
            raise ValueError("This harness expects complete ZYX validation volumes")
        emit("loading_and_normalizing", case=stem)
        image = normalize_image(load_image(domain["image"], dimensions="3d"), config["normalization"])
        if tuple(image.shape[1:]) != native_shape:
            raise ValueError("Image geometry differs from the saved dataset plan")
        if not np.allclose(spacings[stem], target_spacing):
            image, _ = resample_image_mask(
                image, np.zeros(native_shape, dtype=np.uint8), spacings[stem], target_spacing
            )
        object_size = plan["validation_instance_sizes"][stem]
        scale = (
            float(
                np.clip(target_size / object_size, scale_cfg["min_effective_scale"], scale_cfg["max_effective_scale"])
            )
            if scale_cfg["enabled"]
            else 1.0
        )
        scaled_shape = tuple(max(1, round(length * scale)) for length in image.shape[1:])
        if scaled_shape != tuple(image.shape[1:]):
            image = F.interpolate(
                torch.from_numpy(image[None]), size=scaled_shape, mode="trilinear", align_corners=False
            )[0].numpy()
        layout = _tile_layout(tuple(image.shape[1:]), patch_size, overlap)
        last_update = [0.0]

        def callback(event: dict, case: str = stem, total: int = layout.count, last: list[float] = last_update) -> None:
            now = time.monotonic()
            if event.get("phase") == "patch_end" and (now - last[0] >= 10 or event["current"] == total):
                last[0] = now
                emit("inference", case=case, current=event["current"], total=total)

        emit(
            "inference_start",
            case=stem,
            native_shape=native_shape,
            scaled_shape=scaled_shape,
            scale=scale,
            object_size=object_size,
            total_tiles=layout.count,
        )
        progress = _InferenceProgress(CallbackDispatcher(callback), total_patches=layout.count)
        torch.cuda.reset_peak_memory_stats()
        logits = tiled_predict(model, image, device, patch_size, overlap, layout=layout, progress=progress)
        del image
        gc.collect()
        torch.cuda.empty_cache()
        # Keep only the compact network-grid foreground map; native maps are transient.
        compact_path = output / f"{stem}_foreground_logits.npy"
        np.save(compact_path, logits[0])
        emit("restoring_native_grid", case=stem)
        foreground_logits = restore_continuous_maps(logits[:1], native_shape)[0]
        del logits
        gc.collect()
        mask = tifffile.imread(domain["mask"])
        if mask.shape != native_shape or mask.dtype.kind not in "bui":
            raise ValueError("Unexpected original mask shape or dtype")
        hist = ProbabilityHistogram(bins)
        emit("scoring_all_voxels", case=stem, voxels=int(np.prod(native_shape)))
        for z in range(native_shape[0]):
            hist.update(_sigmoid(foreground_logits[z]), mask[z] > 0)
            if z % 20 == 0 or z == native_shape[0] - 1:
                emit("scoring_all_voxels", case=stem, current=z + 1, total=native_shape[0])
        expected_foreground = sum(domain["plane_positive_counts"])
        if int(hist.positive.sum()) != expected_foreground:
            raise ValueError("Foreground voxel count differs from the saved source analysis")
        metrics = hist.metrics()
        pooled.positive += hist.positive
        pooled.negative += hist.negative
        np.savez_compressed(output / f"{stem}_score_histogram.npz", positive=hist.positive, negative=hist.negative)
        case = {
            "case": stem,
            "image": domain["image"],
            "mask": domain["mask"],
            "native_shape": native_shape,
            "scaled_shape": scaled_shape,
            "instance_scale": scale,
            "compact_foreground_logits": str(compact_path),
            "restoration": "restore_continuous_maps then sigmoid, as library 3D inference",
            "seconds": time.monotonic() - case_started,
            "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            **metrics,
        }
        report["cases"].append(case)
        report["pooled"] = pooled.metrics()
        for key in ("roc_auc", "average_precision", "pr_auc_trapezoidal", "dice_at_0_5"):
            values = [item[key] for item in report["cases"] if item[key] is not None]
            report.setdefault("macro_mean", {})[key] = float(np.mean(values)) if values else None
        report["seconds"] = time.monotonic() - started
        write_json(output / "results.json", report)
        emit("case_complete", **{key: value for key, value in case.items() if key != "seconds"}, case_seconds=case["seconds"])
        del foreground_logits, mask, hist
        gc.collect()
    np.savez_compressed(output / "pooled_score_histogram.npz", positive=pooled.positive, negative=pooled.negative)
    emit("complete", results=str(output / "results.json"), macro_mean=report["macro_mean"], pooled=report["pooled"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=2**20)
    parser.add_argument("--overlap", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.overlap < 1:
        parser.error("overlap must be in [0, 1)")
    evaluate(args.run_dir, args.bins, args.overlap)
