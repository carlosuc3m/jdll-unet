"""Native-grid instance IoU threshold sweep for an existing 3D training run.

No training or dataset planning is repeated. Only compact network-grid logits
are cached. Reconstruction is isolated into disconnected foreground regions,
including regions with no annotations, so false positives are not excluded.
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
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

from benchmarks.evaluate_full_volume_auc import write_json
from jdll_unet.infer import _sigmoid, load_model, tiled_predict
from jdll_unet.io import fit_normalization, load_image, normalize_image
from jdll_unet.planning import restore_continuous_maps
from jdll_unet.postprocess import postprocess_instance


def native_region(compact: np.ndarray, shape: tuple[int, ...], box: tuple[slice, ...]) -> np.ndarray:
    """Evaluate the same trilinear coordinates as restore_continuous_maps."""
    factors = np.array([(a - 1) / (b - 1) if b > 1 else 0.0 for a, b in zip(compact.shape, shape, strict=True)])
    starts = np.array([part.start for part in box])
    return ndi.affine_transform(
        compact, np.diag(factors), offset=starts * factors,
        output_shape=tuple(part.stop - part.start for part in box),
        order=1, mode="nearest", prefilter=False,
    )


def reconstruct(foreground: np.ndarray, boundary: np.ndarray, distance: np.ndarray, settings: dict) -> np.ndarray:
    # If no seed can survive the distance cutoff, the library places exactly one
    # fallback marker in each face-connected component. Watershed cannot split it.
    clean_max = float((distance * foreground * (1 - boundary)).max())
    if (
        settings["method"] == "distance_boundary_watershed"
        and clean_max < settings["seed_distance_threshold"]
        and settings.get("min_object_size_physical") is None
        and settings.get("min_object_size", 0) == 0
    ):
        rank = foreground.ndim
        structure = ndi.generate_binary_structure(rank, 1 if settings["connectivity"] == "face" else rank)
        return ndi.label(foreground >= settings["threshold"], structure=structure)[0]
    if settings.get("min_object_size_physical") is not None or settings.get("min_object_size", 0):
        return postprocess_instance(foreground, boundary, distance, **settings)["labels"]
    components, count = ndi.label(
        foreground >= settings["threshold"], structure=np.ones((3,) * foreground.ndim, dtype=bool),
    )
    result = np.zeros(foreground.shape, dtype=np.uint32)
    offset = 0
    for region_id, tight in enumerate(ndi.find_objects(components, max_label=count), start=1):
        box = tuple(
            slice(max(0, s.start - 1), min(size, s.stop + 1))
            for s, size in zip(tight, foreground.shape, strict=True)
        )
        fg = np.where(components[box] == region_id, foreground[box], 0)
        labels = postprocess_instance(fg, boundary[box], distance[box], **settings)["labels"]
        selected = labels > 0
        result[box][selected] = labels[selected] + offset
        offset += int(labels.max())
    return result


class InstanceOverlap:
    """Contingency counts avoid scanning every prediction for every true object."""

    def __init__(self, truth_sizes: np.ndarray) -> None:
        self.truth_sizes = np.asarray(truth_sizes, dtype=np.int64)
        self.prediction_sizes: list[int] = []
        self.overlaps: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def update(self, labels: np.ndarray, truth: np.ndarray) -> None:
        if labels.shape != truth.shape:
            raise ValueError("Prediction and truth shapes differ")
        values, sizes = np.unique(labels, return_counts=True)
        positive = values > 0
        values, sizes = values[positive], sizes[positive]
        if not len(values):
            return
        offset = len(self.prediction_sizes)
        self.prediction_sizes.extend(int(size) for size in sizes)
        both = (labels > 0) & (truth > 0)
        local = np.searchsorted(values, labels[both])
        encoded = truth[both].astype(np.int64) * len(values) + local
        pairs, counts = np.unique(encoded, return_counts=True)
        self.overlaps.append((pairs // len(values), pairs % len(values) + offset, counts))

    def metrics(self) -> dict:
        truth_ids = np.flatnonzero(self.truth_sizes)
        truth_ids = truth_ids[truth_ids != 0]
        truth_areas = self.truth_sizes[truth_ids]
        prediction_areas = np.asarray(self.prediction_sizes, dtype=np.int64)
        intersection = np.zeros((len(truth_ids), len(prediction_areas)), dtype=np.float64)
        for gt, pred, counts in self.overlaps:
            intersection[np.searchsorted(truth_ids, gt), pred] += counts
        union = truth_areas[:, None] + prediction_areas[None] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        gt_indices, pred_indices = linear_sum_assignment(iou, maximize=True)
        assigned = np.zeros(len(truth_ids), dtype=np.float64)
        assigned[gt_indices] = iou[gt_indices, pred_indices]
        matched = assigned > 0
        result = {
            "ground_truth_instances": len(truth_ids), "predicted_instances": len(prediction_areas),
            "sum_matched_iou": float(assigned.sum()),
            "mean_gt_iou": float(assigned.mean()) if len(assigned) else None,
            "matched_only_mean_iou": float(assigned[matched].mean()) if matched.any() else None,
            "unmatched_gt": int((~matched).sum()),
            "per_gt_iou": {str(int(label)): float(score) for label, score in zip(truth_ids, assigned, strict=True)},
        }
        for threshold in (0.25, 0.5, 0.75):
            eligible = iou >= threshold
            # Maximize match count first, then overlap among equal-count matches.
            quality = eligible * (1 + iou / (min(iou.shape) + 1))
            rows, cols = linear_sum_assignment(quality, maximize=True)
            tp = int(eligible[rows, cols].sum())
            fp, fn = len(prediction_areas) - tp, len(truth_ids) - tp
            denominator = tp + 0.5 * fp + 0.5 * fn
            result[f"matching_iou_{threshold:g}"] = {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else None,
                "f1": tp / denominator if denominator else None,
                "pq": float(iou[rows, cols][eligible[rows, cols]].sum() / denominator) if denominator else None,
            }
        return result


def prepare_truth(mask: np.ndarray, expected_extra: int) -> tuple[np.ndarray, np.ndarray]:
    """Recreate deleted run-scoped repairs within tight per-ID bounding boxes."""
    maximum = int(mask.max())
    if maximum + expected_extra > np.iinfo(mask.dtype).max:
        raise ValueError("This evaluation requires a wider compact mask dtype for repaired IDs")
    if expected_extra:
        boxes = ndi.find_objects(mask, max_label=maximum)
        next_id = maximum + 1
        for label, box in enumerate(boxes, start=1):
            if box is None:
                continue
            region = mask[box]
            components, count = ndi.label(region == label)
            for component in range(2, count + 1):
                region[components == component] = next_id
                next_id += 1
        if next_id - maximum - 1 != expected_extra:
            raise ValueError("Ground-truth repairs differ from the saved annotation analysis")
    sizes = np.zeros(int(mask.max()) + 1, dtype=np.int64)
    for plane in mask:
        sizes += np.bincount(plane.ravel(), minlength=len(sizes))
    sizes[0] = 0
    return mask, sizes


def coarse_settings(base: dict) -> list[dict]:
    return [
        {**base, "threshold": threshold}
        for threshold in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999)
    ]


def key(settings: dict) -> str:
    return f"fg={settings['threshold']:g},distance={settings['seed_distance_threshold']:g},boundary={settings['seed_boundary_threshold']:g}"


def aggregate(cases: list[dict]) -> dict:
    gt = sum(case["ground_truth_instances"] for case in cases)
    iou_sum = sum(case["sum_matched_iou"] for case in cases)
    matching = [case["matching_iou_0.5"] for case in cases]
    tp, fp, fn = (sum(item[name] for item in matching) for name in ("tp", "fp", "fn"))
    return {
        "macro_mean_gt_iou": float(np.mean([case["mean_gt_iou"] for case in cases if case["mean_gt_iou"] is not None])),
        "pooled_mean_gt_iou": iou_sum / gt if gt else None,
        "ground_truth_instances": gt, "predicted_instances": sum(case["predicted_instances"] for case in cases),
        "tp_at_0_5": tp, "fp_at_0_5": fp, "fn_at_0_5": fn,
        "f1_at_0_5": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def write_summary(output: Path) -> None:
    report = json.loads((output / "results.json").read_text())
    results = report["results"]
    case_count = max(len(item["cases"]) for item in results.values())
    complete = {name: item for name, item in results.items() if len(item["cases"]) == case_count}
    pooled_key = max(complete, key=lambda name: complete[name]["aggregate"]["pooled_mean_gt_iou"])
    macro_key = max(complete, key=lambda name: complete[name]["aggregate"]["macro_mean_gt_iou"])
    report["best_by_pooled_gt_iou"] = pooled_key
    report["best_by_case_mean_gt_iou"] = macro_key
    write_json(output / "results.json", report)
    best = complete[pooled_key]
    baseline = results["fg=0.5,distance=0.35,boundary=0.5"]
    override = {name: best["settings"][name] for name in ("threshold", "seed_distance_threshold", "seed_boundary_threshold")}
    write_json(output / "recommended_postprocessing.json", {"postprocessing": override})
    lines = [
        "# Full-Volume Instance IoU", "",
        f"Model: `{report['checkpoint']}`. Training remains stopped; model/configuration files were not changed.", "",
        "## Definition", "",
        "Predicted and annotated instances are matched one-to-one to maximize total IoU. "
        "Every annotated object contributes equally to the pooled mean; unmatched objects contribute zero. "
        "Unmatched predictions are reported separately, not penalized by mean GT IoU.", "",
        "All native-resolution voxels and all reference objects are evaluated. "
        "Disconnected source IDs use the training run's face-connectivity repair policy. "
        "Object sizes come from the saved validation metadata, with nominal scaling and no jitter. "
        "Inference uses tile overlap 0.5.", "",
        "These settings were selected using these validation annotations. "
        "The selected score is not an independent test-set estimate.", "",
        "## Best Shared Settings", "", "```json", json.dumps({"postprocessing": override}, indent=2), "```", "",
        f"Best by equal object weight: **{best['aggregate']['pooled_mean_gt_iou']:.6f}** mean IoU "
        f"across **{best['aggregate']['ground_truth_instances']}** objects; "
        f"default thresholds: **{baseline['aggregate']['pooled_mean_gt_iou']:.6f}**.", "",
        f"Best by equal volume weight instead: `{macro_key}`, "
        f"case-mean IoU **{complete[macro_key]['aggregate']['macro_mean_gt_iou']:.6f}**.", "",
        "| Volume | Objects | Default mean IoU | Selected mean IoU | Matched at IoU >= 0.5 | Unmatched predictions at 0.5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, case in best["cases"].items():
        match = case["matching_iou_0.5"]
        lines.append(
            f"| {name} | {case['ground_truth_instances']} | {baseline['cases'][name]['mean_gt_iou']:.6f} "
            f"| {case['mean_gt_iou']:.6f} | {match['tp']} | {match['fp']} |"
        )
    totals = best["aggregate"]
    lines.extend([
        "", f"At matching IoU >= 0.5: {totals['tp_at_0_5']} TP, {totals['fp_at_0_5']} FP, "
        f"{totals['fn_at_0_5']} FN; object F1 = {totals['f1_at_0_5']:.6f}.", "",
        "No small-object filtering was applied (`min_object_size=0`). "
        "Foreground, distance-seed, and boundary-seed cutoffs are distinct from the 0.5 IoU criterion used for matching.", "",
        "## Search", "",
        "17 foreground thresholds from 0.30 to 0.999 were evaluated with default seed settings. "
        "The three leading foreground cutoffs were then combined with distance-seed thresholds "
        "[0.15, 0.25, 0.35, 0.5] and boundary-seed thresholds [0.3, 0.5, 0.7]. "
        f"Total: {len(complete)} settings, each evaluated on all {case_count} volumes.", "",
        "Other settings stayed fixed: `seed_h=0.1`, `min_seed_size=3`, `boundary_weight=1`, "
        "face connectivity, distance/boundary watershed. This is a finite threshold search, not proof of a global optimum.", "",
        "`results.json` contains every tested configuration, per-case scores, per-reference-object IoU, "
        "and detection matching at IoU 0.25, 0.5 and 0.75. "
        "`recommended_postprocessing.json` contains inference overrides only; it does not modify the saved model.", "",
    ])
    (output / "README.md").write_text("\n".join(lines))


def evaluate(run_dir: Path) -> None:
    model_dir = run_dir / "model"
    output = run_dir / "instance-iou"
    output.mkdir(exist_ok=True)
    started = time.monotonic()

    def emit(phase: str, **info: object) -> None:
        event = {
            "phase": phase, "seconds": time.monotonic() - started,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, **info,
        }
        print(json.dumps(event), flush=True)
        write_json(output / "state.json", event)
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")

    config = json.loads((model_dir / "config.json").read_text())
    plan = json.loads((model_dir / "dataset_plan.json").read_text())
    metadata = json.loads((model_dir / "model_metadata.json").read_text())
    target_spacing = tuple(metadata["dataset_plan"]["target_spacing"])
    spacings = {item["case"]: tuple(item["spacing"]) for item in plan["spacing"]["cases"]}
    repairs = {item["path"]: item for item in plan["annotation_preparation"]["sources"]}
    cases = plan["validation_domains"]
    torch.set_num_threads(4)
    device = torch.device("cuda")
    model, _ = load_model(model_dir, device)
    scale_cfg = config["training"]["instance_scale_normalization"]
    base = config["postprocessing"]
    report = {
        "checkpoint": str(model_dir / "model.pt"),
        "matching": "One-to-one maximum-total-IoU assignment; every GT instance contributes, unmatched GT scores zero",
        "selection": "Highest unweighted case-mean GT IoU; validation-tuned, not an independent test estimate",
        "grid": "Full original-resolution volumes; no voxel or object sampling; all predicted regions included",
        "repair": "Same face-connected source-ID splitting as saved training preparation; original files unchanged",
        "inference": "Saved validation object sizes, nominal scaling, overlap=0.5, float32",
        "results": {},
    }
    prior_seconds = 0.0
    results_path = output / "results.json"
    if results_path.exists():
        previous_report = json.loads(results_path.read_text())
        if previous_report["checkpoint"] != report["checkpoint"]:
            raise ValueError("Cached evaluation belongs to a different model")
        report["results"] = previous_report["results"]
        prior_seconds = previous_report.get("seconds", 0.0)
    # Cache all three output channels once. Normalize one native plane at a time
    # with whole-volume statistics to bound temporaries without changing values.
    for domain in cases:
        stem = domain["stem"]
        cache = output / f"{stem}_logits.npy"
        if cache.exists():
            continue
        shape = tuple(domain["spatial_shape"])
        if domain["mask_axes"] != "ZYX" or domain["region"] != [[0, size] for size in shape]:
            raise ValueError("This harness expects complete ZYX volumes")
        if not np.allclose(spacings[stem], target_spacing):
            raise ValueError("This saved-run harness requires native spacing equal to target spacing")
        emit("loading_image", case=stem)
        image = load_image(domain["image"], dimensions="3d")
        statistics = fit_normalization(image, config["normalization"])
        for z in range(image.shape[1]):
            image[:, z] = normalize_image(image[:, z], statistics=statistics)
        object_size = plan["validation_instance_sizes"][stem]
        scale = float(np.clip(
            metadata["instance_scale"]["target_object_size"] / object_size,
            scale_cfg["min_effective_scale"], scale_cfg["max_effective_scale"],
        )) if scale_cfg["enabled"] else 1.0
        scaled_shape = tuple(max(1, round(size * scale)) for size in shape)
        image = F.interpolate(torch.from_numpy(image[None]), size=scaled_shape, mode="trilinear", align_corners=False)[0].numpy()
        emit("inference", case=stem, shape=scaled_shape)
        logits = tiled_predict(model, image, device, tuple(config["training"]["patch_size"]), overlap=0.5)
        previous = np.load(run_dir / "full-volume-auc" / f"{stem}_foreground_logits.npy", mmap_mode="r")
        if not np.allclose(logits[0], previous, atol=1e-5, rtol=1e-5):
            raise ValueError("Foreground inference differs from the preceding evaluation")
        with cache.with_suffix(".tmp").open("wb") as stream:
            np.save(stream, logits)
        cache.with_suffix(".tmp").replace(cache)
        emit("logits_cached", case=stem, bytes=cache.stat().st_size)
        del image, logits, previous
        gc.collect()
        torch.cuda.empty_cache()

    def run_sweep(settings: list[dict], stage: str) -> None:
        minimum = min(item["threshold"] for item in settings)
        for domain in cases:
            stem = domain["stem"]
            if all(stem in report["results"].get(key(item), {}).get("cases", {}) for item in settings):
                emit("case_cached", stage=stage, case=stem)
                continue
            shape = tuple(domain["spatial_shape"])
            emit("native_regions", stage=stage, case=stem, settings=len(settings))
            logits = np.load(output / f"{stem}_logits.npy", mmap_mode="r")
            foreground = restore_continuous_maps(logits[:1], shape)[0]
            for z in range(shape[0]):
                foreground[z] = _sigmoid(foreground[z])
            support, count = ndi.label(foreground >= minimum, structure=np.ones((3, 3, 3), dtype=bool))
            boxes = ndi.find_objects(support, max_label=count)
            record = repairs[str(Path(domain["mask"]).resolve())]
            truth, sizes = prepare_truth(tifffile.imread(domain["mask"]), record["extra_components"])
            if int(sizes.sum()) != sum(domain["plane_positive_counts"]):
                raise ValueError("Validation foreground changed since dataset analysis")
            accumulators = [InstanceOverlap(sizes) for _ in settings]
            last_update = time.monotonic()
            emit("reconstructing", stage=stage, case=stem, regions=count, gt_instances=int((sizes > 0).sum()))
            for region_id, tight in enumerate(boxes, start=1):
                if tight is None:
                    continue
                box = tuple(slice(max(0, s.start - 1), min(size, s.stop + 1)) for s, size in zip(tight, shape, strict=True))
                selected = support[box] == region_id
                fg = np.where(selected, foreground[box], 0)
                highest = float(fg.max())
                applicable = [i for i, item in enumerate(settings) if item["threshold"] <= highest]
                if not applicable:
                    continue
                boundary = _sigmoid(native_region(logits[1], shape, box))
                distance = _sigmoid(native_region(logits[2], shape, box))
                for index in applicable:
                    labels = reconstruct(fg, boundary, distance, {**settings[index], "spacing": spacings[stem]})
                    accumulators[index].update(labels, truth[box])
                if time.monotonic() - last_update > 20:
                    emit("reconstructing", stage=stage, case=stem, current=region_id, total=count)
                    last_update = time.monotonic()
            for settings_item, accumulator in zip(settings, accumulators, strict=True):
                result = report["results"].setdefault(key(settings_item), {"settings": settings_item, "cases": {}})
                result["cases"][stem] = accumulator.metrics()
                result["aggregate"] = aggregate(list(result["cases"].values()))
            report["seconds"] = prior_seconds + time.monotonic() - started
            write_json(output / "results.json", report)
            emit("case_complete", stage=stage, case=stem)
            del logits, foreground, support, truth, accumulators
            gc.collect()

    coarse = coarse_settings(base)
    run_sweep(coarse, "foreground")
    ranked = sorted(
        (report["results"][key(item)] for item in coarse),
        key=lambda item: item["aggregate"]["macro_mean_gt_iou"], reverse=True,
    )
    best_foregrounds = [item["settings"]["threshold"] for item in ranked[:3]]
    emit("foreground_sweep_complete", best=ranked[0]["settings"], scores=ranked[0]["aggregate"])
    refinement = [
        {**base, "threshold": threshold, "seed_distance_threshold": distance, "seed_boundary_threshold": boundary}
        for threshold in best_foregrounds for distance in (0.15, 0.25, 0.35, 0.5) for boundary in (0.3, 0.5, 0.7)
        if key({"threshold": threshold, "seed_distance_threshold": distance, "seed_boundary_threshold": boundary}) not in report["results"]
    ]
    run_sweep(refinement, "seeds")
    ranked = sorted(report["results"], key=lambda name: report["results"][name]["aggregate"]["macro_mean_gt_iou"], reverse=True)
    report["best_by_mean_gt_iou"] = ranked[0]
    report["best_by_f1_at_0_5"] = max(report["results"], key=lambda name: report["results"][name]["aggregate"]["f1_at_0_5"])
    report["seconds"] = prior_seconds + time.monotonic() - started
    report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    write_json(output / "results.json", report)
    write_summary(output)
    emit("complete", best=ranked[0], scores=report["results"][ranked[0]]["aggregate"], results=str(output / "results.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    evaluate(parser.parse_args().run_dir)
