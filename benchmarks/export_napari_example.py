"""Export evaluated predictions and their seed diagnostics without rerunning inference."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.morphology import h_maxima

from benchmarks.evaluate_full_volume_auc import write_json
from benchmarks.evaluate_instance_iou import InstanceOverlap, native_region, prepare_truth, reconstruct
from jdll_unet.infer import _sigmoid
from jdll_unet.planning import restore_continuous_maps


def seed_details(foreground: np.ndarray, boundary: np.ndarray, distance: np.ndarray, settings: dict) -> tuple:
    inside = foreground >= settings["threshold"]
    clean = distance * foreground * (1 - boundary)
    clean[~inside] = 0
    structure = ndi.generate_binary_structure(inside.ndim, 1)
    if float(clean.max()) >= settings["seed_distance_threshold"]:
        seeds = h_maxima(clean, h=settings["seed_h"])
        seeds &= clean >= settings["seed_distance_threshold"]
        seeds &= boundary < settings["seed_boundary_threshold"]
        seeds &= inside
    else:
        seeds = np.zeros(inside.shape, dtype=bool)
    candidates, candidate_count = ndi.label(seeds)
    sizes = np.bincount(candidates.ravel())
    keep = sizes >= settings["min_seed_size"]
    keep[0] = False
    markers, learned_count = ndi.label(keep[candidates], structure=structure)
    components, component_count = ndi.label(inside, structure=structure)
    seeded = {int(v) for v in np.unique(components[markers > 0])}
    learned = np.array(ndi.center_of_mass(seeds, markers, range(1, learned_count + 1))).reshape(-1, inside.ndim)
    fallback = []
    for component_id, box in enumerate(ndi.find_objects(components, max_label=component_count), start=1):
        if component_id in seeded:
            continue
        values = np.where(components[box] == component_id, clean[box], -np.inf)
        position = np.array(np.unravel_index(int(values.argmax()), values.shape))
        fallback.append(position + np.array([s.start for s in box]))
    stats = {
        "foreground_components": component_count,
        "candidate_seed_components": candidate_count,
        "discarded_small_seed_components": candidate_count - learned_count,
        "learned_seed_components": learned_count,
        "fallback_seed_components": len(fallback),
    }
    return learned, np.array(fallback, dtype=float).reshape(-1, inside.ndim), stats


def choose_crop(truth: np.ndarray, scores: dict[str, float], shape: tuple[int, ...]) -> tuple[slice, ...]:
    boxes = ndi.find_objects(truth)
    entries = [(label, box) for label, box in enumerate(boxes, start=1) if box is not None]
    length = np.minimum(shape, truth.shape)
    candidates = []
    for _, box in entries:
        center = np.array([(s.start + s.stop) // 2 for s in box])
        start = np.clip(center - length // 2, 0, np.array(truth.shape) - length)
        stop = start + length
        contained = [
            label for label, region in entries
            if all(a <= s.start and s.stop <= b for s, a, b in zip(region, start, stop, strict=True))
        ]
        values = [scores[str(label)] for label in contained]
        mixed = any(value == 0 for value in values) and any(value >= 0.5 for value in values)
        candidates.append(((mixed, len(contained)), start, stop))
    if not candidates:
        raise ValueError("No annotated instances in the example")
    _, start, stop = max(candidates, key=lambda item: item[0])
    return tuple(slice(int(a), int(b)) for a, b in zip(start, stop, strict=True))


def write_volume(path: Path, array: np.ndarray, spacing: tuple[float, ...]) -> None:
    metadata = {"axes": "ZYX"}
    for axis, value in zip("ZYX", spacing, strict=True):
        metadata[f"PhysicalSize{axis}"] = value
        metadata[f"PhysicalSize{axis}Unit"] = "\u00b5m"
    tifffile.imwrite(
        path, array, ome=True, photometric="minisblack", metadata=metadata,
        compression="zlib", compressionargs={"level": 1}, maxworkers=1,
    )


def export(run: Path, case: str) -> None:
    started = time.monotonic()
    output = run / f"napari-{case}"
    output.mkdir(exist_ok=False)
    report = json.loads((run / "instance-iou/results.json").read_text())
    selected = report["results"][report["best_by_pooled_gt_iou"]]
    settings, expected = selected["settings"], selected["cases"][case]
    plan = json.loads((run / "model/dataset_plan.json").read_text())
    domain = next(item for item in plan["validation_domains"] if item["stem"] == case)
    spacing = tuple(next(item["spacing"] for item in plan["spacing"]["cases"] if item["case"] == case))
    shape = tuple(domain["spatial_shape"])
    if settings["connectivity"] != "face" or settings["min_object_size"] or settings["min_seed_size_physical"]:
        raise ValueError("Seed export expects this run's face connectivity and voxel-based size settings")
    with tifffile.TiffFile(domain["image"]) as source:
        if (source.imagej_metadata or {}).get("unit") != "um":
            raise ValueError("Verify physical units before exporting OME spacing")
    print("Restoring cached predictions; no new model inference", flush=True)
    logits = np.load(run / "instance-iou" / f"{case}_logits.npy", mmap_mode="r")
    foreground = restore_continuous_maps(logits[:1], shape)[0]
    for z in range(shape[0]):
        foreground[z] = _sigmoid(foreground[z])
    support, count = ndi.label(foreground >= settings["threshold"], structure=np.ones((3, 3, 3)))
    record = next(item for item in plan["annotation_preparation"]["sources"] if item["path"] == str(Path(domain["mask"]).resolve()))
    truth, truth_sizes = prepare_truth(tifffile.imread(domain["mask"]), record["extra_components"])
    dtype = np.uint16 if expected["predicted_instances"] <= 65535 else np.uint32
    prediction = np.zeros(shape, dtype=dtype)
    overlaps = InstanceOverlap(truth_sizes)
    centers: dict[str, list] = {"learned": [], "fallback": []}
    totals: dict[str, int] = {}
    offset = 0
    last = time.monotonic()
    for region_id, tight in enumerate(ndi.find_objects(support, max_label=count), start=1):
        box = tuple(slice(max(0, s.start - 1), min(n, s.stop + 1)) for s, n in zip(tight, shape, strict=True))
        fg = np.where(support[box] == region_id, foreground[box], 0)
        boundary = _sigmoid(native_region(logits[1], shape, box))
        distance = _sigmoid(native_region(logits[2], shape, box))
        labels = reconstruct(fg, boundary, distance, {**settings, "spacing": spacing})
        positive = labels > 0
        prediction[box][positive] = labels[positive] + offset
        overlaps.update(labels, truth[box])
        offset += int(labels.max())
        learned, fallback, stats = seed_details(fg, boundary, distance, settings)
        origin = np.array([s.start for s in box])
        centers["learned"].extend((learned + origin).tolist())
        centers["fallback"].extend((fallback + origin).tolist())
        for name, value in stats.items():
            totals[name] = totals.get(name, 0) + value
        if time.monotonic() - last >= 15:
            print(json.dumps({"region": region_id, "total": count, "seconds": time.monotonic() - started}), flush=True)
            last = time.monotonic()
    actual = overlaps.metrics()
    if offset != expected["predicted_instances"] or not np.isclose(actual["mean_gt_iou"], expected["mean_gt_iou"], atol=1e-10, rtol=0):
        raise ValueError("Export does not reproduce evaluated instance counts/IoU")
    if len(centers["learned"]) + len(centers["fallback"]) != offset:
        raise ValueError("Seed count does not match reconstructed instances")
    print(json.dumps({"verified_iou": actual["mean_gt_iou"], "seeds": totals}), flush=True)
    crop = choose_crop(truth, actual["per_gt_iou"], (64, 512, 512))
    crop_shape = tuple(s.stop - s.start for s in crop)
    pred_sizes = np.asarray(overlaps.prediction_sizes)
    intersections = np.zeros((len(truth_sizes), len(pred_sizes)), dtype=np.int64)
    for gt, pred, counts in overlaps.overlaps:
        intersections[gt, pred] += counts
    union = truth_sizes[:, None] + pred_sizes[None] - intersections
    iou = np.divide(intersections, union, out=np.zeros_like(intersections, dtype=float), where=union > 0)
    rows, cols = linear_sum_assignment(iou[1:], maximize=True)
    assigned = {int(row + 1): int(col) for row, col in zip(rows, cols, strict=True) if iou[row + 1, col] > 0}
    gt_boxes = ndi.find_objects(truth)
    rows = []
    for label in np.flatnonzero(truth_sizes):
        matched = assigned.get(int(label))
        box = gt_boxes[label - 1]
        rows.append({
            "gt_id": int(label), "gt_voxels": int(truth_sizes[label]),
            "matched_prediction_id": matched + 1 if matched is not None else 0,
            "iou": float(actual["per_gt_iou"][str(label)]),
            "best_any_prediction_iou": float(iou[label].max()),
            "foreground_recall": float(intersections[label].sum() / truth_sizes[label]),
            "fully_in_crop": all(c.start <= s.start and s.stop <= c.stop for c, s in zip(crop, box, strict=True)),
            **{f"{axis}_{side}": getattr(s, side) for axis, s in zip("zyx", box, strict=True) for side in ("start", "stop")},
        })
    with (output / "objects.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    diagnostics = {
        "seeds": totals, "ground_truth_instances": len(rows),
        "gt_with_no_predicted_foreground": sum(row["foreground_recall"] == 0 for row in rows),
        "gt_overlapping_prediction_but_unmatched": sum(row["foreground_recall"] > 0 and row["iou"] == 0 for row in rows),
        "predicted_instances": offset,
        "prediction_size_quantiles_voxels": np.quantile(pred_sizes, [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1]).tolist(),
        "predictions_below_size": {str(size): int((pred_sizes < size).sum()) for size in (3, 10, 100, 1000)},
        "predictions_without_any_gt_overlap": int((intersections.sum(axis=0) == 0).sum()),
        "full_volume_metrics": actual,
    }
    write_json(output / "diagnostics.json", diagnostics)
    np.savez_compressed(output / "seed_centers.npz", **{name: np.asarray(points).reshape(-1, 3) for name, points in centers.items()})
    print("Writing native-grid TIFFs and compact viewer crop", flush=True)
    write_volume(output / "full_prediction.ome.tif", prediction, spacing)
    write_volume(output / "full_ground_truth.ome.tif", truth, spacing)
    write_volume(output / "prediction.ome.tif", prediction[crop], spacing)
    write_volume(output / "ground_truth.ome.tif", truth[crop], spacing)
    write_volume(output / "foreground_probability.ome.tif", foreground[crop], spacing)
    del foreground, support, prediction, truth, overlaps
    gc.collect()
    for channel, name in ((1, "boundary"), (2, "distance")):
        write_volume(output / f"{name}_probability.ome.tif", _sigmoid(native_region(logits[channel], shape, crop)), spacing)
    image = tifffile.imread(domain["image"], key=range(crop[0].start, crop[0].stop))[:, crop[1], crop[2]]
    write_volume(output / "image.ome.tif", image, spacing)
    manifest = {
        "case": case, "source_image": domain["image"], "source_mask": domain["mask"],
        "native_shape": shape, "crop_shape": crop_shape,
        "crop_zyx": [[s.start, s.stop] for s in crop], "spacing_zyx_um": spacing,
        "crop_selection": "Dense region with both matched (IoU >= 0.5) and unmatched objects; not selected for best score",
        "postprocessing": settings, "full_volume_mean_gt_iou": actual["mean_gt_iou"],
        "crop_gt_ids": [row["gt_id"] for row in rows if row["fully_in_crop"]],
        "crop_gt_ious": {str(row["gt_id"]): row["iou"] for row in rows if row["fully_in_crop"]},
        "checkpoint": str(run / "model/model.pt"),
        "checkpoint_info": json.loads((run / "full-volume-auc/results.json").read_text())["checkpoint_info"],
        "ground_truth_repair": f"{record['extra_components']} disconnected fragments split in RAM; original files unchanged",
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"complete": str(output), "seconds": time.monotonic() - started, "manifest": manifest, "diagnostics": {k: v for k, v in diagnostics.items() if k != "full_volume_metrics"}}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--case", default="blast_022")
    args = parser.parse_args()
    export(args.run_dir, args.case)
