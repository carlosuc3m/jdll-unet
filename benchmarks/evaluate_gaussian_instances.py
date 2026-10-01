"""Bounded-memory, full-native-volume instance evaluation from prepared assets."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import mmap
import resource
import shutil
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

from jdll_unet.config import write_json
from jdll_unet.crop_reading import CropArray
from jdll_unet.geometry import DomainReader
from jdll_unet.image_reading import image_reading_session
from jdll_unet.infer import _sigmoid, _tile_layout, load_model, tiled_predict

from .evaluate_instance_iou import InstanceOverlap, aggregate, native_region, reconstruct
from .prepared_run import load_analysis
from .profile_sample_preprocessing import pair_from_saved


def release_mapping(array):
    mapping = getattr(array, "_mmap", None)
    if mapping is not None and hasattr(mapping, "madvise"):
        mapping.madvise(mmap.MADV_DONTNEED)


def scaled_image(source, shape, emit):
    """Match align_corners=False trilinear resizing without a native float volume."""
    result = np.empty((source.shape[0], *shape), np.float32)
    planes = {}
    for z in range(shape[0]):
        coordinate = float(np.clip((z + 0.5) * source.shape[1] / shape[0] - 0.5, 0, source.shape[1] - 1))
        lower, upper = int(np.floor(coordinate)), int(np.ceil(coordinate))
        planes = {key: value for key, value in planes.items() if key in (lower, upper)}
        for index in (lower, upper):
            if index not in planes:
                plane = source[:, index:index + 1, :, :][:, 0]
                planes[index] = F.interpolate(torch.from_numpy(plane[None]), size=shape[1:],
                                              mode="bilinear", align_corners=False)[0].numpy()
        fraction = np.float32(coordinate - lower)
        result[:, z] = planes[lower] * (1 - fraction) + planes[upper] * fraction
        emit("resampling", current=z + 1, total=shape[0])
    return result


def foreground_regions(logits, shape, minimum, block=(4, 16, 16)):
    """Coarse occupancy only groups exact native foreground; it never samples it."""
    grid = tuple((n + b - 1) // b for n, b in zip(shape, block, strict=True))
    occupied = np.zeros(grid, bool)
    for z in range(0, shape[0], block[0]):
        box = (slice(z, min(z + block[0], shape[0])), slice(0, shape[1]), slice(0, shape[2]))
        foreground = _sigmoid(native_region(logits[0], shape, box)) >= minimum
        plane = foreground.any(axis=0)
        padded = np.pad(plane, ((0, grid[1] * block[1] - shape[1]), (0, grid[2] * block[2] - shape[2])))
        occupied[z // block[0]] = padded.reshape(grid[1], block[1], grid[2], block[2]).any(axis=(1, 3))
    components, count = ndi.label(occupied, structure=np.ones((3, 3, 3), bool))
    boxes = []
    for tight in ndi.find_objects(components, max_label=count):
        boxes.append(tuple(slice(max(0, s.start * b - 1), min(n, s.stop * b + 1))
                           for s, n, b in zip(tight, shape, block, strict=True)))
    return components, boxes


def region_probabilities(logits, shape, box, components, region_id, block=(4, 16, 16)):
    indices = [np.arange(s.start, s.stop) // b for s, b in zip(box, block, strict=True)]
    selected = components[np.ix_(*indices)] == region_id
    probabilities = [_sigmoid(native_region(channel, shape, box)) for channel in logits]
    probabilities[0][~selected] = 0
    return probabilities


def setting_key(settings):
    return json.dumps(settings, sort_keys=True, separators=(",", ":"))


def evaluate(run, prepared_run, output, example, overlap):
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    last_emit = 0.0
    case_name = None

    def emit(phase, **values):
        nonlocal last_emit
        now = time.monotonic()
        if phase in {"resampling", "inference", "reconstructing"} and now - last_emit < 15:
            return
        last_emit = now
        event = {"phase": phase, "case": case_name, "seconds": now - started,
                 "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, **values}
        print(json.dumps(event), flush=True)
        write_json(output / "state.json", event)
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")

    config = json.loads((run / "model/config.json").read_text())
    plan = json.loads((run / "model/dataset_plan.json").read_text())
    metadata = json.loads((run / "model/model_metadata.json").read_text())
    cases = plan["validation_domains"]
    if example not in {case["stem"] for case in cases}:
        raise ValueError(f"Validation case not found: {example}")
    target_spacing = metadata["dataset_plan"]["target_spacing"]
    spacings = {row["case"]: row["spacing"] for row in plan["spacing"]["cases"]}
    cache = prepared_run / "prepared_cache"
    assets = {}
    for domain in cases:
        pair = pair_from_saved(domain)
        if pair.image_axes != "ZYX" or pair.mask_axes != "ZYX" or pair.domain_shape != pair.spatial_shape:
            raise ValueError("Expected complete single-channel ZYX validation volumes")
        if not np.allclose(spacings[pair.stem], target_spacing):
            raise ValueError("This evaluation requires native spacing equal to planned target spacing")
        base = cache / hashlib.sha256(str(pair.mask.resolve()).encode()).hexdigest()[:20]
        asset = json.loads(base.with_suffix(".json").read_text())
        for name, expected in asset["fingerprint"]["files"].items():
            stat = Path(name).stat()
            if [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns] != expected:
                raise ValueError(f"Source changed since preparation: {name}")
        analysis = load_analysis(base.with_suffix(".npz"))
        sizes = np.zeros(max(analysis.objects, default=0) + 1, np.int64)
        for label, region in analysis.objects.items():
            sizes[label] = region.count
        if int(sizes.sum()) != sum(domain["plane_positive_counts"]):
            raise ValueError("Prepared mask foreground differs from validation metadata")
        assets[pair.stem] = (pair, base, asset, sizes)
    checkpoint = output / "checkpoint.pt"
    if not checkpoint.exists():
        shutil.copyfile(run / "model/weights_best.pt", checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_epoch = state["epoch"]
    if state["model_config"]["training"]["seed"] != config["training"]["seed"]:
        raise ValueError("Cached checkpoint differs from the requested run")
    del state
    identity = {"source_run": str(run.resolve()), "checkpoint_epoch": checkpoint_epoch,
                "tile_overlap": overlap, "tile_blending": "gaussian", "gaussian_sigma_scale": 0.125,
                "min_seed_size": 1, "example": example,
                "source_fingerprints": {k: a[2]["fingerprint"] for k, a in assets.items()}}
    request = output / "evaluation_config.json"
    if request.exists() and json.loads(request.read_text()) != identity:
        raise ValueError("Existing evaluation belongs to different inputs or settings")
    write_json(request, identity)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    model, _ = load_model(checkpoint, device)
    base_settings = {**config["postprocessing"], "min_seed_size": 1, "min_seed_size_physical": None}
    results_path = output / "results.json"
    report = json.loads(results_path.read_text()) if results_path.exists() else {
        "checkpoint": str(checkpoint), "checkpoint_epoch": checkpoint_epoch, "results": {},
        "objective": "Pooled mean one-to-one maximum-assignment IoU over every GT instance; misses score zero",
        "caveat": "Thresholds fitted on validation; not an independent test score. Unmatched predictions reported separately.",
        "inference": identity,
    }

    class Progress:
        def __init__(self, total):
            self.total, self.completed = total, 0

        def patch_start(self, starts):
            pass

        def patch_end(self):
            self.completed += 1
            emit("inference", current=self.completed, total=self.total)

    with image_reading_session() as session:
        reader = DomainReader(max_bytes=32 * 1024**2, session=session)
        for case_name, (pair, _, asset, _) in assets.items():
            path = output / f"{case_name}_logits.npy"
            if path.exists():
                emit("inference_cached")
                continue
            scale_cfg = config["training"]["instance_scale_normalization"]
            scale = float(np.clip(metadata["instance_scale"]["target_object_size"] / plan["validation_instance_sizes"][case_name],
                                  scale_cfg["min_effective_scale"], scale_cfg["max_effective_scale"]))
            shape = tuple(max(1, round(n * scale)) for n in pair.spatial_shape)
            emit("case_start", native_shape=pair.spatial_shape, scaled_shape=shape, scale=scale)
            image = scaled_image(CropArray(pair, reader, statistics=asset["normalization"]), shape, emit)
            tile = tuple(config["training"]["patch_size"])
            layout = _tile_layout(shape, tile, overlap)
            emit("inference_start", patches=layout.count)
            logits = tiled_predict(model, image, device, tile, overlap, blend_mode="gaussian",
                                   progress=Progress(layout.count))
            if not np.isfinite(logits).all():
                raise ValueError("Non-finite inference logits")
            temporary = path.with_suffix(".tmp.npy")
            np.save(temporary, logits)
            temporary.replace(path)
            emit("inference_complete", bytes=path.stat().st_size)
            del image, logits
            gc.collect()
            torch.cuda.empty_cache()

        # Every native foreground voxel at any tested threshold belongs to a region.
        # The grouping includes background-only predictions, never just GT boxes.
        def sweep(settings, stage):
            nonlocal case_name
            for case_name, (pair, base, asset, sizes) in assets.items():
                missing = [s for s in settings if case_name not in report["results"].get(setting_key(s), {}).get("cases", {})]
                if not missing:
                    continue
                logits = np.load(output / f"{case_name}_logits.npy", mmap_mode="r")
                region_path = output / f"{case_name}_regions.npz"
                if region_path.exists():
                    with np.load(region_path) as data:
                        components = data["components"]
                        boxes = [tuple(slice(int(a), int(b)) for a, b in row) for row in data["boxes"]]
                else:
                    emit("finding_native_regions")
                    components, boxes = foreground_regions(logits, pair.spatial_shape, 0.2)
                    np.savez_compressed(region_path, components=components,
                                        boxes=np.array([[(s.start, s.stop) for s in box] for box in boxes], np.int32))
                truth = np.load(base.with_suffix(".npy"), mmap_mode="r") if asset["repaired"] else CropArray(pair, reader, mask=True, original_mask=True)
                accumulators = [InstanceOverlap(sizes) for _ in missing]
                emit("sweep_start", stage=stage, settings=len(missing), regions=len(boxes), gt_instances=int((sizes > 0).sum()),
                     largest_region_voxels=max((int(np.prod([s.stop - s.start for s in b])) for b in boxes), default=0))
                for region_id, box in enumerate(boxes, start=1):
                    probabilities = region_probabilities(logits, pair.spatial_shape, box, components, region_id)
                    reference = np.array(truth[box], copy=True)
                    highest = float(probabilities[0].max())
                    for setting, accumulator in zip(missing, accumulators, strict=True):
                        if highest < setting["threshold"]:
                            continue
                        labels = reconstruct(*probabilities, {**setting, "spacing": spacings[case_name]})
                        accumulator.update(labels, reference)
                        del labels
                    del probabilities, reference
                    release_mapping(truth)
                    emit("reconstructing", stage=stage, current=region_id, total=len(boxes))
                for setting, accumulator in zip(missing, accumulators, strict=True):
                    entry = report["results"].setdefault(setting_key(setting), {"settings": setting, "cases": {}})
                    entry["cases"][case_name] = accumulator.metrics()
                    entry["aggregate"] = aggregate(list(entry["cases"].values()))
                write_json(results_path, report)
                emit("sweep_case_complete", stage=stage)
                del logits, truth, accumulators, components
                gc.collect()

        coarse = [{**base_settings, "threshold": t} for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99)]
        sweep(coarse, "foreground")
        ranking = sorted((report["results"][setting_key(s)] for s in coarse),
                         key=lambda r: r["aggregate"]["pooled_mean_gt_iou"], reverse=True)
        thresholds = [r["settings"]["threshold"] for r in ranking[:3]]
        emit("foreground_search_complete", best=ranking[0]["settings"], scores=ranking[0]["aggregate"])
        refined = [{**base_settings, "threshold": t, "seed_distance_threshold": d,
                    "seed_boundary_threshold": b, "seed_h": h}
                   for t, d, b, h in product(thresholds, (0.1, 0.2, 0.35, 0.5), (0.3, 0.5, 0.7), (0.05, 0.1, 0.2))]
        sweep(refined, "seeds")
        complete = [r for r in report["results"].values() if len(r["cases"]) == len(assets)]
        best = max(complete, key=lambda r: r["aggregate"]["pooled_mean_gt_iou"])
        best_case = max(complete, key=lambda r: r["cases"][example]["mean_gt_iou"])
        report["best_shared"] = best
        report["best_example"] = best_case
        write_json(output / "recommended_postprocessing.json", {"postprocessing": best["settings"],
                   "tile_overlap": overlap, "tile_blending": "gaussian"})

        case_name = example
        pair, base, asset, sizes = assets[example]
        logits = np.load(output / f"{example}_logits.npy", mmap_mode="r")
        truth = np.load(base.with_suffix(".npy"), mmap_mode="r") if asset["repaired"] else CropArray(pair, reader, mask=True, original_mask=True)
        with np.load(output / f"{example}_regions.npz") as data:
            components = data["components"]
            boxes = [tuple(slice(int(a), int(b)) for a, b in row) for row in data["boxes"]]
        exports = []
        for name, selected in (("best_shared", best), ("best_iou", best_case)):
            expected = selected["cases"][example]
            dtype = np.uint16 if expected["predicted_instances"] <= 65535 else np.uint32
            path = output / f"{example}_{name}_prediction.npy"
            temporary = path.with_suffix(".tmp.npy")
            prediction = np.lib.format.open_memmap(temporary, mode="w+", dtype=dtype, shape=pair.spatial_shape)
            accumulator, offset = InstanceOverlap(sizes), 0
            emit("export_start", file=str(path), dtype=str(np.dtype(dtype)))
            for region_id, box in enumerate(boxes, start=1):
                probabilities = region_probabilities(logits, pair.spatial_shape, box, components, region_id)
                labels = reconstruct(*probabilities, {**selected["settings"], "spacing": spacings[example]})
                positive = labels > 0
                if int(labels.max()) + offset > np.iinfo(dtype).max:
                    raise ValueError("Prediction IDs exceed selected storage dtype")
                prediction[box][positive] = labels[positive] + offset
                accumulator.update(labels, truth[box])
                offset += int(labels.max())
                prediction.flush()
                release_mapping(prediction)
                release_mapping(truth)
                emit("reconstructing", stage="export", current=region_id, total=len(boxes))
            actual = accumulator.metrics()
            if offset != expected["predicted_instances"] or not np.isclose(actual["mean_gt_iou"], expected["mean_gt_iou"], atol=1e-10):
                raise ValueError("Export does not reproduce threshold-sweep result")
            prediction.flush()
            del prediction
            temporary.replace(path)
            exports.append({"path": str(path), "shape": list(pair.spatial_shape), "axes": "ZYX", "dtype": str(np.dtype(dtype)),
                            "spacing": spacings[example], "settings": selected["settings"], "metrics": actual,
                            "source_image": str(pair.image), "source_mask": str(pair.mask)})
        report.update(exports=exports, complete=True, seconds=time.monotonic() - started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        write_json(results_path, report)
        emit("complete", scores=best["aggregate"], example_iou=best_case["cases"][example]["mean_gt_iou"],
             output=str(output), settings_tested=len(complete))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prepared-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--example", default="blast_022")
    parser.add_argument("--overlap", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.overlap < 1:
        parser.error("Overlap must be in [0, 1)")
    evaluate(args.run, args.prepared_run, args.output, args.example, args.overlap)
