"""Profile a saved 3D run's sample path without repeating annotation preparation."""

from __future__ import annotations

import argparse
import gc
import json
import resource
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import fields
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

from jdll_unet import augment, dataset, targets
from jdll_unet.annotations import AnnotationPreparation, AnnotationRecord
from jdll_unet.geometry import DomainReader, available_host_memory
from jdll_unet.image_reading import image_reading_session
from jdll_unet.io import ImageMaskPair, fit_normalization, normalize_image
from jdll_unet.spatial_augment import collate_spatial_samples
from jdll_unet.trainer import _move_target


class Timings:
    def __init__(self):
        self.rows = defaultdict(lambda: {"seconds": 0.0, "exclusive_seconds": 0.0, "calls": 0})
        self.stack = []

    @contextmanager
    def measure(self, name):
        frame = [perf_counter(), 0.0]
        self.stack.append(frame)
        try:
            yield
        finally:
            elapsed = perf_counter() - frame[0]
            self.stack.pop()
            row = self.rows[name]
            row["seconds"] += elapsed
            row["exclusive_seconds"] += elapsed - frame[1]
            row["calls"] += 1
            if self.stack:
                self.stack[-1][1] += elapsed

    def wrap(self, function, name):
        def measured(*args, **kwargs):
            with self.measure(name):
                return function(*args, **kwargs)
        return measured


def detailed_normalize(image, *, statistics, timings):
    """Match production arithmetic and temporary dtypes, not an optimized variant."""
    assert statistics["type"] == "percentile"
    with timings.measure("normalize.float32_copy"):
        output = image.astype(np.float32, copy=True)
    for channel, (offset, scale) in enumerate(statistics["channels"]):
        with timings.measure("normalize.subtract"):
            difference = output[channel] - offset
        with timings.measure("normalize.divide"):
            normalized = difference / scale
            del difference
        with timings.measure("normalize.clip"):
            clipped = np.clip(normalized, 0, 1)
        with timings.measure("normalize.writeback_float32"):
            output[channel] = clipped
            del clipped, normalized
    return output


def pair_from_saved(value):
    allowed = {field.name for field in fields(ImageMaskPair)}
    value = {key: item for key, item in value.items() if key in allowed}
    for key in ("image", "mask"):
        value[key] = Path(value[key])
    for key in ("spatial_shape", "plane_positive_counts", "label_values", "eligible_centers"):
        if value.get(key) is not None:
            value[key] = tuple(value[key])
    value["region"] = tuple(tuple(bounds) for bounds in value.get("region", ()))
    return ImageMaskPair(**value)


def trusted_preparation(plan, pair):
    record = next(row for row in plan["annotation_preparation"]["sources"] if Path(row["path"]) == pair.mask.resolve())
    if record["status"] != "unchanged" or record["original_ids"] != len(pair.label_values):
        raise ValueError("This profiler requires an unchanged source verified by the saved run")
    preparation = AnnotationPreparation()
    preparation.records[preparation._key(pair, "3d")] = AnnotationRecord(
        str(pair.mask.resolve()), dict.fromkeys(pair.label_values, 1)
    )
    preparation.instance_mode = True
    return preparation


def profile_samples(plan, config, case, repeats, use_cuda):
    pair = pair_from_saved(next(row for row in plan["training_domains"] if row["stem"] == case))
    if np.prod(pair.spatial_shape) > 110_000_000:
        raise ValueError("Full preprocessing is limited to small volumes to avoid host memory pressure")
    cfg = config["training"]["augmentation"]
    augmentation = augment.make_augmentation_config(
        cfg["profile"], tuple(cfg["patch_size"]), cfg["foreground_oversampling"], cfg["foreground_probability"], cfg
    )
    spacing = tuple(plan["spacing"]["target_spacing"])
    case_spacing = tuple(next(row["spacing"] for row in plan["spacing"]["cases"] if row["case"] == case))
    results = []
    with image_reading_session() as session:
        session.domain_reader = DomainReader(max_bytes=512 * 1024**2, session=session)
        session.annotations = trusted_preparation(plan, pair)
        data = dataset.JdllSegmentationDataset(
            [pair], "instance_friendly", None, config["normalization"], augmentation, True,
            dimensions="3d", seed=config["training"]["seed"],
            instance_sizes={case: plan["training_instance_sizes"][case]},
            case_spacings={case: case_spacing}, target_spacing=spacing, sample_count=repeats * 3,
            defer_photometric=True, defer_spatial=True,
        )
        for phase, count in (("first_source_load", 1), ("cached_source", repeats), ("uncached_source_cached_statistics", repeats)):
            for index in range(count):
                available = available_host_memory()
                if available is not None and available < 4 * 1024**3:
                    raise MemoryError("Less than 4 GiB available; stopping before full-volume preprocessing")
                if phase == "uncached_source_cached_statistics":
                    session.domain_reader.cache.clear()
                    session.domain_reader.bytes = 0
                timings = Timings()
                hooks = (
                    (dataset, "load_domain_image", "image_read"),
                    (dataset, "load_domain_mask", "mask_read"),
                    (dataset, "fit_normalization", "fit_percentiles"),
                    (dataset, "resample_image_mask", "spacing_resample"),
                    (dataset, "apply_augmentation", "cpu_augmentation"),
                    (augment, "sample_patch", "crop_selection"),
                    (augment, "_spatial_affine", "cpu_affine_labels"),
                    (dataset, "prepare_target", "cpu_targets"),
                    (targets, "compact_instance_labels", "compact_patch_ids"),
                    (targets, "normalized_instance_distance", "patch_distance_targets"),
                    (dataset.JdllSegmentationDataset, "_load_item", "load_item"),
                    (dataset.JdllSegmentationDataset, "_normalization_stats", "statistics_lookup_or_fit"),
                )
                with ExitStack() as stack:
                    for module, name, label in hooks:
                        stack.enter_context(patch.object(module, name, timings.wrap(getattr(module, name), label)))

                    def normalization(image, *, statistics, timings=timings):
                        with timings.measure("normalize_volume"):
                            return detailed_normalize(image, statistics=statistics, timings=timings)

                    stack.enter_context(patch.object(dataset, "normalize_image", normalization))
                    with timings.measure("cpu_sample_total"):
                        sample = data[index]
                if use_cuda:
                    device = torch.device("cuda")
                    with timings.measure("collate_and_pin"):
                        images, target = collate_spatial_samples([sample])
                        images.pin_memory()
                        target = {key: value.pin_memory() for key, value in target.items()}
                    torch.cuda.synchronize()
                    with timings.measure("gpu_image_transfer_and_spatial"):
                        images = images.materialize(device)
                        torch.cuda.synchronize()
                    with timings.measure("gpu_target_transfer"):
                        target = _move_target(target, device)
                        torch.cuda.synchronize()
                    with timings.measure("gpu_foreground_boundary_targets"):
                        target = targets.complete_device_targets("instance_friendly", target, None)
                        torch.cuda.synchronize()
                    with timings.measure("gpu_photometric"):
                        images = augment.apply_tensor_photometric_augmentation(images, target["valid"], augmentation)
                        torch.cuda.synchronize()
                    del images, target
                results.append({"phase": phase, "index": index, "timings": dict(timings.rows)})
                print(json.dumps({"case": case, **results[-1]}), flush=True)
                del sample
                gc.collect()
    return results


def profile_reads(plan, case, repeats):
    pair = pair_from_saved(next(row for row in plan["training_domains"] if row["stem"] == case))
    rows = []
    with image_reading_session() as session:
        reader = DomainReader(max_bytes=512 * 1024**2, session=session)
        session.annotations = trusted_preparation(plan, pair)
        for index in range(repeats):
            timings = Timings()
            with timings.measure("image_read"):
                image = dataset.load_domain_image(pair, "3d", reader, raw=True)
            with timings.measure("mask_read"):
                mask = dataset.load_domain_mask(pair, "3d", reader, raw=True)
            rows.append({"index": index, "image_bytes": image.nbytes, "mask_bytes": mask.nbytes, "timings": dict(timings.rows)})
            print(json.dumps({"case": case, **rows[-1]}), flush=True)
            del image, mask
            gc.collect()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--case", default="blast_099")
    parser.add_argument("--read-case", default="blast_008")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(14)
    torch.manual_seed(42)
    plan = json.loads((args.run / "model/dataset_plan.json").read_text())
    config = json.loads((args.run / "model/config.json").read_text())
    probe = np.arange(48, dtype=np.uint16).reshape(1, 3, 4, 4)
    statistics = fit_normalization(probe, config["normalization"])
    np.testing.assert_array_equal(
        detailed_normalize(probe, statistics=statistics, timings=Timings()),
        normalize_image(probe, statistics=statistics),
    )
    if args.cuda:
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
    report = {
        "run": str(args.run), "case": args.case, "read_case": args.read_case,
        "numpy_version": np.__version__, "torch_threads": torch.get_num_threads(),
        "normalization_offset_type": type(statistics["channels"][0][0]).__name__,
        "normalization_subtraction_dtype": str((probe.astype(np.float32) - statistics["channels"][0][0]).dtype),
        "notes": [
            "Production settings; saved unchanged-annotation checks reused, no annotation preparation repeated.",
            "First load means application cache miss, not a deliberately flushed OS disk cache.",
            "Normalization substeps mirror original expression; exact equality checked on a small array.",
            "Inclusive timings overlap; exclusive_seconds removes instrumented children.",
            "CUDA preprocessing timings synchronize stages and exclude model execution.",
            "Large case: read-only profile; full normalization deliberately not run under host memory pressure.",
        ],
    }
    report["samples"] = profile_samples(plan, config, args.case, args.repeats, args.cuda)
    report["large_source_reads"] = profile_reads(plan, args.read_case, args.repeats)
    report["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}; peak RSS {report['peak_rss_gib']:.3f} GiB", flush=True)


if __name__ == "__main__":
    main()
