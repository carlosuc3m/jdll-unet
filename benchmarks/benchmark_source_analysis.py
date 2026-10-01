"""Bounded source-startup timings; optional reference of the former 3D size loop.

Run with PYTHONPATH=. and OMP/MKL/OPENBLAS_NUM_THREADS=1. Inputs are untouched.
This harness expects Blast3D's <case>_image.tif / <case>_masks.tif naming.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import signal
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile
import torch

from jdll_unet import annotations, geometry
from jdll_unet.annotations import AnnotationPreparation
from jdll_unet.config import parse_training_config
from jdll_unet.image_reading import image_reading_session
from jdll_unet.io import ImageMaskPair
from jdll_unet.planning import build_dataset_plan
from jdll_unet.task_detect import detect_task_from_pairs
from jdll_unet.training_geometry import measure_case_instances


def legacy_size(pair, spacing, seed):
    """Former default estimator: two unique scans, all coordinates/axes, then sample.

    Uses the current loader/validator, so this is an algorithm comparison, not
    a checkout of the old library. No connectivity is repeated here.
    """
    mask = geometry.load_domain_mask(pair, "3d")
    original_labels = int(np.count_nonzero(np.unique(mask)))
    complete, border = [], []
    shape = np.asarray(mask.shape)
    voxel_volume = float(np.prod(spacing))
    for label in (int(value) for value in np.unique(mask) if int(value) != 0):
        coords = np.argwhere(mask == label)
        if len(coords) < 4:
            continue
        touches = bool(np.any(coords.min(axis=0) == 0) or np.any(coords.max(axis=0) == shape - 1))
        physical = coords.astype(np.float64) * np.asarray(spacing)
        axes = tuple(float(value) for value in 2 * np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov(physical.T)), 0)))
        diameter = 2 * (3 * len(coords) * voxel_volume / (4 * math.pi)) ** (1 / 3)
        (border if touches else complete).append((diameter, axes))
    selected = []
    rng = np.random.default_rng(seed)
    for candidates in (complete, border):
        remaining = 21 - len(selected)
        if remaining <= 0:
            break
        indexes = np.arange(len(candidates)) if len(candidates) <= remaining else rng.choice(len(candidates), remaining, replace=False)
        selected.extend(candidates[int(index)] for index in indexes)
    return {
        "original_labels": original_labels,
        "sampled_instances": len(selected),
        "median_diameter": float(np.median([diameter for diameter, _ in selected])) if selected else None,
    }


def status_bytes(name):
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(name + ":"):
            return int(line.split()[1]) * 1024
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("case")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--inspection-only", action="store_true")
    parser.add_argument("--stage-seconds", type=float, default=90)
    parser.add_argument("--total-seconds", type=float, default=900)
    args = parser.parse_args()
    torch.set_num_threads(1)
    available = geometry.available_host_memory()
    if available is None or available < 3 * 1024**3:
        raise RuntimeError("At least 3 GiB available RAM is required for this bounded sample")
    extra = min(4 * 1024**3, available - 2 * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (status_bytes("VmSize") + extra, resource.RLIM_INFINITY))

    def timeout(_signum, _frame):
        raise TimeoutError("Benchmark time budget reached")

    signal.signal(signal.SIGALRM, timeout)
    result = {
        "case": args.case, "python": sys.executable, "load_average": os.getloadavg(),
        "available_ram_gib": available / 1024**3, "extra_address_space_gib": extra / 1024**3,
        "stages": {}, "status": "running", "statistics_calls": 0,
    }
    started = time.perf_counter()
    original_analysis = annotations.analyze_mask
    metrics = {"analysis_seconds": 0.0, "analysis_calls": 0}

    def analyze(*a, **kw):
        before = time.perf_counter()
        try:
            return original_analysis(*a, **kw)
        finally:
            metrics["analysis_seconds"] += time.perf_counter() - before
            metrics["analysis_calls"] += 1

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    def stage(name, operation):
        remaining = args.total_seconds - (time.perf_counter() - started)
        if remaining <= 0:
            raise TimeoutError("Total time budget reached")
        print(json.dumps({"starting": name, "case": args.case}), flush=True)
        before = time.perf_counter()
        previous = metrics.copy()
        signal.setitimer(signal.ITIMER_REAL, min(args.stage_seconds, remaining))
        try:
            return operation()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            result["stages"][name] = {
                "seconds": time.perf_counter() - before,
                "rss_mib_after": status_bytes("VmRSS") / 1024**2,
                "included_analysis_seconds": metrics["analysis_seconds"] - previous["analysis_seconds"],
                "analysis_calls": metrics["analysis_calls"] - previous["analysis_calls"],
            }
            save()
            print(json.dumps({"finished": name, **result["stages"][name]}), flush=True)

    pair = ImageMaskPair(args.dataset / f"{args.case}_image.tif", args.dataset / f"{args.case}_masks.tif", args.case)
    with tifffile.TiffFile(pair.mask) as handle:
        result["shape"] = list(handle.series[0].shape)
        voxels = int(handle.series[0].size)
    try:
        with tempfile.TemporaryDirectory(prefix="jdll-source-analysis-") as temporary, image_reading_session() as session:
            session.domain_reader = geometry.DomainReader(geometry.resolve_data_cache_bytes("auto"), session=session)
            session.annotations = prep = AnnotationPreparation(cache_dir=Path(temporary))
            result["data_cache_mib"] = session.domain_reader.max_bytes / 1024**2
            with patch.object(annotations, "analyze_mask", analyze):
                pair, _ = stage("inspection", lambda: geometry.inspect_pair(pair))
                result["label_count"] = len(pair.label_values)
                if args.inspection_only:
                    result["status"] = "inspection_complete"
                    return
                if voxels * 32 > geometry.available_host_memory() - 1024**3:
                    result["status"] = "stopped_before_connectivity: insufficient memory headroom"
                    return
                detection = stage("task_detection", lambda: detect_task_from_pairs(
                    [pair], dataset_path=args.dataset, dimensions="3d", preparation=prep,
                ))
                result["task"] = detection["task"]
                if detection["task"] != "instance_friendly":
                    result["status"] = "complete_non_instance"
                    return
                stage("instance_preparation_reuse", lambda: prep.prepare([pair], "3d", lambda: None))
                plan = stage("spacing_plan", lambda: build_dataset_plan([pair], "3d"))
                spacing = plan.cases[0].spacing
                cfg = parse_training_config({
                    "model_name": "timing-only", "dataset_path": args.dataset, "output_dir": temporary,
                    "architecture": "resenc-tiny-3d", "task": "instance_friendly",
                })
                estimate, _ = stage("instance_size_estimation", lambda: measure_case_instances(pair, cfg, "3d", spacing))
                result["estimate"] = {
                    "median_diameter": estimate.median_diameter_px,
                    "sampled_instances": estimate.sampled_instances,
                    "available_instances": estimate.available_instances,
                } if estimate else None
                if args.legacy:
                    seed = geometry.stable_seed(cfg.seed, 0, str(pair.image.resolve()) + str(pair.region))
                    reference = stage("legacy_size_estimation", lambda: legacy_size(pair, spacing, seed))
                    result["legacy_estimate"] = reference
                    if estimate is not None:
                        np.testing.assert_allclose(estimate.median_diameter_px, reference["median_diameter"], rtol=1e-12)
                        assert estimate.sampled_instances == reference["sampled_instances"]
                result["status"] = "complete"
    except (MemoryError, TimeoutError) as exc:
        result["status"] = f"stopped: {type(exc).__name__}: {exc}"
    finally:
        result["statistics_calls"] = metrics["analysis_calls"]
        result["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        save()
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
