"""Compare exact mask summaries against a saved pre-optimization module.

Run with python -m benchmarks.benchmark_mask_analysis --reference /tmp/baseline.py
--dataset-2d ... --dataset-3d ... --output /tmp/mask-analysis.json.
Only masks are decoded, one at a time. No dataset files are modified.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

from jdll_unet.label_statistics import analyze_mask


def reference_function(path):
    name = "jdll_unet._benchmark_reference"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.analyze_mask


def assert_equal(left, right):
    assert left.shape == right.shape
    assert left.labels == right.labels
    assert left.plane_positive_counts == right.plane_positive_counts
    for a, b in zip((left.objects, *left.planes), (right.objects, *right.planes), strict=True):
        assert list(a) == list(b)
        for label in a:
            assert a[label].count == b[label].count
            assert a[label].bounds == b[label].bounds
    np.testing.assert_array_equal(left.foreground_indices, right.foreground_indices)


def peak_rss_mib():
    status = Path("/proc/self/status")
    if status.exists():
        # ru_maxrss can include the launching process's pre-exec high-water mark.
        for line in status.read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    divisor = 1024**2 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--dataset-2d", type=Path, required=True)
    parser.add_argument("--dataset-3d", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    reference = reference_function(args.reference)
    files_2d = sorted(args.dataset_2d.glob("*_mask.tif"))
    files_3d = sorted(args.dataset_3d.glob("*_masks.tif"))
    if not files_2d or not files_3d:
        parser.error("Both datasets must contain matching TIFF masks")
    files_3d = [files_3d[i] for i in np.linspace(0, len(files_3d) - 1, min(10, len(files_3d)), dtype=int)]
    started = time.perf_counter()
    for dtype in (np.uint8, np.uint16):
        analyze_mask(np.zeros((2, 2), dtype))
    report = {"warmup_seconds": time.perf_counter() - started, "repeats": args.repeats, "datasets": {}}
    for name, files in (("2d", files_2d), ("3d", files_3d)):
        records = []
        for index, path in enumerate(files):
            started = time.perf_counter()
            mask = tifffile.imread(path, maxworkers=1)
            row = {"file": str(path), "shape": list(mask.shape), "dtype": str(mask.dtype),
                   "decode_seconds": time.perf_counter() - started, "baseline": [], "optimized": []}
            for repeat in range(args.repeats):
                functions = [("baseline", reference), ("optimized", analyze_mask)]
                if (index + repeat) % 2:
                    functions.reverse()
                results = []
                for key, function in functions:
                    started = time.perf_counter()
                    result = function(mask, path)
                    row[key].append(time.perf_counter() - started)
                    results.append(result)
                assert_equal(*results)
                del results, result
            row["baseline_median"] = float(np.median(row["baseline"]))
            row["optimized_median"] = float(np.median(row["optimized"]))
            row["reduction_percent"] = 100 * (1 - row["optimized_median"] / row["baseline_median"])
            records.append(row)
            del mask
            if name == "3d" or (index + 1) % 25 == 0:
                print(json.dumps({"dataset": name, "completed": index + 1, **row}), flush=True)
        baseline = sum(row["baseline_median"] for row in records)
        optimized = sum(row["optimized_median"] for row in records)
        report["datasets"][name] = {
            "count": len(files), "baseline_seconds": baseline, "optimized_seconds": optimized,
            "reduction_percent": 100 * (1 - optimized / baseline) if baseline else 0,
            "speedup": baseline / optimized if optimized else 0,
            "decode_seconds": sum(row["decode_seconds"] for row in records), "files": records,
        }
        report["peak_rss_mib"] = peak_rss_mib()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"dataset": name, **{key: value for key, value in report["datasets"][name].items() if key != "files"}}), flush=True)


if __name__ == "__main__":
    main()
