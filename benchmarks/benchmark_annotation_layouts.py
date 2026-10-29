"""Compare the pre-NumPy annotation helper with the current bounded implementation.

Linux fork isolates peak RSS per repetition without repeatedly importing/JITing
the libraries. Source generation, JIT compilation and checksums are not timed.
"""

# Numerical imports must follow the thread-limit setup.
# ruff: noqa: E402

import argparse
import gc
import hashlib
import json
import multiprocessing as mp
import os
import random
import resource
import statistics
import sys
import time
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[name] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numba
import numpy as np
import skimage
from skimage.measure import label as label_components

from jdll_unet.annotations import _fill_source_ids, component_labels_and_sources


def previous(mask):
    components, count = label_components(mask, background=0, connectivity=1, return_num=True)
    source_ids = np.zeros(count + 1, dtype=np.int64)
    _fill_source_ids(components.ravel(), mask.ravel(), source_ids)
    return components, source_ids


def grid(shape):
    mask = np.zeros(shape, np.uint16)
    counter = 0
    depth = shape[0] if len(shape) == 3 else 1
    for z in range(0, depth, 8):
        for y in range(0, shape[-2], 32):
            for x in range(0, shape[-1], 32):
                counter += 1
                region = (slice(y + 8, y + 24), slice(x + 8, x + 24))
                mask[((slice(z + 2, z + 6), *region) if len(shape) == 3 else region)] = counter % 5000 + 1
    return mask


def make_case(name):
    shape = (2048, 2048) if name == "2d_contiguous_small" else (4096, 4096) if name in (
        "2d_contiguous", "2d_rgb_pipeline") else (64, 512, 512)
    mask = grid(shape)
    if name == "3d_strided":
        source = np.zeros((*shape, 3), np.uint16)
        source[..., 0] = mask
        return source[..., 0]
    if name == "3d_transposed":
        return np.ascontiguousarray(mask.transpose(1, 2, 0)).transpose(2, 0, 1)
    if name == "3d_big_endian":
        return mask.astype(">u4")
    if name == "2d_rgb_pipeline":
        return np.repeat(mask[..., None], 3, axis=-1)
    if name == "3d_planar_pipeline":
        return np.stack([mask, mask, mask])
    return mask


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def worker(connection, source, case, method):
    try:
        if method == "native_reference":
            source = source.astype(source.dtype.newbyteorder("="))
        gc.collect()
        initial_peak = peak_mb()
        cpu_start, start = time.process_time(), time.perf_counter()
        mask = source
        if case == "2d_rgb_pipeline":
            mask = np.take(source, 0, axis=-1) if method == "previous" else source[..., 0]
        elif case == "3d_planar_pipeline":
            mask = np.take(source, 0, axis=0) if method == "previous" else source[0]
        result = (previous if method in ("previous", "native_reference") else component_labels_and_sources)(mask)
        seconds, cpu_seconds = time.perf_counter() - start, time.process_time() - cpu_start
        peak = peak_mb()
        checksum = hashlib.sha256()
        for array in result:
            checksum.update(memoryview(np.ascontiguousarray(array)).cast("B"))
        connection.send({"seconds": seconds, "cpu_seconds": cpu_seconds, "peak_mb": peak,
                         "extra_peak_mb": peak - initial_peak, "digest": checksum.hexdigest()})
    except Exception as error:
        connection.send({"error": type(error).__name__ + ": " + str(error).splitlines()[0]})
    finally:
        connection.close()


def run(source, case, method):
    context = mp.get_context("fork")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=worker, args=(writer, source, case, method))
    process.start()
    writer.close()
    if not reader.poll(120):
        process.terminate()
        process.join()
        raise RuntimeError("Annotation benchmark timed out")
    result = reader.recv()
    reader.close()
    process.join()
    if process.exitcode:
        raise RuntimeError(f"Benchmark child exited with {process.exitcode}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    small = grid((32, 32))
    previous(small)
    component_labels_and_sources(small)
    component_labels_and_sources(np.repeat(small[..., None], 3, axis=-1)[..., 0])
    component_labels_and_sources(np.asfortranarray(small))
    component_labels_and_sources(np.zeros((2, 32, 32), np.uint16).transpose(2, 1, 0))
    component_labels_and_sources(small.astype(">u4"))
    report = {"python": sys.version.split()[0], "numpy": np.__version__, "skimage": skimage.__version__,
              "numba": numba.__version__, "repeats": args.repeats, "cases": {}}
    cases = ("2d_contiguous_small", "2d_contiguous", "3d_contiguous", "3d_strided", "3d_transposed",
             "2d_rgb_pipeline", "3d_planar_pipeline", "3d_big_endian")
    for case in cases:
        source = make_case(case)
        # Warm-touch input pages before forking; no source reads are charged as file I/O.
        source.sum(dtype=np.uint64)
        rows = {"previous": [], "current": []}
        expected = None
        if case == "3d_big_endian":
            unsupported = run(source, case, "previous")
            assert "error" in unsupported, "The historical implementation unexpectedly accepted big endian"
            reference = run(source, case, "native_reference")
            assert "error" not in reference, reference
            expected = reference["digest"]
        for repeat in range(args.repeats):
            methods = ["current"] if expected is not None else ["previous", "current"]
            random.Random(repeat + 73).shuffle(methods)
            for method in methods:
                value = run(source, case, method)
                assert "error" not in value, value
                rows[method].append(value)
        digests = {row["digest"] for values in rows.values() for row in values}
        assert len(digests) == 1 and (expected is None or expected in digests), "Changed annotation results"
        summary = {method: {key: statistics.median(row[key] for row in values)
                            for key in ("seconds", "cpu_seconds", "peak_mb", "extra_peak_mb")}
                   for method, values in rows.items() if values}
        report["cases"][case] = {"shape": source.shape, "strides": source.strides, "dtype": str(source.dtype),
                                 "results": summary, "trials": rows, "identical_results": True}
        if expected is not None:
            report["cases"][case]["previous_error"] = unsupported["error"]
        print(case, json.dumps(summary), flush=True)
        del source
        gc.collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
