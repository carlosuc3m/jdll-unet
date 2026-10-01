"""Profile the real training loop without changing production execution.

Run from the repository: python -m benchmarks.profile_training --help
CUDA event spans include host dispatch gaps; CPU subtimings overlap their parent
stages. Only the uninstrumented mode should be used for throughput comparisons.
"""

from __future__ import annotations

import argparse
import cProfile
import csv
import functools
import io
import json
import os
import pstats
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import ExitStack, contextmanager, nullcontext, suppress
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from jdll_unet import augment, crop_reading, dataset, targets, trainer
from jdll_unet.crop_sampling import CropSampler
from jdll_unet.image_reading import ImageReadSession
from jdll_unet.schedulers import LearningRateScheduler
from jdll_unet.spatial_augment import SpatialImageBatch


class BenchmarkComplete(BaseException):
    """Stop before validation/checkpointing, still unwinding training cleanup."""


def peak_process_rss_mb() -> float | None:
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024**2 if sys.platform == "darwin" else 1024)


@contextmanager
def monitor_vram(output: Path, enabled: bool):
    """Sample this process's driver-visible VRAM without querying in the step loop."""
    result = {"sampled_process_peak_mb": None, "vram_samples": 0}
    if not enabled:
        yield result
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".vram.csv").open("w+") as stream:
        process = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
                "--loop-ms=200",
            ],
            stdout=stream,
            stderr=subprocess.DEVNULL,
        )
        try:
            yield result
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            stream.seek(0)
            samples = []
            for row in csv.reader(stream):
                if len(row) == 2 and row[0].strip() == str(os.getpid()):
                    with suppress(ValueError):
                        samples.append(float(row[1]))
            result.update(sampled_process_peak_mb=max(samples, default=None), vram_samples=len(samples))


def run(args: argparse.Namespace) -> dict:
    using_cuda = args.device == "cuda"
    saved = json.loads(args.config.read_text())
    request = dict(saved.get("training", saved))
    request.update(
        model_name=f"{args.device}-profile",
        starting_point="scratch",
        base_model=None,
        device=args.device,
        epochs=1,
        steps_per_epoch=args.warmup + args.steps,
        num_workers=0,
        preview_count=0,
        progress_update_interval=args.report_every,
        log_update_interval=args.report_every,
    )
    if args.dataset is not None:
        request["dataset_path"] = str(args.dataset)
    if args.batch_size is not None:
        request.update(batch_size=args.batch_size, effective_batch_size=args.batch_size)
    if args.cache_mb is not None:
        request["data_cache_mb"] = args.cache_mb
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    if using_cuda and not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    stage_cpu = defaultdict(list)
    stage_cuda = defaultdict(list)
    active = False
    started = 0.0
    elapsed = 0.0
    resolved = {}
    warmup_peak_allocated = warmup_peak_reserved = 0
    profiler = cProfile.Profile() if args.mode == "cpu" else None
    detailed = args.mode in {"stages", "trace"}
    device_profiler = (
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
            + ([torch.profiler.ProfilerActivity.CUDA] if using_cuda else [])
        )
        if args.mode == "trace"
        else None
    )

    def timed(name, function, cuda=False):
        @functools.wraps(function)
        def wrapper(*pos, **kw):
            if not active or not detailed:
                return function(*pos, **kw)
            begin = time.perf_counter()
            events = (
                (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                if cuda and using_cuda
                else None
            )
            if events:
                events[0].record()
            try:
                with torch.profiler.record_function(name) if device_profiler else nullcontext():
                    return function(*pos, **kw)
            finally:
                if events:
                    events[1].record()
                    stage_cuda[name].append(events)
                stage_cpu[name].append(time.perf_counter() - begin)

        return wrapper

    original_iter = torch.utils.data.DataLoader.__iter__
    original_build = trainer.build_unet

    def loader_iter(loader):
        nonlocal active, started, warmup_peak_allocated, warmup_peak_reserved
        iterator = original_iter(loader)
        for index in range(len(loader)):
            if index == args.warmup * resolved["accumulation_steps"]:
                if using_cuda:
                    torch.cuda.synchronize()
                    warmup_peak_allocated = torch.cuda.max_memory_allocated()
                    warmup_peak_reserved = torch.cuda.max_memory_reserved()
                    torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                active = True
                if profiler:
                    profiler.enable()
                if device_profiler:
                    device_profiler.start()
            yield timed("data_loader_total", lambda: next(iterator))()

    def build(*pos, **kw):
        model = original_build(*pos, **kw)
        model.forward = timed("forward", model.forward, cuda=True)
        return model

    def callback(event):
        nonlocal active, elapsed, resolved
        if event["type"] == "training_plan":
            resolved = event
            print(
                json.dumps(
                    {
                        "plan": {
                            key: event[key]
                            for key in (
                                "architecture",
                                "patch_size",
                                "microbatch_size",
                                "accumulation_steps",
                                "steps_per_epoch",
                            )
                        }
                    }
                ),
                flush=True,
            )
        # Exercise serialization, but do not pretend to measure Java/Appose UI latency.
        json.dumps(event)
        if event["type"] == "progress" and event["current"] == request["steps_per_epoch"]:
            if using_cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            active = False
            if profiler:
                profiler.disable()
            if device_profiler:
                device_profiler.stop()
            raise BenchmarkComplete()

    with (
        monitor_vram(args.output, args.monitor_vram and using_cuda) as vram,
        tempfile.TemporaryDirectory(prefix=f"jdll-{args.device}-profile-") as output,
        ExitStack() as patches,
    ):
        request["output_dir"] = output
        if getattr(args, "reuse_run", None) is not None:
            from .prepared_run import SavedRun
            replay = SavedRun(args.reuse_run)
            replay.prepare()
            patches.enter_context(replay.activate())
        patches.enter_context(patch.object(torch.utils.data.DataLoader, "__iter__", loader_iter))
        if detailed:
            patches.enter_context(patch.object(trainer, "build_unet", build))
            for owner, name, label, cuda in (
                (dataset.JdllSegmentationDataset, "_load_item", "source_load_normalization_stats", False),
                (ImageReadSession, "region", "region_read_decode", False),
                (CropSampler, "starts", "crop_coordinates", False),
                (crop_reading, "normalize_image", "crop_normalization", False),
                (augment, "sample_patch", "crop_selection", False),
                (dataset, "apply_augmentation", "cpu_crop_label_geometry", False),
                (augment, "_spatial_affine", "cpu_affine_labels", False),
                (dataset, "prepare_target", "cpu_targets", False),
                (targets, "normalized_instance_distance", "cpu_distance_targets", False),
                (SpatialImageBatch, "materialize", "image_transfer_spatial_gpu", True),
                (trainer, "_move_target", "target_transfer", True),
                (trainer, "complete_device_targets", "dense_targets_gpu", True),
                (trainer, "apply_tensor_photometric_augmentation", "photometric_gpu", True),
                (trainer, "compute_loss", "loss", True),
                (torch.Tensor, "backward", "backward", True),
                (torch.amp.GradScaler, "step", "optimizer", True),
                (torch.amp.GradScaler, "update", "scaler_update", True),
                (trainer, "_tensor_losses_to_float", "loss_reporting_sync", True),
                (LearningRateScheduler, "step_batch", "scheduler", False),
                (torch.optim.Optimizer, "zero_grad", "zero_grad", False),
            ):
                patches.enter_context(patch.object(owner, name, timed(label, getattr(owner, name), cuda)))
        with suppress(BenchmarkComplete):
            trainer.train(request, task=callback)
        data_cache = json.loads((Path(output) / "dataset_plan.json").read_text())["data_cache"]
        resolved_training = json.loads((Path(output) / "config.json").read_text())["training"]
    if elapsed <= 0:
        raise RuntimeError("Benchmark did not complete its measurement window")
    result = {
        "mode": args.mode,
        "device": args.device,
        "mixed_precision": resolved_training["mixed_precision"],
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name() if using_cuda else None,
        "cpu_threads": torch.get_num_threads(),
        "peak_process_rss_mb": peak_process_rss_mb(),
        "steps": args.steps,
        "warmup": args.warmup,
        "report_every": args.report_every,
        "seconds_per_step": elapsed / args.steps,
        "cache_mb_override": args.cache_mb,
        "resolved_data_cache_mb": data_cache["max_bytes"] / 1024**2,
        "patches_per_second": args.steps * resolved["effective_batch_size"] / elapsed,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2 if using_cuda else None,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 1024**2 if using_cuda else None,
        "run_peak_allocated_mb": (
            max(warmup_peak_allocated, torch.cuda.max_memory_allocated()) / 1024**2 if using_cuda else None
        ),
        "run_peak_reserved_mb": (
            max(warmup_peak_reserved, torch.cuda.max_memory_reserved()) / 1024**2 if using_cuda else None
        ),
        **vram,
        "plan": resolved,
        "cpu_ms_per_step": {key: 1000 * sum(values) / args.steps for key, values in stage_cpu.items()},
        "cuda_span_ms_per_step": {
            key: sum(start.elapsed_time(end) for start, end in events) / args.steps
            for key, events in stage_cuda.items()
        },
        "cpu_call_ms_p95": {key: float(np.percentile(values, 95) * 1000) for key, values in stage_cpu.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if profiler:
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(60)
        args.output.with_suffix(".profile.txt").write_text(stream.getvalue())
    if device_profiler:
        args.output.with_suffix(".operators.txt").write_text(
            device_profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=50)
        )
        device_profiler.export_chrome_trace(str(args.output.with_suffix(".trace.json")))
    print(json.dumps({key: value for key, value in result.items() if key != "plan"}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--reuse-run", type=Path, help="Reuse a verified saved run and its persistent prepared assets")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--mode", choices=("throughput", "stages", "cpu", "trace"), default="stages")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--cache-mb", type=float)
    parser.add_argument("--report-every", type=int, default=5)
    parser.add_argument("--monitor-vram", action="store_true", help="Sample process VRAM with nvidia-smi every 200 ms")
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 1 or args.report_every < 1:
        parser.error("steps, warmup and report-every must be positive")
    run(args)


if __name__ == "__main__":
    main()
