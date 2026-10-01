"""Monitor the real Appose-facing training entry point in an isolated process.

The parent samples Linux RSS/high-water marks without imposing an address-space
cap. Safety cancellation affects only the benchmark child, never other jobs.
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch


def read_status(pid):
    result = {}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            name, _, value = line.partition(":")
            if name in {"VmRSS", "VmHWM", "VmSwap"}:
                result[name] = int(value.split()[0]) * 1024
    except (OSError, ValueError):
        pass
    return result


def host_memory():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, _, value = line.partition(":")
        if name in {"MemAvailable", "SwapFree", "SwapTotal"}:
            values[name] = int(value.split()[0]) * 1024
    return values


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def worker(args):
    started = time.monotonic()
    state = {"phase": "imports", "case": None, "completed_steps": 0, "epoch": 0,
             "completed_checks": {}, "checks_complete": False}
    events = args.output / "events.jsonl"

    def event(kind, **payload):
        row = {"event": kind, "seconds": time.monotonic() - started, **payload,
               "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
        with events.open("a") as stream:
            stream.write(json.dumps(row, default=str) + "\n")
        write_json(args.output / "state.json", {**state, "seconds": row["seconds"]})
        print(json.dumps(row, default=str), flush=True)

    event("worker_started", pid=os.getpid(), python=sys.executable)
    import numpy as np
    import torch

    from jdll_unet import annotations, dataset, geometry, trainer, training_geometry
    from jdll_unet.appose_api import train

    monitor = None
    monitored = {}
    if args.activation_every:
        from .activation_monitor import ActivationMonitor
        monitor = ActivationMonitor(args.output / "activations", args.activation_every, args.epochs,
                                    args.activation_reference)

    class Task:
        def is_cancelled(self):
            return (args.output / "cancel.json").exists()

        def update(self, message="", current=None, maximum=None, info=None):
            payload = dict(info or {})
            state["completed_steps"] = payload.get("step", state["completed_steps"])
            state["epoch"] = payload.get("epoch", state["epoch"])
            if payload.get("type") == "training_plan":
                monitored["precision"] = payload["resolved_precision"]
                state["phase"] = "ready_for_training"
                state["checks_complete"] = True
                state["steps_per_epoch"] = payload.get("steps_per_epoch")
                write_json(args.output / "resolved_training_plan.json", payload)
            elif payload.get("type") == "progress":
                state["phase"] = "validation" if "validation" in message.lower() else "training"
            event("callback", payload=payload, message=message, current=current, maximum=maximum)
            if monitor is not None:
                epoch = 0 if payload.get("type") == "training_plan" else state["epoch"]
                ready = payload.get("type") == "training_plan" or (
                    payload.get("type") == "progress" and "validation" in message.lower()
                )
                if ready and monitor.due(epoch):
                    previous = state["phase"]
                    state["phase"] = "activation_diagnostics"
                    event("activation_start", epoch=epoch)
                    dtype = getattr(torch, monitored["precision"])
                    path = monitor.record(epoch, monitored["model"], monitored["validation"], dtype)
                    event("activation_report", epoch=epoch, path=str(path))
                    state["phase"] = previous

    def instrument(function, phase, describe=None):
        @functools.wraps(function)
        def wrapped(*a, **kw):
            previous = state["phase"], state["case"]
            state["phase"] = phase
            state["case"] = describe(*a, **kw) if describe else None
            event("phase_start", phase=phase, case=state["case"])
            before = time.monotonic()
            succeeded = False
            try:
                result = function(*a, **kw)
                succeeded = True
                return result
            finally:
                if succeeded:
                    counts = state["completed_checks"]
                    counts[phase] = counts.get(phase, 0) + 1
                event("phase_end", phase=phase, case=state["case"], duration=time.monotonic() - before,
                      succeeded=succeeded)
                state["phase"], state["case"] = previous
                write_json(args.output / "state.json", state)
        return wrapped

    original_analyze = annotations.AnnotationPreparation.analyze

    def source_analyze(self, pair, *a, **kw):
        dimensions = kw.get("dimensions", a[0] if a else "2d")
        if self._key(pair, dimensions) in self.records:
            return original_analyze(self, pair, *a, **kw)
        record = instrument(original_analyze, "annotation_preparation", lambda _self, p, *_, **__: p.stem)(self, pair, *a, **kw)
        event("annotation_storage", case=pair.stem, storage=record.storage,
              dtype=str(record.labels.dtype) if record.labels is not None else None,
              bytes=record.labels.nbytes if record.labels is not None else 0,
              skipped_reason=record.skipped_reason, extra_components=record.extra_components)
        return record

    original_set_epoch = dataset.JdllSegmentationDataset.set_epoch

    def set_epoch(self, epoch):
        result = original_set_epoch(self, epoch)
        if self.training and state["checks_complete"]:
            state.update(phase="training", case=None, epoch=epoch)
            event("epoch_start", epoch=epoch, total_epochs=args.epochs)
        return result

    config = dict.fromkeys((
        "task", "axes", "input_channels", "output_classes", "patch_size", "batch_size", "learning_rate",
        "foreground_oversampling", "foreground_probability", "augmentation_profile", "mixed_precision",
        "deep_supervision", "context_slices",
    ), "auto")
    config.update(
        model_name=args.output.name, output_dir=str(args.output / "model"),
        dataset_path=str(args.dataset), starting_point="scratch", architecture=args.architecture, device="cuda",
        epochs=args.epochs, num_workers=0, annotation_preparation={"disk_reserve_mb": args.disk_reserve_mb},
        empty_patch_fraction=args.empty_patch_fraction,
    )
    if args.resume_run is not None:
        config = json.loads((args.resume_run / "request.json").read_text())
        if config["epochs"] != args.epochs or config["dataset_path"] != str(args.dataset):
            raise ValueError("Resume must preserve the original dataset and total epochs")
        config.update(model_name=args.output.name, output_dir=str(args.output / "model"))
    if args.validation_mode is not None:
        config["validation"] = {"mode": args.validation_mode, "full_every": 1}
    if args.reuse_run is not None:
        config["data_cache_mb"] = 128
    if args.steps is not None:
        config["steps_per_epoch"] = args.steps
    write_json(args.output / "request.json", config)
    event("environment", torch=torch.__version__, numpy=np.__version__, torch_threads=torch.get_num_threads(),
          cuda_available=torch.cuda.is_available(), host=host_memory(), config=config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available in the selected Python environment")
    state["phase"] = "training_backend"
    with ExitStack() as stack:
        if args.resume_run is not None:
            checkpoint = args.resume_run / "model" / "weights_last.pt"
            stack.enter_context(patch.object(trainer, "_train", functools.partial(trainer._train, resume_from=checkpoint)))
            if monitor is not None:
                original_probes = args.resume_run / "activations" / "probes.pt"
                monitor.probes = torch.load(original_probes, map_location="cpu", weights_only=True)
                monitor.output.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original_probes, monitor.output / "probes.pt")
                for report in (args.resume_run / "activations").glob("epoch_*.json"):
                    monitor.recorded.add(int(report.stem.split("_")[-1]))
                    shutil.copyfile(report, monitor.output / report.name)
        if args.reuse_run is not None:
            from .prepared_run import SavedRun
            replay = SavedRun(args.reuse_run, normalize_projection=True if args.normalize_projections else None)
            state["phase"] = "restore_prepared_assets"
            event("restoring_prepared_assets", run=str(args.reuse_run))
            replay.prepare()
            stack.enter_context(replay.activate())
        if monitor is not None:
            original_build = trainer.build_unet
            original_make_dataset = trainer.make_dataset

            def capture_build(*a, **kw):
                model = original_build(*a, **kw)
                monitored["model"] = model
                return model

            def capture_dataset(*a, **kw):
                data = original_make_dataset(*a, **kw)
                if not data.training:
                    monitored["validation"] = data
                return data

            stack.enter_context(patch.object(trainer, "build_unet", capture_build))
            stack.enter_context(patch.object(trainer, "make_dataset", capture_dataset))
        stack.enter_context(patch.object(dataset.JdllSegmentationDataset, "set_epoch", set_epoch))
        for module, name, phase, describe in (
            (geometry, "inspect_pair", "inspection", lambda p, **_: p.stem),
            (annotations.AnnotationPreparation, "analyze", None, None),
            (trainer, "resolve_training_geometry", "dataset_planning", None),
            (training_geometry, "build_dataset_plan", "spacing_planning", None),
            (training_geometry, "plan_patch_and_microbatch", "memory_planning", None),
            (training_geometry, "validate_network_shape", "network_shape_check", None),
            (training_geometry, "measure_case_instances", "instance_size_estimation", lambda p, *_, **__: p.stem),
            (trainer, "_resolve_loss_weights", "loss_configuration", None),
            (trainer, "build_unet", "model_initialization", None),
            (trainer, "make_dataset", "sampling_setup", None),
            (dataset.JdllSegmentationDataset, "_load_item", "load_training_sample", lambda ds, i: ds.pairs[ds.items[i % len(ds.items)][0]].stem),
            (dataset.JdllSegmentationDataset, "__getitem__", "prepare_training_patch", None),
            (trainer, "_full_volume_validation", "full_volume_validation", None),
            (trainer, "_save_previews", "validation_previews", None),
            (trainer, "_save_checkpoint", "checkpoint", lambda p, *_, **__: str(p)),
        ):
            replacement = source_analyze if phase is None else instrument(getattr(module, name), phase, describe)
            stack.enter_context(patch.object(module, name, replacement))
        try:
            result = train(config, task=Task())
            write_json(args.output / "result.json", result)
            state["phase"] = "cancelled" if result.get("cancelled") else "complete"
            event("finished", result=result)
        except BaseException as exc:
            state["phase"] = "failed"
            event("failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            raise


def supervise(args):
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "control.json", {"max_seconds": args.max_seconds})
    command = [sys.executable, "-m", "benchmarks.measure_training_memory", "--worker", "--dataset", str(args.dataset),
               "--output", str(args.output), "--steps", str(args.steps) if args.steps is not None else "auto",
               "--epochs", str(args.epochs), "--architecture", args.architecture,
               "--disk-reserve-mb", str(args.disk_reserve_mb),
               "--empty-patch-fraction", str(args.empty_patch_fraction)]
    if args.reuse_run is not None:
        command.extend(["--reuse-run", str(args.reuse_run)])
    if args.resume_run is not None:
        command.extend(["--resume-run", str(args.resume_run)])
    if args.validation_mode is not None:
        command.extend(["--validation-mode", args.validation_mode])
    if args.normalize_projections:
        command.append("--normalize-projections")
    if args.activation_every:
        command.extend(["--activation-every", str(args.activation_every)])
    if args.activation_reference is not None:
        command.extend(["--activation-reference", str(args.activation_reference)])
    started = time.monotonic()
    report = {"status": "running", "command": command, "host_at_start": host_memory(), "peak_rss_mib": 0.0,
              "sampled_peak_rss_mib": 0.0, "peak_swap_mib": 0.0, "phase_peaks_mib": {}, "last_state": {},
              "minimum_available_mib": float("inf"), "minimum_swap_free_mib": float("inf"), "stop_reason": None}
    cancellation_time = None
    last_print = 0.0
    mib = 1024**2
    try:
        with (args.output / "worker.log").open("w") as log, (args.output / "memory.csv").open("w") as memory_log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            report["pid"] = child.pid
            writer = csv.writer(memory_log)
            writer.writerow(["seconds", "rss_mib", "hwm_mib", "swap_mib", "available_mib", "phase", "case"])
            try:
                while child.poll() is None:
                    elapsed = time.monotonic() - started
                    status = read_status(child.pid)
                    host = host_memory()
                    try:
                        state = json.loads((args.output / "state.json").read_text())
                    except (OSError, ValueError):
                        state = {"phase": "imports", "case": None}
                    phase = state["phase"]
                    rss, hwm, swap = (status.get(name, 0) / mib for name in ("VmRSS", "VmHWM", "VmSwap"))
                    available = host["MemAvailable"] / mib
                    disk_free = shutil.disk_usage(args.output).free / mib
                    report["peak_rss_mib"] = max(report["peak_rss_mib"], hwm)
                    report["sampled_peak_rss_mib"] = max(report["sampled_peak_rss_mib"], rss)
                    report["peak_swap_mib"] = max(report["peak_swap_mib"], swap)
                    report["minimum_available_mib"] = min(report["minimum_available_mib"], available)
                    report["minimum_swap_free_mib"] = min(report["minimum_swap_free_mib"], host["SwapFree"] / mib)
                    report["phase_peaks_mib"][phase] = max(report["phase_peaks_mib"].get(phase, 0), rss)
                    report["last_state"] = state
                    report["seconds"] = elapsed
                    writer.writerow([round(elapsed, 3), rss, hwm, swap, available, phase, state.get("case")])
                    max_seconds = json.loads((args.output / "control.json").read_text())["max_seconds"]
                    reason = None
                    if max_seconds > 0 and elapsed >= max_seconds:
                        reason = "time_limit"
                    if available < 1024:
                        reason = "memory_pressure"
                    if args.max_rss_gb and rss > args.max_rss_gb * 1024:
                        reason = "process_memory_limit"
                    if disk_free < 512:
                        reason = "disk_pressure"
                    cancel_path = args.output / "cancel.json"
                    if cancel_path.exists():
                        try:
                            reason = json.loads(cancel_path.read_text()).get("reason", "user_cancellation")
                        except (OSError, ValueError):
                            reason = "user_cancellation"
                    if reason and cancellation_time is None:
                        cancellation_time = elapsed
                        report["stop_reason"] = reason
                        report["stop_context"] = {"host": host, "process": status, "state": state}
                        write_json(args.output / "cancel.json", {"reason": reason})
                    if cancellation_time is not None and (
                        available < 1024 or disk_free < 512 or reason == "process_memory_limit"
                        or elapsed - cancellation_time > 60
                    ):
                        report["forced_termination"] = True
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                        break
                    if elapsed - last_print >= 15:
                        last_print = elapsed
                        memory_log.flush()
                        write_json(args.output / "summary.json", report)
                        print(json.dumps({"seconds": round(elapsed, 1), "phase": phase, "case": state.get("case"),
                                          "rss_mib": rss, "peak_rss_mib": report["peak_rss_mib"],
                                          "available_mib": available, "disk_free_mib": disk_free,
                                          "epoch": state.get("epoch"), "step": state.get("completed_steps"),
                                          "completed_checks": state.get("completed_checks"),
                                          "stop_reason": report["stop_reason"]}), flush=True)
                    time.sleep(0.1)
                report["exit_code"] = child.wait()
                report["peak_rss_mib"] = max(
                    report["peak_rss_mib"], resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
                )
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
        result_file = args.output / "result.json"
        result = json.loads(result_file.read_text()) if result_file.exists() else None
        report["result"] = result
        report["status"] = "complete" if result and not result.get("cancelled") and report["exit_code"] == 0 else "incomplete"
    finally:
        # Abrupt termination cannot run the library's normal cache cleanup.
        shutil.rmtree(args.output / "model" / ".annotation_cache", ignore_errors=True)
        report["seconds"] = time.monotonic() - started
        write_json(args.output / "summary.json", report)
        print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=lambda value: None if value == "auto" else int(value), default=3,
                        help="Optimizer steps per epoch; auto uses the normal library policy")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--architecture", default="resenc-tiny-3d")
    parser.add_argument("--disk-reserve-mb", type=float, default=256)
    parser.add_argument("--empty-patch-fraction", type=float, default=0)
    parser.add_argument("--max-rss-gb", type=float, default=12, help="Stop the worker above this RSS; zero disables")
    parser.add_argument("--reuse-run", type=Path)
    parser.add_argument("--resume-run", type=Path, help="Continue the last completed epoch into a new output directory")
    parser.add_argument("--validation-mode", choices=("light", "full"))
    parser.add_argument("--normalize-projections", action="store_true",
                        help="Use normalized projections when reusing an older run for scratch training")
    parser.add_argument("--activation-every", type=int, default=0)
    parser.add_argument("--activation-reference", type=Path)
    parser.add_argument("--max-seconds", type=int, default=900, help="Zero disables the time limit")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or (args.steps is not None and args.steps < 1) or args.max_seconds < 0 or args.disk_reserve_mb < 0:
        parser.error("Epochs and steps must be positive; time limit and disk reserve must be nonnegative")
    if not 0 <= args.empty_patch_fraction < 1 or args.max_rss_gb < 0:
        parser.error("Empty patch fraction must be in [0, 1); RSS limit must be nonnegative")
    if args.activation_every < 0 or (args.activation_reference is not None and not args.activation_reference.is_file()):
        parser.error("Activation interval must be nonnegative and its reference file must exist")
    worker(args) if args.worker else supervise(args)


if __name__ == "__main__":
    main()
