# Training Profiling

[Measured findings from 2026-09-21](RESULTS_2026-09-21.md) include the saved-config
regression, decoded-source caching, reporting cadence and batch-size comparisons.

Run from the repository root, using the Python environment being evaluated:

```bash
python -m benchmarks.profile_training \
  --config /path/to/model/config.json \
  --mode stages --warmup 50 --steps 150 \
  --output /tmp/unet-profile/stages.json
```

The script uses the real training loop, original dataset and augmentation policy,
the selected device and zero DataLoader workers. It starts a fresh model, warms
up before timing, then stops before validation or checkpointing. Only run-owned
temporary training files are removed. Input images, annotations and existing
models are untouched.

Modes:

- `throughput`: no stage instrumentation; compare patches/second here.
- `stages`: host wall timings and asynchronous CUDA event spans, synchronized
  only at measurement boundaries and the normal reporting points.
- `cpu`: cProfile output alongside the JSON result; this selects the profiler,
  not the training device.
- `trace`: PyTorch CPU/CUDA operator table and Chrome trace. Use a short measured
  window, such as `--steps 12`, to keep the trace manageable.

Options include `--device cpu|cuda` (default: CUDA), `--dataset`, `--batch-size`,
`--cache-mb`, `--report-every` and `--threads`. Batch overrides set both microbatch
cap and effective batch size.
`--threads` changes only this benchmark process, not library defaults.

For CPU/CUDA comparisons use `--mode throughput` and keep the config, batch,
cache budget, thread count, warmup, measured steps and reporting interval fixed.
Device-specific execution remains unchanged: CPU augmentation and full-precision
training on CPU, deferred CUDA augmentation and configured AMP on CUDA. Results
record the actual mixed-precision setting. Run devices sequentially to avoid
contending for CPU resources.

CUDA results record exact PyTorch peaks for allocated and reserved memory. The
`run_peak_*` fields also include warmup; the other peak fields start after warmup.
These are not total process VRAM. Add `--monitor-vram` to sample this process with
`nvidia-smi` every 200 ms in a separate monitor process, without querying in the
training step. It produces a `.vram.csv` file and `sampled_process_peak_mb`;
sampling can miss brief peaks. All memory fields ending in `_mb` use MiB.
Monitoring is optional and can perturb performance; keep it consistent across
comparisons. There are still zero DataLoader workers.

`peak_process_rss_mb` records the operating system's process RSS high-water mark
on supported Unix systems, including imports, startup, decoded-source cache and
training. It excludes other processes and is not total machine RAM use. This
metric is collected after timing, without a monitoring thread or worker.

CPU subtimings overlap their parent stages. CUDA event spans include host launch
gaps; they are not pure GPU kernel execution time. The operator trace separates
actual CUDA kernel work. Profilers perturb timings: confirm improvements using
`throughput`, keep other workloads idle, and repeat comparisons. Warmup does not
guarantee that every source's normalization statistics have already been fitted.

Callbacks serialize progress in Python. Java/Appose transport and UI rendering
are not measured. Match the deployed Python/PyTorch versions before interpreting
these numbers as UI performance predictions.
