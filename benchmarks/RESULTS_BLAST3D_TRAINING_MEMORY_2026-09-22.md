# Blast3D Training-Path Memory Check

## Scope

The benchmark called the actual `jdll_unet.appose_api.train(config, task=task)`
entry point used by DeepIcy, with `resenc-tiny-3d`, CUDA, automatic configuration,
and zero DataLoader workers. It reused DeepIcy's linked dataset at
`/home/carlos/git/deep-icy/datasets/dataset-test-3d-3`: 67 training volumes and
4 validation volumes. No source images or annotations were changed.

The requested measurement was bounded to one epoch with three optimizer steps,
followed by the usual validation and output stages. **It stopped during startup;
no optimizer steps, full-volume validation, or previews were reached.** Java GUI
memory and linked-dataset creation are not included. Python was
`/home/carlos/micromamba/envs/ofm/bin/python`, not the Appose-managed environment.

## Observations

Final run: `/tmp/jdll-blast3d-deepicy-memory-20260922-03`.
The supervisor sampled process RSS/high-water marks and host memory every 100 ms,
and obtained the child's final peak RSS from `getrusage`. No address-space cap
or thread-count overrides were imposed. Other user processes remained running.

| Measurement | Result |
| --- | ---: |
| Elapsed time before termination | 1,210.71 s (20.18 min) |
| Completed image/mask inspections | 71 / 71 |
| Total inspection time | 156.43 s (2.61 min) |
| Completed connectivity/preparation checks | 32 / 71 |
| Time in those 32 checks | 1,031.17 s (17.19 min) |
| Mean / median preparation time per case | 32.22 s / 30.09 s |
| Preparation time range per completed case | 16.31-72.85 s |
| Process peak resident RAM, all reached phases | 11,774.58 MiB (11.50 GiB) |
| Sampled inspection-phase peak resident RAM | 2,003.08 MiB (1.96 GiB) |
| Maximum sampled process swap | 688.92 MiB |
| Minimum sampled host available RAM | 5,489.17 MiB |

Peak RSS includes resident file-backed mask pages, not just anonymous working
memory. It excludes GPU VRAM and other processes. The 11.50 GiB is an observed
startup peak, **not a demonstrated upper bound for training or validation**.

## Time Projection

Using the mean of 32 completed checks:

```text
inspection + preparation = 156.43 + 71 * 32.22 seconds
                         = approximately 40.7 minutes
```

This is an extrapolation for these two startup stages, not a measured complete
startup or total training duration. A rough 35-50 minute planning allowance is
more appropriate than an exact prediction; it is not a statistical confidence
interval. Case order, sizes, repairs, disk writes, swapping and concurrent load
affect the result. Later cache writes were skipped when the disk reserve check
failed, so the sampled preparation workload was not uniform. Later training
setup, batch loading, model work and validation remain unmeasured. Prior
single-case projections should not be read as full-training estimates.

## Disk Safety and Storage Correction

The old implementation retained repaired connectivity workspaces directly as
`int64`. A 101x2048x2048 volume used 3.15625 GiB plus its NPY header, regardless
of its small original IDs. Approximately 29 GiB accumulated in the run-owned
annotation cache, leaving only about 657 MiB available on the filesystem.

The previous attempts' caches had already been removed. Remaining files were
actively memory-mapped: unlinking them would not release their blocks while
mapped, and truncating them could corrupt the running process. The worker was
therefore terminated for disk safety, and the supervisor removed its cache.
Approximately 29 GiB was reclaimed. The benchmark was **not restarted**.
The final summary records exit code -15; the external cancellation reason is
recorded in the run's `cancel.json`, rather than its internal watchdog field.

Prepared-mask retention now chooses `uint8`, `uint16`, `uint32`, or `uint64`
from the known maximum repaired ID, without another volume scan or changing
existing IDs. Both RAM and disk budgets use that compact size. Disk writes
convert chunks of at most 1,048,576 voxels, avoiding a second full-volume cast.
For the same example volume, uint8 uses 404 MiB and uint16 uses 808 MiB.

These storage savings follow directly from dtype sizes and are covered by
regression tests. **No post-change dataset timing or RAM benchmark was run.**
Connectivity's transient working arrays remain unchanged, so the storage
reduction must not be interpreted as an equivalent reduction in peak RAM.

Two earlier attempts stopped on overly conservative swap-occupancy guards,
before training, despite adequate available RAM. Their peaks were 10.07 GiB
and 9.44 GiB; neither is a full-run result. Their temporary caches were removed.
