# Blast3D Shared Source Analysis

Dataset: `/home/carlos/Pictures/dicy_paper/blast3d/dataset/train`.
These measurements cover startup checks, not training iterations. No input
files were changed. Approximate downsampling and periodic audits are not implemented.

## Changes

- Inspection accumulates exact per-ID counts, bounds and per-plane summaries
  in tiles of at most 1,048,576 pixels. Large sparse IDs use compact local
  mappings, not an allocation indexed by the largest source ID.
- Native image dtypes are retained during inspection. Integer image types
  require no finiteness scan; floating images are checked in bounded chunks,
  including float32 overflow. Mask validation remains exact and dtype-aware.
- Task detection, connectivity preparation and sizing reuse source statistics.
  Connectivity still checks all equal-valued regions at full resolution.
  Corrected IDs have separate statistics, retained with prepared-mask analysis.
  Unchanged masks reuse inspection statistics. Spatial holdout domains have
  separate summaries; changing the source file invalidates cached results.
- The instance sample is selected reproducibly before object-specific work.
  Equivalent diameters use counts, with physical voxel volume in 3D. Only the
  expert principal-axis measure computes coordinates, for selected objects
  inside their bounds, using bounded blocks of centered moments.
- 2.5D uses the same exact cross-section policy as before: central section for
  complete objects, largest available section for Z-truncated fallback objects.

## Measurements

Historical baseline: [previous report](RESULTS_BLAST3D_2026-09-21.md).

| Startup stage | Blast_099 before | Blast_099 now |
| --- | ---: | ---: |
| Inspection | 9.200 s | 1.470 s |
| Task detection including connectivity | 6.443 s | 6.475 s |
| Reusing instance preparation | 0.000207 s | 0.000051 s |
| Spacing plan | 0.018 s | 0.004454 s |
| Size estimation | 59.961 s | 0.000679 s |
| Total of these stages | 75.623 s | 7.950 s |

This is about **9.5x faster** against the earlier sample. It is not a controlled
comparison on an idle machine. Both runs used 75 instance IDs in the same
101x960x962 volume. The new run performed exactly **one** aggregate mask analysis,
inside inspection (0.750 s included in its total), and no further aggregate
analysis during connectivity, preparation reuse or sizing.

To compare the algorithms under current conditions, the benchmark also ran a
reference implementation of the former sizing loop after the new stages.
That reference includes two unique-label scans and per-label coordinates and
principal axes before sampling, but uses the current mask loader/validator.
It took **45.565 s**, versus **0.000679 s** for the cached estimator. Both selected
21 objects and returned the same physical median diameter, **12.575077005561086**.

The cached-time ratio alone is misleading: the new estimator consumes work
already done by inspection. Charging it the entire fresh aggregate analysis
as well gives approximately **0.751 s**, around **61x faster** than the old sizing
algorithm on this sample. This does not imply a 61x end-to-end startup speedup.

For the larger **Blast_005**, 101x2048x2048 with 57 IDs:

- Inspection: **31.155 s previously, 3.110 s now**, about 10x faster.
- Included exact mask analysis: **1.146 s**, one call.
- Inspection process peak RSS: **1484 MiB**, versus the previous **3889 MiB**.
- Large-volume connectivity and sizing were not run because their transient
  connectivity workspaces exceed the available memory headroom.

The small-case process peak RSS remained around 2493 MiB, including the old
reference run. Full-resolution connectivity still needs substantial transient
memory; the bounded statistics workspace does not remove that requirement.

## Full-Dataset Estimate

For the known 67 cases and 25,397,942,528 voxels:

| Stage | Current projection |
| --- | ---: |
| Inspection | 3.2 minutes |
| Connectivity | 29.4 minutes |
| Cached default size calculations | Negligible relative to the above |
| Total of these stages | About 33 minutes |

Inspection uses a two-sample linear fit with a per-case intercept. Connectivity
scales the small case's time by total voxel count. Thus the old **4.5-5.5-hour**
projection becomes roughly **half an hour**, about **8-10x faster** under those
assumptions. This is not a completed full-dataset measurement or a confidence
interval. Connectivity is now the dominant measured startup cost.

The projection assumes sufficient RAM for each connectivity operation and a
similar repair workload. Swapping, widespread repairs, storage throughput,
different object counts and other system load can substantially change it.
Other startup stages, per-patch work and training are excluded. No attempt was
made to run all 67 cases beyond the user's 15-minute benchmark limit.

## Reproduction

```bash
env PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MALLOC_ARENA_MAX=2 \
  /home/carlos/micromamba/envs/ofm/bin/python benchmarks/benchmark_source_analysis.py \
  /home/carlos/Pictures/dicy_paper/blast3d/dataset/train Blast_099 \
  --legacy --output benchmarks/RESULTS_BLAST3D_SHARED_Blast_099.json
```

Use `Blast_005 --inspection-only` and a different output path for the larger
case. The harness limits each stage to 90 seconds and the run to 900 seconds,
checks available RAM, and caps additional process address space at 4 GiB.

At measurement time, available RAM was approximately 7 GiB and the 1-minute
system load was 31-33. Other CPU-heavy user jobs were left untouched. No tests
or other agent benchmarks ran concurrently. TIFF decoding used its normal
thread policy; PyTorch and BLAS/OpenMP threads were limited to one. No OS caches
were flushed, so these are not cold-storage benchmarks. The historical small
sample did not use the same allocator-arena setting.

Raw reports: [small case](RESULTS_BLAST3D_SHARED_Blast_099.json),
[large case](RESULTS_BLAST3D_SHARED_Blast_005.json).
