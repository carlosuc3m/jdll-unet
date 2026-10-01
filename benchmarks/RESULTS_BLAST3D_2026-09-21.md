# Blast3D Startup Check Timings

Historical baseline before shared source analysis. See the
[follow-up measurements](RESULTS_BLAST3D_SHARED_ANALYSIS_2026-09-21.md) for the
implemented optimization and remaining costs.

Dataset: `/home/carlos/Pictures/dicy_paper/blast3d/dataset/train`.
This is a mixed folder; the timing harness paired `_image.tif` with
`_masks.tif` directly without copying, linking, or changing source files.
No training iterations were run. Sampling stopped because the projected full
run substantially exceeded the user's 15-minute limit.

## Dataset Size

67 image/mask pairs contain 25,397,942,528 spatial voxels in total.

| Shape Z,Y,X | Cases |
| --- | ---: |
| 101,2048,2048 | 52 |
| 96,2048,2048 | 3 |
| 91,2048,2048 | 1 |
| 86,2048,2048 | 1 |
| 101,960,962 | 3 |
| 101,960,1568 | 5 |
| 101,1200,1564 | 2 |

Images are compressed uint16 TIFFs; masks are compressed uint8 TIFFs.
A typical image decodes to 808 MiB and its mask to 404 MiB, before working
copies and float32/int64 conversions.

## Measured Samples

Actual library functions were called in sequence with a shared
`ImageReadSession`, automatic data cache, and `AnnotationPreparation`.
Timings exclude Python import/startup and include each function's normal work.
The automatic task result for Blast_099 was `instance_friendly`, score 6.

| Stage | Blast_099, 101x960x962, 75 IDs | Blast_005, 101x2048x2048, 57 IDs |
| --- | ---: | ---: |
| Source inspection | 9.200 s | 31.155 s |
| Automatic task detection including connectivity | 6.443 s | Not run |
| Instance preparation reusing detection | 0.000207 s | Not run |
| Spacing plan | 0.018 s | Not run |
| Instance-size estimation | 59.961 s | Not run |
| Total measured stages | 75.623 s | 31.155 s |
| Process peak RSS | 2491 MiB | 3889 MiB |

Included inspection subtimings (do not add these to the totals):

| Operation | Blast_099 | Blast_005 |
| --- | ---: | ---: |
| Image decode | 1.219 s | 2.806 s |
| Mask decode | 0.498 s | 0.746 s |
| Unique-label calculation | 4.513 s | 10.958 s |

Task detection reused inspection labels and cached pixels: no repeated decode
or unique-label scan occurred in that stage. Instance preparation also reused
the existing connectivity result. Size estimation performed two unique-label
calculations, totaling 6.209 s, included in its 59.961 s total.

## Rough Full-Dataset Projection

These are extrapolations, not completed dataset measurements:

- Inspection: about 31.5 minutes, using a two-sample linear fit against voxels
  with a per-case intercept.
- Connectivity: about 29.2 minutes, scaling the smaller sample by total voxels.
- Instance sizing: about 3.6-4.5 hours **if** volumes have roughly 57-75 IDs,
  the two observed counts. Unique-label work was scaled by voxels; the remaining
  work was scaled by voxels times object count.
- Together: roughly 4.5-5.5 hours under those assumptions, excluding other
  startup work, training iterations, and validation inference.

The object-count range is a scenario, not a measured range for all 67 cases or
a statistical confidence interval. Different label counts, memory pressure,
page-cache state, or system load can substantially change these estimates.

## Remaining Bottleneck

`jdll_unet/scale.py:estimate_3d_instance_size` still iterates every label and
calls `np.argwhere(repair.labels == label)` over the full mask. It calculates
coordinates and principal axes for every object even with the default
equivalent-sphere measure, then selects up to 21 objects afterward.

Consequently the configured sample limit does not bound the expensive work.
In Blast_099, 75 IDs were processed before selecting 21. This separate
per-object full-volume scan remains after the task-detection optimization.

## Conditions And Safeguards

- Python: `/home/carlos/micromamba/envs/ofm/bin/python`; not the Appose runtime.
- Other CPU-heavy Python jobs were present; available RAM was about 4.8-5.0
  GiB, and swap was nearly full. This is not an idle-machine benchmark.
- BLAS/OpenMP/PyTorch CPU threads were limited to one in the benchmark process;
  TIFF decoding retained its normal library behavior.
- Each stage had a 90-second time budget and a process address-space cap.
- The first large-volume inspection attempts hit that artificial safety cap;
  they are not reported as completed timings or as proof of an uncapped library
  failure. The successful large sample used `MALLOC_ARENA_MAX=2` and a cap of
  the initial virtual size plus about 3.95 GiB, retaining 1 GiB of available-RAM
  headroom at setup. The small sample used the default allocator configuration.
- Large-volume connectivity/sizing was not attempted because several
  voxel-sized int64 workspaces would exceed the available memory headroom.
- No OS caches were flushed. In particular, the successful large inspection
  followed bounded retries, so it is not a cold-cache disk measurement.

The temporary harness is `/tmp/jdll-blast3d-checks.py`; raw reports are
`/tmp/jdll-blast3d-checks-Blast_099.json` and
`/tmp/jdll-blast3d-checks-Blast_005.json`.
