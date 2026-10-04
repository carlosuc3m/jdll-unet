# JDLL UNet Backend

This repository contains the first JDLL-owned UNet backend described in
`jdll-unet-backend-plan.md`.

The Python package is named `jdll_unet` and exposes Appose-friendly entry
points:

```python
from jdll_unet.appose_api import train, infer, detect_task
```

The implementation supports lightweight 2D, 2.5D, and true 3D UNet training and
inference for binary semantic, multiclass semantic, and instance-friendly
segmentation datasets laid out as `images/` and `masks/`, or as explicit
`train/images`, `train/masks`, `val/images`, and `val/masks` folders.
Ordinary 2D images support TIFF, PNG, BMP, and JPEG; integer masks support TIFF,
PNG, and BMP, while lossy JPEG masks remain prohibited. BMP is intentionally
2D-only, and volumetric images and labels require TIFF stacks.
For accepted extensions, confirmed header mismatches select the actual format's
reader. TIFF volumes, axes, spacing, and precision survive misleading raster
extensions; palette masks retain their integer indices. Successful corrections
are remembered per run, folder, extension, and image/mask role and reported
through callbacks. Corrupt data, JPEG mask contents, independent TIFF series,
and animated rasters fail clearly instead of falling back to a partial image.
For JDLL compatibility, channel-last 2D label masks use channel zero and emit a
warning when additional channels are discarded. Dimension-aware loading keeps
`Z,Y,X` masks volumetric and rejects ambiguous 3D channel layouts.

## Architectures

Universal residual-encoder presets are available for `2d`, `2.5d`, and `3d`:

- `resenc-tiny-*`: `[16,32,64,128]`, reference budget 4 GB.
- `resenc-medium-*`: `[24,48,96,192,320]`, reference budget 8 GB.
- `resenc-big-*`: `[32,64,128,256,320]`, reference budget 16 GB.
- `resenc-large-*`: `[32,64,128,256,384,512]`, reference budget 24 GB.

Replace `*` with `2d`, `2.5d`, or `3d`. Legacy `tiny-*` and `medium-*`
architecture names remain loadable for existing configurations and checkpoints.
The universal preset fixes model capacity, context, and preferred patch size
independently of installed hardware. Runtime planning may reduce microbatch and
patch size, with gradient accumulation preserving the effective batch target.
Small (`tiny`) 2D training defaults to microbatch cap and effective batch 32 on
CUDA, or 16 on CPU. MPS, 2.5D, 3D and other model sizes retain their existing microbatch
caps and default effective batch four. Explicit batch settings, including values
in saved configurations, override these defaults.
Small (`tiny`) models use an automatic patch budget per epoch, so larger
effective batches reduce the number of optimizer updates instead of increasing
patch exposure. Medium, big and large retain their minimum optimizer-step budget.

The four presets are complete speed/quality tiers rather than capacity-only
ablation variants. Their automatic spatial and 2.5D context defaults are:

| Preset | 2D/2.5D preferred patch | 2.5D context slices | 3D preferred patch | Deep supervision |
| --- | --- | ---: | --- | --- |
| Small (`tiny`) | `[128,128]` | 5 | `[16,64,64]` | no |
| Medium | `[256,256]` | 7 | `[24,96,96]` | yes |
| Big | `[384,384]` | 9 | `[32,128,128]` | yes |
| Large | `[512,512]` | 11 | `[48,160,160]` | yes |

Context counts remain fixed across hardware and may be overridden explicitly.
The memory planner caps microbatch by preset, reduces it before reducing the
patch, and uses gradient accumulation to preserve the effective batch target. It then
shrinks the preferred patch only when required by the smaller of detected
available memory and the preset's 4/8/16/24 GB reference budget. Preferred and
resolved decisions are persisted in model metadata and emitted as a
`training_plan` callback before the first epoch.
An explicit patch size is kept fixed while microbatch can still decrease to fit
the estimated budget. Neither the memory estimate nor the measured tiny-2D VRAM
usage guarantees that batch 32 fits every 1 GB GPU, channel count or patch size.

The default architecture is `resenc-tiny-2d`. Genuine 2.5D variants are also
available as `tiny-2.5d`, `medium-2.5d`, `resenc-tiny-2.5d`, and
`resenc-medium-2.5d`. They use a 2D UNet with an odd number of neighboring Z
slices flattened into input channels; configure the total with
`"context_slices"`; when omitted it resolves to 5/7/9/11 by preset. Missing
context beyond either Z boundary is zero padded.
Context sampling supports `adjacent`, `fixed_stride`, and `nearest_physical`.
The automatic physical target is the median resolved training Z spacing and
always selects real slices; it never interpolates a 2.5D context channel.

The `resenc-*` variants keep the UNet encoder-decoder shape but replace encoder
conv blocks with residual blocks for better gradient flow. Deep supervision can
be overridden with `"deep_supervision"`; it defaults off for Small and on for
Medium, Big, and Large. The trainer applies auxiliary losses to intermediate
decoder outputs while inference uses only the primary full-resolution output.

Convolutional UNet blocks use group normalization by default because it is
stable for the small batches common in biomedical segmentation. Set
`"model_normalization"` to `group`, `instance`, `batch`, or `none` to override
the default. This is separate from the image-intensity `"normalization"` setting.

True 3D models use image tensors shaped `C,Z,Y,X`, masks shaped `Z,Y,X`, and
logits shaped `B,C,Z,Y,X`. Multipage TIFF/OME-TIFF image and label stacks are
loaded as volumes; RGB 2D images are rejected for
true 3D models instead of being guessed as volumes.

## Physical Planning

The trainer reads explicit JSON sidecars and OME/ImageJ TIFF metadata in `Z,Y,X`
order. If at least half the cases have spacing, missing axes use the known
per-axis median. Otherwise missing cases use `spacing.default_spacing`, which
defaults to `[1,1,1]`. Provenance is never discarded.

True 3D data is reversibly resampled to a dataset target grid. Strongly
anisotropic datasets use a robust coarse-axis target with a threefold automatic
upsampling safeguard. Kernels and strides are derived per stage: coarse axes use
`1` kernels/strides until physical resolutions become comparable, and no axis is
downsampled below four feature-map positions. Expert kernel, stride, target
spacing, and patch overrides remain possible through resolved configuration.

Training writes reusable user configuration to `config.json`, measured dataset
information to `dataset_fingerprint.json`, generated resolved spacing sidecars
to `resolved_spacings/`, and resolved model/runtime decisions to
`model_metadata.json`.

For semantic tasks, the fingerprint includes connected-region scale
diagnostics in resampled model space. It reports pooled and per-class p10, p25,
median, p75, and p90 area fractions for 2D and 2.5D center slices, or volume
fractions for 3D. Border-touching regions are tracked separately and provide a
fallback when no complete regions exist. At inference, pass one of
`semantic_region_fraction`, `semantic_region_area` (2D/2.5D pixels), or
`semantic_region_volume` (3D voxels) to compare an approximate region size with
the training distribution. Inference rescales XY by the square root of the
area ratio for 2D/2.5D, or XYZ by the cube root of the volume ratio for 3D,
then restores predictions to the input geometry. The default scale-factor
bounds are 0.25 and 4.0; override them with `semantic_scale_min_factor` and
`semantic_scale_max_factor`. Comparison and applied-scale details are returned
in inference metadata.

`semantic_region_size` is a dimension-aware alias for area or volume.
`object_size` is also accepted for convenience on semantic models, where it
means area/volume; on instance models it continues to mean object diameter.

## Install

```bash
python -m pip install -e ".[test]"
```

## Minimal Training Example

```python
from jdll_unet.appose_api import train

result = train(
    {
        "model_name": "cells",
        "output_dir": "models/unet/cells",
        "dataset_path": "datasets/cells",
        "starting_point": "scratch",
        "architecture": "resenc-tiny-2d",
        "deep_supervision": False,
        "model_normalization": "group",
        "lr_scheduler": {"type": "poly"},
        "device": "auto",
        "epochs": 100,
        "seed": 42,
    }
)
print(result["model_path"])
```

Training writes `config.json`, `weights_best.pt`, `weights_last.pt`,
`model.pt`, `training.log`, `metrics.json`, and optional previews into the
model folder.

Training also saves `dataset_plan.json`, which records source geometry, accepted
and skipped cases, split regions, spacing, padding, eligible planes, and sampling
quotas. The effective configuration and checkpoints contain resolved values;
per-case decisions stay in the dataset plan.

Decoded images and masks use a shared, run-scoped LRU cache so repeated patches
do not repeatedly decompress the same source. `data_cache_mb="auto"` allows up
to 10% of available host RAM, capped at 512 MiB (64 MiB if availability cannot be
determined). Linux container memory limits are considered. An explicit
nonnegative MiB budget is supported; `0` disables this cache. Planning, training
and validation share the budget; large arrays bypass it. This is separate from
the prepared-instance-mask cache. The resolved byte limit is recorded in
`dataset_plan.json`; the exported config stores the resolved MiB budget, like
other resolved settings. Set it back to `"auto"` when reusing the configuration
to recalculate the budget for another machine.

Normalization statistics are cached per source/domain (per plane for 2D sampling
from volumes). 2D/2.5D patches are normalized after cropping; 3D retains
normalization before spacing resampling. CUDA training automatically runs image
scale resizing, flips, quarter-turns, affine/elastic resampling, low-resolution
simulation, blur, and photometric augmentation on the GPU, including with
`num_workers=0`. Spatial stages run as batched operations with independent
parameters per sample; differently sized source crops are resized in compatible
groups without padding every 3D crop to the largest source shape.
Transform parameters, label geometry, and validity are planned on CPU; the same
geometry is replayed on GPU for all image/context channels. Instance boundaries
and distances are constructed from the transformed labels, not warped from
precomputed targets. Foreground, semantic, and boundary targets are built on the
GPU. Cropping, label/validity geometry, elastic coordinate-field generation, and
exact instance-distance targets remain on CPU; no GPU-to-CPU image or mask
transfer is needed. No background loader threads or processes are added.
Validation and CPU-only training retain their existing path.
`training.log` records the loaded trainer path and image augmentation backend,
which can be used to verify the package loaded by DeepIcy/Appose. Restart its
Python worker after updating the package.

Progress and log intervals count optimizer steps. Reported training losses are
means since the preceding progress/log report; epoch losses cover the full epoch.
Loss values stay on the device between reports. Validity checks use the original
CPU mask, including the exact support of every deep-supervision head.

For opt-in performance diagnosis, see [benchmarks/README.md](benchmarks/README.md).
Profiling adds no hooks or synchronization to ordinary training runs.

## Training Geometry and Validation

2D training accepts standalone images and individual planes from volumes when
the supplied dataset contains at least one valid standalone 2D image/mask pair
(a singleton Z=1 stack also qualifies). This condition applies to scratch
training and fine-tuning across the dataset, not separately within each split.
Volume-only datasets require a 2.5D or 3D model.
2.5D and 3D training use eligible volumes and report skipped standalone images.
Explicit TIFF axes and image/mask spatial geometry determine the interpretation;
ambiguous pairs require an export with explicit axes.

Splits keep each original source together, including sources reached through
links. One sufficiently large 2D image, or one volume for a 2.5D/3D model, uses
disjoint spatial regions for training and validation. All preprocessing, context, and augmentation stay within
the assigned region. Infeasible holdouts fail clearly; training content is never
reused as validation. This is within-source validation, not an independent specimen.

Each run uses one patch shape validated against the actual network. The default
padding limit is one real domain length per side on each spatial axis. A real
depth of 8 can support a depth-16 patch; depth 4 cannot. Added spatial padding
does not contribute to targets, losses, or metrics. This validity information
does not identify unannotated real content.
Spatial image padding is zero in training, validation, and tiled inference.
Padded targets remain invalid, not annotated background; image-only augmentation
does not turn padded spatial locations into nonzero input.

2.5D keeps the resolved context count and stride. For depth 4, context 11, and
stride 1, centers 1 and 2 are eligible; all four real planes remain available
as context. Missing context is zero-filled and does not invalidate a real center.
See [the integration handoff](README_TRAINING_GEOMETRY_HANDOFF.md) for callbacks,
artifacts, and Java integration details.

## Empty Training Samples

Training excludes images with empty masks and rejects empty sampled patches by
default. Validation images and patches are always retained, including empty
ones, and their counts are logged. Configure the training policy with:

```python
"skip_empty_images": True,
"skip_empty_patches": True,
"empty_patch_max_retries": 8,
"include_empty_patches_after_max_retries": False,
```

`skip_empty_patches=True` is the stricter policy and always excludes empty output
patches, including after augmentation. Sampling retries eligible targets within
the source and has a foreground-centered fallback with spatial deformation
disabled. Training fails clearly if no feasible foreground target remains.
The legacy `include_empty_patches_after_max_retries` flag cannot override this
stricter policy.

When negative patches are permitted, `max_empty_plane_fraction=0.20` caps empty
training planes per volume and epoch. Twenty positive planes permit up to five
empty planes; three positive planes permit none. Empty subsets rotate
reproducibly across epochs, and repeated draws also respect the cap. Wholly
empty volumes contribute no plane samples under this quota. Validation keeps
eligible real empty content. Advanced callers can set `max_empty_plane_fraction`
in `[0,1)` and the finite nonnegative per-side `max_padding_ratio` (default `1.0`).

## Instance Scale Normalization

`instance_friendly` models normalize each image or volume toward a canonical
median instance diameter by default. Training masks use up to 21 reproducibly
sampled instances. Border-touching
and tiny instances are excluded by default. Binary masks are split into
connected components; instance-ID masks use their label IDs.
Before size estimation and training, source annotations are prepared once per
run. Explicit semantic tasks never split class labels. For automatic detection,
source label summaries are checked first, reusing inspection results. Connectivity
is skipped when it cannot change the task decision. Otherwise one equal-label
connected-component pass handles all IDs, and its results are retained and reused
if an instance task is selected. There is no full-volume pass per object ID.
Detection statistics mark unmeasured connectivity with `connectivity_analyzed=false`
and an empty component-count mapping; this does not mean every ID was checked.
High label counts alone do not force an instance interpretation, and ambiguous
cases still require an explicit task. Neither detection nor preparation requires
consecutive IDs.

Instance preparation uses face connectivity (4-neighbor in 2D, 6-neighbor across
the whole volume in 2.5D/3D). The first component retains its original ID; extra
components receive unused IDs above the original maximum. Size estimation,
sampling, patch targets and validation then share these identities. Patch targets
compact IDs to `1..N` for efficient indexing, without splitting fragments created
by slicing, cropping or augmentation. Touching objects sharing an ID cannot be
separated by connected components.

Original annotation files are never modified. Unchanged masks need no prepared
copy. Corrected masks use a bounded RAM cache, then read-only disk-backed copies.
Both store the smallest lossless unsigned integer type for the repaired IDs
(`uint8`, `uint16`, `uint32`, or `uint64`), without renumbering existing IDs.
Disk writes convert bounded chunks rather than allocating another full volume;
cache budgets use the compact size, not the connectivity workspace size.
The default disk directory is `<output_dir>/.annotation_cache`; run-owned files
are removed on completion, cancellation or failure. Source or policy changes
invalidate cached analysis. If cache storage is unavailable, the original IDs
are used unchanged, with a warning and **no repeated per-patch repair**. This can
leave separate objects sharing an ID. The RAM budget covers retained masks;
connected-component analysis still needs transient working memory for one source.

```python
"annotation_preparation": {
    "repair_disconnected_instances": True,
    "ram_cache_mb": 128,
    "cache_dir": None,
    "disk_reserve_mb": 256,
    "warning_fraction": 0.10,
}
```

Affected IDs, affected sources, fractions, storage choices/dtypes/bytes, and skipped repairs
are recorded under `annotation_preparation` in `dataset_plan.json`, with a log
summary. A warning is raised when either affected fraction reaches
`warning_fraction`; it never overrides the selected task. Configuration contains
only the policy, not dataset-specific results.
Instance distance transforms operate on local bounding boxes.

Inspection, task detection, annotation preparation and instance sizing share
run-scoped source statistics. Exact per-ID counts, bounds and cross-sections
are accumulated in bounded tiles at native resolution, without a full-volume
scan for every ID. Repaired annotations have their own reusable statistics;
spatial holdout domains are measured separately. The configured instance limit
applies before expensive object-specific measurements, using reproducible
random sampling rather than the first IDs. Equivalent diameters use cached
counts; principal axes are computed only when requested and only for sampled
objects. This sample limit never limits connectivity validation or repair.
No approximate downsampling or periodic approximate-analysis audits are used.

```python
"instance_scale_normalization": {
    "enabled": True,
    "target_object_fraction": 0.25,
    "object_size_measure": "equivalent_sphere_diameter",
    "max_instances_per_image": 21,
    "exclude_border_instances": True,
    "min_instance_area": 4,
    "training_scale_jitter": [0.5, 2.0],
    "jitter_distribution": "log_uniform",
    "min_effective_scale": 0.25,
    "max_effective_scale": 4.0,
}
```

For 2D/2.5D the target diameter is relative to patch pixels. For 3D it is
relative to the smallest physical patch extent. Training
draws log-uniform scale jitter and extracts the corresponding crop directly
from the original image before resizing it to the fixed patch size. Validation
uses its reproducibly sampled mask median without jitter. Dataset-derived
measurements are written separately to `dataset_statistics.json`.

Inference requires the approximate median object diameter in native input
pixels. It rescales the image to the model's canonical object size, performs
tiled prediction, restores foreground and boundary probabilities to the
original geometry, and then creates instance labels from foreground, boundary,
and normalized per-instance distance predictions:

```python
result = infer(
    {"model_path": "models/cells/model.pt", "object_size": 18},
    {"image_path": "images/cells.tif"},
)
```

## Fine-Tuning

Fine-tuning recovers the complete architecture from the source model; callers
should omit `architecture` and use an automatic learning rate:

```python
result = train(
    {
        "model_name": "adapted-cells",
        "output_dir": "models/adapted-cells",
        "dataset_path": "datasets/new-cells",
        "starting_point": "fine_tune",
        "base_model": "models/source-cells",
        "learning_rate": "auto",
    }
)
```

The source backbone, dimensionality, kernels, strides, normalization, context,
and deep-supervision topology are reconstructed strictly. Only input
convolutions and output heads may be adapted. Omitted settings and explicit
`auto` inherit the source values, including its context count. Fine-tuning uses
one tenth of the original scratch `base_learning_rate` for backbone parameters
and that base rate for adapted layers. Successive fine-tunes keep the same base
rate. Unrecoverable legacy provenance falls back to base `1e-3`, with a warning,
giving backbone/adapted rates `1e-4` and `1e-3`. A fresh
optimizer and scheduler preserve this group ratio. `config.json`,
`model_metadata.json`, checkpoints, and the `training_plan` callback record the
source paths, resolved rates, adaptation summaries, and complete tensor audit.

For 2.5D instance models, object identities are canonicalized per volume with
efficient 3D connected components. Disconnected regions sharing an annotation
ID receive fresh IDs in the shared prepared mask. Up to 21 objects are measured per volume,
prioritizing objects that do not touch a Z boundary; Z-boundary objects supply
their largest available cross-section only when needed. One volume-level XY
scale is shared by all center slices. Context channels receive one synchronized
XY crop and transform, while validation uses every Z plane without jitter.

2.5D inference accepts one approximate XY `object_size`; 3D accepts an
approximate physical equivalent-sphere diameter. The expert `principal_axes`
measurement is also supported. Three-dimensional EDT targets use physical
spacing. Reconstruction blends all tiled maps, restores native geometry,
extracts robust distance markers, and runs boundary-aware marker-controlled
watershed once at native resolution.

The mixed boundary target uses a one-voxel outside ring, both object sides of a
touching-ID interface, and the outermost object voxel at array edges. Physical
minimum seed/object sizes and face/full connectivity are configurable.

## Learning Rate Scheduling

Training uses polynomial decay by default:

```python
"lr_scheduler": {
    "type": "poly",
    "min_lr": 0.0,
    "poly_power": 0.9,
}
```

Supported scheduler types:

- `poly`: nnU-Net-compatible epoch-level polynomial decay with power `0.9`; this is the default.
- `cosine`: per-step cosine annealing from `learning_rate` to `min_lr`.
- `plateau`: epoch-level reduction when the validation score stops improving, using `plateau_factor`, `plateau_patience`, and `plateau_threshold`.
- `none`: constant learning rate.

For convenience, `lr_scheduler` may be either a string such as `"cosine"` or a
mapping with scheduler options. `learning_rate_scheduler` and `scheduler` are
accepted as aliases.

## Losses

The trainer chooses a composite segmentation loss from the detected task:

- Binary semantic: BCE with logits plus Dice loss.
- Multiclass semantic: cross entropy plus Dice loss.
- Instance-friendly: foreground BCE/Dice, boundary BCE, and foreground Smooth L1 normalized-distance loss.

Patch training uses a deterministic random stream. Small (`tiny`) models in
2D, 2.5D and 3D use a patch budget for automatic epoch length:

```text
patches_per_epoch = max(1000, 10 * training_cases)
steps_per_epoch = ceil(patches_per_epoch / effective_batch_size)
```

The minimum is configurable with `minimum_patches_per_epoch` (default 1000),
and the per-case budget with `expected_patches_per_case` (default 10).
The batch is the resolved effective batch after gradient accumulation. Whole
batches can round the actual patch count up slightly. Cases are training
images/volumes, not context slices. For 150 cases, small 2D defaults give
47 updates on CUDA (batch 32) or 94 on CPU (batch 16), both using 1504 patches.
Equal patch counts do not imply equal optimization: larger effective batches
perform fewer updates, and polynomial decay still advances per epoch.

Medium, big and large models retain their existing automatic optimizer-step budget:

```text
steps_per_epoch = max(250, ceil(10 * training_cases / effective_batch_size))
```

An explicit `steps_per_epoch` overrides automatic scheduling for every preset.
`minimum_steps_per_epoch` defaults to `null` (use the preset policy); an explicit
positive value sets an optimizer-step floor, including for small models. Old
saved configurations with explicit steps or a minimum of 250 retain those
settings. To adopt the new automatic small-model policy, set
`steps_per_epoch="auto"` and `minimum_steps_per_epoch=null` (or omit the latter).

For 2.5D/3D, fixed patch validation runs every epoch and selects checkpoints.
Full-volume validation is optional and diagnostic only, disabled by default
(including the final epoch). Early stopping is disabled; use cooperative
cancellation. Existing 2D behavior is unchanged: full validation defaults to
every five epochs plus the final epoch, with full-case checkpoint selection and
patience of 20 full validations; `validation.mode="light"` uses patch selection.

### Volumetric Validation

The configurable volumetric defaults are:

```json
{
  "preview_count": 4,
  "validation": {
    "minimum_batches": 50,
    "minimum_samples": 100,
    "foreground_fraction": 0.33,
    "minimum_foreground": 0.01,
    "minimum_source_fraction": 0.5,
    "max_sampling_overlap": 0.10,
    "candidate_attempts": 16,
    "preview_max_bytes": "auto",
    "tile_overlap": 0.25,
    "tile_blending": "constant",
    "full_every": 0,
    "early_stopping_patience": 0
  }
}
```

`B` is the resolved validation microbatch, not the effective accumulated
training batch. The base budget is `max(50, ceil(100 / B)) * B` patches:
100 at B=1/2, 200 at B=4. Sampling uses a seeded spatial lattice, normally with
no overlap; insufficient capacity permits up to the configured overlap per
axis. This is independent of inference tile overlap. If that capacity is still
below the base budget, use all available lattice patches, remove foreground
requirements, and allow a partial final batch. This is a bounded feasible
packing, not a claim of globally optimal arbitrary-coordinate packing.

Normally `ceil(0.33 * samples)` patches are forced positives, measured on real
target support (the central plane for 2.5D). At least half of eligible positive
sources contribute where feasible. The other samples are unconstrained, not
necessarily empty. Cached foreground reservoirs supply candidates; masks are
checked before images. Search stops when the quota/coverage is met or after
`candidate_attempts * forced_quota` attempts. For sparse annotations, lower the
occupancy threshold to the strongest feasible bounded candidate set; if still
necessary, report a quota/coverage shortfall. Never invent foreground, repeat
patches, change scale, or discard genuine negative validation cases to fill it.

`validation_plan.json` records source paths/fingerprints, held-out geometry, fixed
domain/cell coordinates, forced sample indices, requested/achieved counts,
coverage, and fallback diagnostics. It is reused across epochs and compatible
resume. Inputs retain nominal scale normalization, without augmentation or
jitter. One forward supplies the loss, metrics, and retained preview anchors.
Dice/IoU pool valid voxel counts; multiclass Dice averages foreground classes.
Instance reconstruction metrics and binary Dice loss average per patch.
Pointwise losses weight their actual valid foreground/background support;
multiclass soft Dice pools per-class statistics. Deep supervision uses the same
rules per head before its configured head-weighted total.

Best selection maximizes regular `dice` (binary), `mean_dice` (multiclass), or
`foreground_dice` (instance-friendly). The plateau scheduler uses that same
regular score. Ties preserve the existing overwrite policy but are not reported
as improvements. Regular history remains in `metrics.json`; full diagnostics
are separate in `full_validation_metrics.json` and cannot replace the selector.
Full semantic diagnostics apply configured semantic cleanup; instance object
metrics use configured reconstruction while foreground Dice remains the standard
0.5-threshold metric. No automatic threshold optimization is performed.

`weights_last.pt`, `weights_best.pt`, and `model.pt` are published before optional
diagnostics. An interrupted volumetric epoch uses `weights_pending.pt` or
`weights_cancelled.pt`, retaining an existing completed last checkpoint. Completed
volumetric checkpoints preserve RNG state. Resume requires the matching saved
plan, history, selection contract, and `validation_state.json`; incompatible
legacy full-selector runs fail explicitly. Manual request tokens do not survive
resume. This does not provide exact mid-epoch continuation.

Four enlarged previews normally reuse four anchors and evaluate twelve extra
tiles. Layout is one patch deep and two tiles along each XY axis. 2.5D previews
predict only the actual center plane, not the context stack. Extra tiles never
enter regular metrics. Continuous outputs are blended before reconstruction.
`tile_overlap` and `tile_blending` match inference defaults and may be overridden
for validation; they do not borrow another backend's tiling policy.

The automatic preview working-memory budget is 256 MiB for tiny/medium and
600 MiB for big/large. Estimates include channels, labels, stitching, and
reconstruction workspace; separate source-crop/available-RAM checks also apply.
Previews reduce extent/count or are skipped when necessary. This is not a
whole-process RAM/VRAM cap. Full diagnostics read normalized/resampled crops,
stitch on CPU, and process domains sequentially. Native reconstruction still
requires sizeable work arrays; unsafe full-domain estimates fail the diagnostic
without invalidating regular checkpoints. No large-case performance guarantee
is implied by the small correctness tests.

Compatibility: `light_steps` aliases `minimum_batches`; contradictory explicit
values are rejected. Volumetric `light_every` must be 1. `mode="light"` disables
full validation and conflicts with a positive `full_every`; it never changes
volumetric checkpoint selection. A positive legacy volumetric early-stopping
patience is rejected with instructions to set it to zero. Saved settings resolve
automatic defaults and round-trip through the parser.

Augmentation defaults are preset-aware: `tiny` and `medium` use the balanced
profile, while `big` and `large` use the strong profile. Three-dimensional
affine rotation stays in the high-resolution plane for anisotropic data; blur,
low-resolution simulation, and elastic deformation account for physical axis
spacing. Images use continuous interpolation and labels always use nearest
neighbor interpolation.

Focal loss is available as an additive term for class-imbalanced datasets. It is
off by default; enable it directly through `loss_weights`:

```python
"loss_weights": {
    "dice": 1.0,
    "bce": 1.0,
    "cross_entropy": 1.0,
    "focal": 0.5,
    "boundary": 0.5,
    "boundary_focal": 0.25,
},
"focal_gamma": 2.0,
"focal_alpha": None,
```

For sparse foreground datasets, focal can be enabled automatically from a mask
sample:

```python
"auto_focal": True,
"auto_focal_foreground_threshold": 0.05,
"auto_focal_weight": 0.5,
```

`auto_focal_foreground_threshold` is the foreground-pixel fraction, measured as
foreground pixels divided by total pixels. For instance-friendly models,
`auto_focal_boundary_threshold` and `auto_boundary_focal_weight` do the same for
the boundary/separator channel.

## Callbacks

Training accepts the optional `task` argument as a generic callback target.
Inference accepts both the backward-compatible `task` argument and the
framework-neutral `callback` keyword. Supported forms:

- Appose-style object exposing `update(message=..., current=..., maximum=..., info=...)`.
- Callable accepting one flat event payload dictionary.
- Object exposing `emit(payload)`.
- A list or tuple containing any mix of the above.

Every event payload contains a string `type`, such as `progress`, `preview`,
`inference_progress`, `warning`, `complete`, `cancelled`, or `error`. Inference
progress phases are `inference_start`, repeated `patch_start`/`patch_end`,
`merge_start`, and `inference_end`. A callable can return `False` to request
cooperative cancellation. Cancelled inference raises `InferenceCancelled`
without clearing the loaded-model cache.

### Full-Validation Control and JDLL Handoff

Existing `train(config, task=callback)` calls remain valid. The optional incoming
`control` is separate from outgoing callbacks and supports either a thread-safe
controller or a nonblocking callable returning a request ID, a sequence of IDs,
or `None`. Callback return value `False` remains cancellation, never a command.

```python
from jdll_unet import FullValidationController, train

control = FullValidationController()
# A caller thread can invoke this while train() is running; retries reuse the ID.
control.request_full_validation("request-001")
result = train(config, task=callback, control=control)
```

Java owns transport. For example, create a fresh run-specific signal path and
atomically replace its contents with a new unique token per click. Pass a Python
polling adapter as `control`; the adapter reads that token without deleting a
newer request. Repeated reads of the same token are deduplicated by the backend.
Do not submit another task to the busy Appose worker. The core package has no
Java/Appose dependency and does not interpret private Java transport fields in
the training configuration.

The boundary decision occurs before epoch-end regular validation. Requests
arriving after it wait until the next boundary, including requests made during
previews or a running full pass. Multiple pending requests and a periodic trigger
coalesce into one pass. `full_every=N` first runs at N; a pass started at E moves
the next periodic attempt to E+N (also when that attempt fails). For example,
periodic 5 plus manual 8 gives next 13. Disabled periodic mode stays disabled
after a manual request. No unconditional final pass or extra epoch is created.

Event contract (flat payloads; existing generic messages remain readable):

| Type | Machine-readable information |
| --- | --- |
| `validation_plan` | Plan path, actual batch size, sample/batch counts, capacity, quota/coverage and fallback fields. |
| `validation` | `status=started/progress/completed`, 1-based `epoch`, `current/maximum`, `unit=patches`, losses/metrics and aggregation description on completion. |
| `checkpoint` | `kind=last/best`, `action=saved/overwritten`, epoch, path, scope; best additionally has metric, direction, value and `improved`. |
| `preview` | Existing `preview_path` and `latest_preview_path`, emitted only after publication. `validation_previews` reports actual saved previews/additional tiles. |
| `full_validation` | `run_id`, lifecycle status; an accepted pass additionally carries `pass_id`, `request_ids`, selected epoch, and reason. Progress includes domain/tile counts; completion carries separate full metrics and `next_epoch`. |

Full-control statuses are `ready`, `pending`, `accepted`, `started`, `progress`,
`completed`, `failed`, `cancelled`, and `closed`. `ready.supported=true` advertises
volumetric support. `pending` carries one `request_id`; pass events acknowledge
a list of `request_ids`. Clear the GUI's pending toggle on `accepted`, not by
repeatedly sending its boolean value. A failed optional pass does not close the
controller or stop training. `closed.unserved_request_ids` identifies late
requests for which no further boundary exists. Cancellation takes precedence.
Keep full metrics and validation phase progress separate from regular `val/*`
curves and training iteration progress. UNet's selector remains Dice, unlike
StarDist's validation-loss selector.

Volumetric preview manifests have `format_version=2`, scope, actual preview/tile
counts, and backward-compatible `items` with PNG paths. Each item adds `assets`
for image, target, prediction, validity and optional probabilities. Asset records
contain an absolute NPY path, axes, shape, and dtype. Metadata describes spacing,
source/held-out region, model-grid origin and resampling transform, class-index
mapping, and actual 2.5D context indices (`null` for padded context). Source target
IDs are preserved after any configured annotation preparation; semantic
predictions use classifier indices, and instance
predictions use reconstructed IDs. Never infer Z predictions from context channels.
Epoch-specific assets are completed before atomic manifest/event publication;
the current and previous published epochs are retained for asynchronous viewers.
Java should consume these arrays/metadata rather than reconstruct geometry or
run another inference pass. No Java implementation is included in this change.

## Minimal Inference Example

```python
from jdll_unet.appose_api import infer

result = infer(
    {"model_path": "models/unet/cells/model.pt", "device": "cpu"},
    {"image_path": "datasets/cells/images/example.tif"},
    callback=lambda event: print(event["type"], event.get("phase")),
)
mask = result["outputs"]["mask"]
```

## Validation

```bash
python -m pytest -q
python -m compileall -q jdll_unet tests
python -m ruff check .
python -m build --sdist --wheel
```
