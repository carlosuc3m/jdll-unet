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
patch size, with gradient accumulation targeting effective batch four.

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
patch, and uses gradient accumulation to preserve effective batch four. It then
shrinks the preferred patch only when required by the smaller of detected
available memory and the preset's 4/8/16/24 GB reference budget. Preferred and
resolved decisions are persisted in model metadata and emitted as a
`training_plan` callback before the first epoch.

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

After the initial patch attempt, the sampler retries up to
`empty_patch_max_retries` times. If those attempts are empty and
`include_empty_patches_after_max_retries` is false, sampling continues from the
next training image. If it is true, the final empty patch is used. Training
fails clearly when every training mask is empty.

## Instance Scale Normalization

`instance_friendly` models normalize each image or volume toward a canonical
median instance diameter by default. Training masks use up to 21 reproducibly
sampled instances. Border-touching
and tiny instances are excluded by default. Binary masks are split into
connected components; instance-ID masks use their label IDs.

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

For 2.5D instance models, object identities are canonicalized per volume with
efficient 3D connected components. Disconnected regions sharing an annotation
ID receive fresh in-memory IDs. Up to 21 objects are measured per volume,
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

Patch training uses a deterministic random stream and an optimizer-step budget:

```text
steps_per_epoch = max(250, ceil(10 * training_cases / effective_batch_size))
```

Light patch validation runs every epoch. Full tiled per-case validation runs
every five epochs by default, selects checkpoints, and drives early stopping
after 20 unimproved full validations. Setting `validation.mode` to `light`
disables full-case selection.

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
