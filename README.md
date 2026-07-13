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

Available 2D architecture names:

- `tiny-2d`: compact baseline UNet for CPU-friendly training.
- `medium-2d`: wider/deeper baseline UNet for GPUs or longer CPU runs.
- `resenc-tiny-2d`: tiny UNet with residual encoder blocks.
- `resenc-medium-2d`: medium UNet with residual encoder blocks.
- `tiny-3d`: true 3D UNet using `Conv3d` for volumetric TIFF stacks.
- `medium-3d`: wider true 3D UNet for GPU or longer CPU runs.

The `resenc-*` variants keep the UNet encoder-decoder shape but replace encoder
conv blocks with residual blocks for better gradient flow. Deep supervision can
be enabled with `"deep_supervision": true`; the trainer then applies auxiliary
losses to intermediate decoder outputs while inference still uses the primary
full-resolution output.

Convolutional UNet blocks use group normalization by default because it is
stable for the small batches common in biomedical segmentation. Set
`"model_normalization"` to `group`, `instance`, `batch`, or `none` to override
the default. This is separate from the image-intensity `"normalization"` setting.

True 3D models use image tensors shaped `C,Z,Y,X`, masks shaped `Z,Y,X`, and
logits shaped `B,C,Z,Y,X`. Multipage TIFF/OME-TIFF image and label stacks are
loaded as volumes for `tiny-3d` and `medium-3d`; RGB 2D images are rejected for
true 3D models instead of being guessed as volumes.

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
        "architecture": "tiny-2d",
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

## 2D Instance Scale Normalization

`instance_friendly` 2D models normalize each image toward a canonical median
instance diameter by default. Training masks are measured using the median
equivalent diameter of up to 21 reproducibly sampled instances. Border-touching
and tiny instances are excluded by default. Binary masks are split into
connected components; instance-ID masks use their label IDs.

```python
"instance_scale_normalization": {
    "enabled": True,
    "target_object_fraction": 0.25,
    "object_size_measure": "equivalent_diameter",
    "max_instances_per_image": 21,
    "exclude_border_instances": True,
    "min_instance_area": 4,
    "training_scale_jitter": [0.5, 2.0],
    "jitter_distribution": "log_uniform",
    "min_effective_scale": 0.25,
    "max_effective_scale": 4.0,
}
```

The target diameter is `target_object_fraction * min(patch_size)`. Training
draws log-uniform scale jitter and extracts the corresponding crop directly
from the original image before resizing it to the fixed patch size. Validation
uses its reproducibly sampled mask median without jitter. Dataset-derived
measurements are written separately to `dataset_statistics.json`.

Inference requires the approximate median object diameter in native input
pixels. It rescales the image to the model's canonical object size, performs
tiled prediction, restores foreground and boundary probabilities to the
original geometry, and then creates instance labels:

```python
result = infer(
    {"model_path": "models/cells/model.pt", "object_size": 18},
    {"image_path": "images/cells.tif"},
)
```

This normalization currently supports 2D instance models only. Semantic,
2.5D, and 3D scale policies are intentionally unchanged.

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

- `poly`: per-step polynomial decay from `learning_rate` to `min_lr`; this is the default.
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
- Instance-friendly: foreground BCE plus foreground Dice, with an auxiliary boundary BCE term.

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

Training and inference accept the optional `task` argument as a generic
callback target. Supported forms:

- Appose-style object exposing `update(message=..., current=..., maximum=..., info=...)`.
- Callable accepting one flat event payload dictionary.
- Object exposing `emit(payload)`.
- A list or tuple containing any mix of the above.

Every event payload contains a string `type`, such as `progress`, `preview`,
`warning`, `complete`, `cancelled`, or `error`. A callable can return `False`
to request cooperative cancellation during training.

## Minimal Inference Example

```python
from jdll_unet.appose_api import infer

result = infer(
    {"model_path": "models/unet/cells/model.pt", "device": "cpu"},
    {"image_path": "datasets/cells/images/example.tif"},
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
