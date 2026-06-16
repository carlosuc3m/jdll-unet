# JDLL UNet Backend

This repository contains the first JDLL-owned UNet backend described in
`jdll-unet-backend-plan.md`.

The Python package is named `jdll_unet` and exposes Appose-friendly entry
points:

```python
from jdll_unet.appose_api import train, infer, detect_task
```

The implementation supports lightweight 2D and 2.5D UNet training and
inference for binary semantic, multiclass semantic, and instance-friendly
segmentation datasets laid out as `images/` and `masks/`, or as explicit
`train/images`, `train/masks`, `val/images`, and `val/masks` folders.

## Architectures

Available 2D architecture names:

- `tiny-2d`: compact baseline UNet for CPU-friendly training.
- `medium-2d`: wider/deeper baseline UNet for GPUs or longer CPU runs.
- `resenc-tiny-2d`: tiny UNet with residual encoder blocks.
- `resenc-medium-2d`: medium UNet with residual encoder blocks.

The `resenc-*` variants keep the UNet encoder-decoder shape but replace encoder
conv blocks with residual blocks for better gradient flow. Deep supervision can
be enabled with `"deep_supervision": true`; the trainer then applies auxiliary
losses to intermediate decoder outputs while inference still uses the primary
full-resolution output.

Convolutional UNet blocks use group normalization by default because it is
stable for the small batches common in biomedical segmentation. Set
`"model_normalization"` to `group`, `instance`, `batch`, or `none` to override
the default. This is separate from the image-intensity `"normalization"` setting.

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
