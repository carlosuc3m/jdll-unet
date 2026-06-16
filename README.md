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
