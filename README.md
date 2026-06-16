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
