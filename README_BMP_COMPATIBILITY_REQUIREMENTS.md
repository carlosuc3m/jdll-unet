# BMP Dataset Compatibility Requirements

## Context

JDLL accepts `.bmp` files as both segmentation images and masks. BMP is not a
common primary microscopy format; TIFF is substantially more common because it
supports scientific metadata, larger bit depths, stacks, and compression.
Nevertheless, BMP still appears in exported teaching datasets and legacy
computer-vision datasets. Supporting it is inexpensive and avoids a Java/Python
compatibility gap.

## Required Behavior

`jdll-unet` must accept `.bmp` wherever it currently accepts ordinary 2D PNG
files:

- `.bmp` input images;
- `.bmp` integer label masks;
- dataset discovery and image/mask pairing;
- direct loading through `load_image` and `load_mask`.

BMP support is required only for 2D data. Volumetric BMP datasets are not
supported; TIFF remains the required stack format.

Mask values must be preserved exactly. JPEG must remain unsupported for masks
because lossy compression changes label values.

## Implementation Guidance

- Add `.bmp` to `IMAGE_EXTENSIONS` and `MASK_EXTENSIONS`.
- Use the existing `imageio.v3` loading path; do not add a dependency solely for
  BMP.
- Keep the existing dtype, finite-value, integer-label, shape, and channel
  validation after loading.
- Do not convert BMP files or silently change their values.

## Tests

Add tests that verify:

- dataset discovery pairs a BMP image with a BMP mask;
- `load_image` returns a channels-first `float32` array;
- `load_mask` preserves integer labels as `int64`;
- mixed image extensions, such as a BMP image and PNG mask, still pair by stem;
- BMP input is rejected clearly when requested as a 3D volume.
