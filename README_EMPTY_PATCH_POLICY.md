# Empty Image and Empty Patch Training Policy

## Context

JDLL/DeepIcy will be used by bioimage analysts with datasets that may be fully annotated, sparsely annotated, or even made from one large partially annotated image.

In dense segmentation, pixels with value `0` usually mean true background. In sparse annotation, pixels with value `0` may instead mean unknown/unreviewed area. Treating unknown pixels as background can damage training because unannotated target objects become false negatives.

The current UNet sampler loads full-resolution images/masks, samples training patches, and trains on the sampled patch. There is no global resize. Foreground oversampling can center patches on positive labels, but random sampling may still produce completely empty patches. Images whose masks contain no positive labels can also produce all-zero training patches.

## Question To Resolve

Should the UNet trainer skip images and/or patches with no positive labels by default?

This affects:

- dense semantic segmentation;
- instance-friendly segmentation;
- sparse annotation workflows;
- single large-image training;
- false-positive control.

## Why Empty Samples Can Be Harmful

All-zero images or patches are harmful when zero means unknown rather than trusted background.

Failure modes:

- The model learns unannotated objects as background.
- Large mostly-unannotated images dominate training with negative pixels.
- Foreground/background imbalance increases.
- Recall drops, especially for rare or small objects.
- Validation metrics can be misleading if unknown pixels are counted as true negatives.

This is especially relevant for histopathology or microscopy images where only selected regions are annotated.

## When Empty Samples Are Useful

Empty images or patches can be useful when they are true reviewed negatives.

Examples:

- Negative control images.
- Background-only fields intentionally included by the user.
- Rare-object segmentation where absence is meaningful.
- Deployment data often contains no target objects.
- The model tends to hallucinate false positives on debris, tissue texture, autofluorescence, or staining artifacts.

For ordinary positive patches, the model already sees many negative/background pixels around the labeled objects, so pure empty patches are not always necessary.

## Recommended Default

Use a positive-biased policy by default:

- Skip training images whose masks have no positive labels.
- Avoid returning completely empty training patches.
- If a sampled patch is empty, retry sampling a few times.
- Prefer foreground-centered patches when positive labels exist.
- Log how many images/patches were skipped as empty.

This is the safer default for small biological datasets and sparse annotations.

## Recommended Advanced Option

Add an explicit advanced option:

```text
include_empty_patches: false
negative_patch_fraction: 0.0
```

If enabled:

```text
include_empty_patches: true
negative_patch_fraction: 0.05  # or 0.10
```

This would allow a small controlled amount of true-negative training signal without letting empty patches dominate.

The UI wording should avoid jargon:

```text
Use reviewed empty regions as background examples
```

or:

```text
Include background-only patches
```

## Sparse Annotation Mode

Sparse annotation needs explicit valid/ignore-area support.

Recommended dataset convention:

```text
images/
masks/
valid/
```

Where:

- `mask > 0` means annotated target.
- `mask == 0` inside `valid > 0` means reviewed background.
- `valid == 0` means unknown/unreviewed and must not contribute to loss or metrics.

In sparse mode:

- Empty patches should only be allowed if they overlap valid reviewed regions.
- Unknown pixels must be masked out in loss and metrics.
- Foreground sampling should sample positive labels inside valid regions.
- Background sampling should sample only from valid regions.

Without a valid mask, sparse-mode all-zero images should be rejected or skipped, not used as background.

## Implementation Requirements

Dataset-level behavior:

- During dataset inspection, count images with no positive labels.
- Exclude empty images from training by default.
- If all images are empty, fail with a clear error.
- Keep validation behavior explicit: empty validation images are useful only if known reviewed negatives.

Patch-level behavior:

- Add a sampler option to reject empty patches.
- Retry a configurable number of times when an empty patch is sampled.
- If retries fail, sample another image or fall back according to `negative_patch_fraction`.
- Ensure the batch generation does not loop forever on datasets with very few positives.

Logging:

- Log skipped empty training images.
- Log skipped empty validation images, if any.
- Log empty-patch retry/fallback policy.
- Log whether empty patches are disabled or included as hard negatives.

Configuration:

Suggested keys:

```json
{
  "skip_empty_images": true,
  "skip_empty_patches": true,
  "empty_patch_max_retries": 8,
  "include_empty_patches": false,
  "negative_patch_fraction": 0.0,
  "sparse_annotation": "auto",
  "valid_mask_required_for_sparse": true
}
```

## Open Design Questions

- Should empty validation images be skipped by default, or retained only when explicitly marked valid/reviewed?
- Should sparse annotation mode be inferred from the presence of `valid/` masks, or should the Java UI ask the user?
- Should empty images be kept for task detection/statistics but excluded from training?
- Should hard-negative mining be added later as a separate mechanism instead of random empty patch sampling?

## Proposed First Step

Implement the conservative behavior first:

- Skip empty training images by default.
- Retry/avoid empty training patches by default.
- Add config keys for future hard-negative support.
- Add logging.
- Defer full sparse valid-mask support to a separate implementation pass.

---

# Multichannel Label Mask Compatibility

## Java Contract

JDLL treats a multichannel segmentation mask as a single label image by using
its first channel. The Java dataset inspection reports that the mask is
multichannel, includes it in the dataset summary, and validates/counts labels
from channel zero. The original file is linked into the normalized dataset; it
is not rewritten.

`jdll-unet` must apply the same rule when loading the linked mask:

- a 2D mask shaped `Y,X,C` with `C > 1` uses `mask[..., 0]`;
- the discarded channels must not influence task detection, targets, losses, or
  metrics;
- the loader should expose a warning or callback/log message stating that only
  the first channel is used;
- single-channel masks and duplicated RGB label planes must continue to work;
- JPEG masks remain unsupported.

Do not reject a multichannel mask merely because its channels differ. That was
safer as a standalone backend default, but it conflicts with JDLL's established
first-channel behavior and causes Java-approved linked datasets to fail later in
Python.

For true 3D masks, `Z,Y,X` must not be mistaken for `Y,X,C`. Apply channel
collapsing only when the configured dimensions and array shape identify a real
channel axis. Ambiguous shapes should fail clearly rather than silently removing
a spatial axis.

## Tests

- A duplicated RGB mask collapses to its first channel.
- An RGB mask with different channel values also uses its first channel.
- A single-channel mask is unchanged.
- A `Z,Y,X` 3D label volume remains volumetric.
- Ambiguous 3D channel layouts fail with an actionable error.

---

# Fixed Patch Size and Automatic Scale Normalization

## Context

The DeepIcy UI should keep model choices simple: for example `small` and `medium`.
We do not want users to choose large patch sizes manually, and we do not want
large objects to force memory-heavy patch sizes.

However, microscopy objects can have very different pixel sizes across datasets
or even across images. A fixed patch size such as `96x96`, `128x128`, or
`256x256` may be too small relative to the objects being segmented.

If an object is larger than the patch:

- many patches contain only object interior;
- the model sees too little object/background context;
- boundary learning is weaker;
- instance-friendly targets become less reliable;
- validation patches can look acceptable while full-image inference is poor.

Instead of increasing patch size, the preferred approach is to keep patch memory
fixed and normalize image scale automatically.

## Desired Training Behavior

Patch size should stay fixed by model family.

Example:

```text
small  -> fixed patch size
medium -> fixed patch size
```

Before sampling a patch, the trainer should rescale the original image and mask
so that objects appear at a reasonable target size inside the patch.

The trainer should always rescale from the original image/mask, not from a
previously rescaled copy. This prevents cumulative interpolation damage.

## Object Size Estimation

During dataset inspection, estimate object size in pixels.

For instance masks:

- use per-object bounding boxes;
- compute equivalent diameter from area;
- compute bbox max side and min side;
- collect p50, p75, p90 values.

For binary semantic masks:

- compute connected components of `mask > 0`;
- use connected-component bbox/area statistics.

For multiclass semantic masks:

- compute connected components per non-background class;
- optionally aggregate globally and per class.

Ignore tiny/noisy components using existing or configurable small-object filters.

Suggested statistics:

```text
object_diameter_p50
object_diameter_p75
object_diameter_p90
object_bbox_side_p50
object_bbox_side_p90
```

## Target Object Size

Choose a target object size relative to patch size.

Example for 2D:

```text
target_object_diameter_px = 0.25 - 0.35 * patch_side
```

For a `128x128` patch, a useful target may be around `32-44 px`.

Recommended initial heuristic:

```text
target_object_diameter_px = 0.30 * min(patch_size)
```

Alternative:

```text
target_object_bbox_p90 <= 0.70 * min(patch_size)
target_object_diameter_p50 ~= 0.30 * min(patch_size)
```

The precise constants should be easy to configure.

## Base Scale Calculation

Compute a base scale from measured object size:

```text
base_scale = target_object_size_px / measured_object_size_px
```

Example:

```text
patch_size = 128
target_object_size = 38 px
measured_object_diameter_p50 = 190 px
base_scale = 38 / 190 = 0.20
```

This means training should sample patches from an image/mask rescaled to 20% of
the original pixel dimensions.

Use robust statistics, not the largest object.

Recommended first version:

```text
measured_object_size_px = object_diameter_p50
```

Then clamp with p90 safety:

```text
if object_bbox_side_p90 * base_scale > 0.85 * min(patch_size):
    base_scale *= (0.85 * min(patch_size)) / (object_bbox_side_p90 * base_scale)
```

## Scale Augmentation

Do not train at one fixed scale only. Train using a range around the base scale.

Recommended:

```text
scale = base_scale * exp(uniform(log(0.75), log(1.35)))
```

or:

```text
scale = uniform(base_scale * 0.75, base_scale * 1.25)
```

Log-uniform is usually better because scale is multiplicative.

Suggested configurable keys:

```json
{
  "auto_scale": true,
  "target_object_fraction_of_patch": 0.30,
  "scale_jitter_min": 0.75,
  "scale_jitter_max": 1.35,
  "scale_statistic": "diameter_p50",
  "scale_p90_patch_fraction_limit": 0.85
}
```

## Upsampling and Downsampling Guardrails

Scaling must be clamped.

Do not downsample so much that small objects disappear:

```text
small_object_diameter_p10 * scale >= min_visible_object_px
```

Suggested:

```text
min_visible_object_px = 4
preferred_min_visible_object_px = 8
```

Do not upsample excessively:

```text
scale <= max_upscale
```

Suggested:

```text
max_upscale = 2.0
```

If objects are already very small, the trainer should not blindly upscale to the
target size unless it is useful. Excessive upsampling does not create new
information and may overfit interpolation artifacts.

Suggested config:

```json
{
  "min_visible_object_px": 4,
  "preferred_min_visible_object_px": 8,
  "max_downscale": 0.1,
  "max_upscale": 2.0
}
```

## Interpolation Rules

Image and mask must be transformed together.

Use:

```text
image -> bilinear/bicubic interpolation
mask  -> nearest-neighbor interpolation
valid -> nearest-neighbor interpolation
```

For instance-friendly segmentation:

- resize the original label mask with nearest-neighbor;
- generate boundary/distance/separation targets after resizing;
- do not resize already-generated distance maps unless explicitly intended.

For semantic segmentation:

- nearest-neighbor preserves class IDs.

## Patch Sampling Order

Preferred training order:

```text
1. Load original image/mask.
2. Choose sample scale.
3. Resize image/mask/valid from original.
4. Sample patch from resized arrays.
5. Apply spatial/intensity augmentation.
6. Generate or finalize targets.
7. Train.
```

For performance, optional caching can be added later:

- cache resized arrays for a few common scales;
- cache per-image object statistics;
- avoid precomputing all scales on disk.

First version can resize on the fly if performance is acceptable.

## Validation

Validation should be deterministic.

Recommended:

- use `base_scale` without random jitter for validation patches/previews;
- save previews resized back to the original image size when shown in the UI;
- log the validation scale.

Open question:

- Should validation metrics be computed at scaled resolution or original
  resolution?

Suggested first version:

- compute validation on scaled patches for training feedback;
- during preview/inference, resize final predictions back to original size.

## Inference

Inference must be consistent with training.

If the model was trained with automatic scaling, inference should:

```text
1. Estimate or read the expected training scale from config.
2. Resize input image to model scale.
3. Run tiled inference at fixed tile/patch size.
4. Resize prediction/probability/mask back to original image size.
```

For label outputs:

```text
probabilities -> linear interpolation back to original size
class labels  -> nearest-neighbor or argmax after probability resize
instance labels -> postprocess at scaled resolution, then resize/relabel carefully
```

For instance-friendly output, the safest approach is usually:

```text
resize probabilities back -> postprocess at original resolution
```

rather than resizing integer instance labels after postprocessing.

## Sparse Annotation Interaction

Sparse valid masks must be scaled with the image and mask.

Patch sampling must still respect valid regions:

- foreground patches centered on labeled objects inside valid regions;
- background patches only from `valid > 0`;
- invalid pixels ignored in loss and metrics.

Auto-scale should not turn unknown areas into background.

## Logging Requirements

Training logs should include:

```text
Auto scale: enabled
Object size statistic: diameter_p50=...
Object size p90=...
Patch size: ...
Target object size: ...
Base scale: ...
Scale jitter range: ...
Effective scale range: ...
Small-object visibility after scaling: ...
Image interpolation: ...
Mask interpolation: nearest
```

If scaling is clamped, log why:

```text
Base scale clamped from 0.04 to 0.10 because min visible object size would be <4 px.
```

## Proposed First Implementation

Add automatic scale normalization to the Python backend, not the Java preparer.

Java should remain thin:

- optionally report dataset-level object size statistics if already available;
- pass config flags to Python;
- log backend-reported scale settings.

Python should:

- inspect object sizes;
- compute base scale;
- apply scale jitter during training;
- resize image/mask/valid from originals before patch sampling;
- use label-safe interpolation;
- save scale configuration into `config.json`.

The first version can be 2D only. 3D support can follow once the 3D backend is
stable.
