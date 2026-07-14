# DeepIcy and JDLL-UNet Preset Alignment

## Status

This document defines the accepted preset contract shared by DeepIcy, JDLL, and
`jdll-unet`. The Python backend implements these defaults.

The user-facing model sizes are:

- Small
- Medium
- Big
- Large

DeepIcy must not expose legacy architecture names. Each size maps to a
`resenc-*` architecture for 2D, Fast 3D (2.5D), or True 3D training.

## Design Principles

Model capacity and training context are related but separate:

- Encoder channels and block counts define model capacity.
- Patch size defines spatial context and most activation memory.
- Fast-3D context slices define through-plane context without using 3D
  convolutions.
- True-3D patch depth must remain large enough for useful volumetric context and
  instance scale normalization, but Tiny must remain practical on modest GPUs.
- Preferred patches may be reduced by the memory planner or by source image
  dimensions. They must never be enlarged automatically beyond the preset.
- All spatial patch dimensions should remain compatible with the resolved
  encoder strides. The existing geometry planner may stop downsampling an axis
  before it becomes too small.

## Capacity Presets

| DeepIcy size | Architecture prefix | Encoder channels | Encoder blocks | Reference budget |
| --- | --- | --- | --- | --- |
| Small | `resenc-tiny` | `[16, 32, 64, 128]` | `[1, 2, 2, 2]` | 4 GB |
| Medium | `resenc-medium` | `[24, 48, 96, 192, 320]` | `[1, 2, 2, 2, 2]` | 8 GB |
| Big | `resenc-big` | `[32, 64, 128, 256, 320]` | `[1, 3, 4, 4, 4]` | 16 GB |
| Large | `resenc-large` | `[32, 64, 128, 256, 384, 512]` | `[1, 3, 4, 6, 6, 6]` | 24 GB |

These capacity definitions remain the same for 2D, 2.5D, and 3D. The operator
dimensionality and input geometry change, not the meaning of the size name.

## Proposed Spatial Presets

| DeepIcy option | Backend architecture | Preferred patch | Context slices |
| --- | --- | --- | --- |
| Small | `resenc-tiny-2d` | `[128, 128]` | not applicable |
| Medium | `resenc-medium-2d` | `[256, 256]` | not applicable |
| Big | `resenc-big-2d` | `[384, 384]` | not applicable |
| Large | `resenc-large-2d` | `[512, 512]` | not applicable |
| Small - Fast 3D | `resenc-tiny-2.5d` | `[128, 128]` | 5 |
| Medium - Fast 3D | `resenc-medium-2.5d` | `[256, 256]` | 7 |
| Big - Fast 3D | `resenc-big-2.5d` | `[384, 384]` | 9 |
| Large - Fast 3D | `resenc-large-2.5d` | `[512, 512]` | 11 |
| Small - True 3D | `resenc-tiny-3d` | `[16, 64, 64]` | not applicable |
| Medium - True 3D | `resenc-medium-3d` | `[24, 96, 96]` | not applicable |
| Big - True 3D | `resenc-big-3d` | `[32, 128, 128]` | not applicable |
| Large - True 3D | `resenc-large-3d` | `[48, 160, 160]` | not applicable |

The Fast-3D sequence grows by two planes per tier. A sequence such as
`5, 9, 13, 17` would increase through-plane context too aggressively for
anisotropic microscopy and would often include unrelated structures. Context
sampling must continue using the resolved physical stride, so these values are
the number of real planes sampled rather than the number of interpolated planes.

The True-3D Tiny patch keeps 16 Z planes but reduces XY from the current
`96 x 96` to `64 x 64`. Reducing Z below 16 is not recommended as the default:
with instance target diameter set to one quarter of the smallest physical patch
extent, a shallow isotropic patch would normalize objects to an excessively
small diameter. Larger tiers grow both depth and XY context while remaining
substantially smaller than the previous Big and Large 3D preferred volumes.

## Common Architecture Defaults

Unless a compatible custom configuration overrides them, all presets use:

```json
{
  "block_type": "residual",
  "model_normalization": "group",
  "activation": "relu",
  "dropout": 0.0,
  "decoder_convolutions_per_level": 2,
  "deep_supervision": "preset"
}
```

Deep supervision is disabled for Small and enabled for Medium, Big, and Large.
Custom configuration can override it. The memory planner includes its training
overhead when resolving the patch and microbatch; inference uses only the
primary full-resolution output.

## Batch and Microbatch Defaults

The effective batch remains four. Gradient accumulation supplies the difference
between the effective batch and the device microbatch.

| Preset | CPU microbatch | CUDA/MPS microbatch |
| --- | ---: | ---: |
| Small 2D or Fast 3D | 4 | 4 |
| Medium 2D or Fast 3D | 2 | 4 |
| Big 2D or Fast 3D | 1 | 2 |
| Large 2D or Fast 3D | 1 | 1 |
| Small True 3D | 1 | 2 |
| Medium, Big, or Large True 3D | 1 | 1 |

The current `default_batch_size` implementation gives Big and Large 2D models
the Tiny batch rule. It must be updated as part of this preset alignment.

## Training Defaults

The existing automatic training policy remains shared by all presets:

- AdamW with learning rate `1e-3` and weight decay `1e-5`.
- Polynomial learning-rate schedule.
- Effective batch size four.
- Balanced augmentation for Small and Medium.
- Strong augmentation for Big and Large.
- Empty training images and empty sampled patches skipped by default.
- Foreground oversampling enabled.
- Instance scale normalization enabled for instance-friendly tasks.
- Patch and runtime decisions persisted in `config.json`,
  `dataset_fingerprint.json`, and `model_metadata.json`.

## Backend Requirements

`jdll-unet` should provide one source of truth for these presets:

1. `architecture_defaults()` resolves the capacity definition.
2. `default_patch_size()` resolves the preferred patch above.
3. A preset-aware context resolver returns `5, 7, 9, 11` for Fast 3D when the
   user did not explicitly override `context_slices`.
4. A preset-aware microbatch resolver returns the values above.
5. The memory planner uses the smaller of detected available memory and the
   preset reference budget. It reduces microbatch first, using divisors of the
   effective batch, and then may shrink patches. It records preferred and
   resolved values and reduction reasons. Context count is never changed based
   on hardware.
6. The resolved training plan is emitted through the Appose callback before the
   first epoch and is written to the training log.

Legacy `tiny-*` and `medium-*` checkpoints must remain loadable, but new DeepIcy
training runs use only `resenc-*` architecture names.

## Java Requirements

JDLL and DeepIcy should display only the user-facing names. They pass the exact
backend architecture identifier and leave patch size, context, and microbatch on
automatic defaults unless a compatible custom model configuration overrides
them.

Before a dataset is selected, the UI offers the four 2D sizes. After a volume is
recognized, it offers all four Fast-3D and all four True-3D variants, defaulting
to Small - Fast 3D.

The UI log must show both the selected preset and the backend-resolved plan,
including architecture, dimensions, preferred and resolved patch, context
slices and stride, batch, microbatch, accumulation, spacing, scale policy,
augmentation, skipped empty samples, and steps per epoch.

## Acceptance Tests

- Every one of the twelve architecture identifiers resolves successfully.
- Every preferred patch has the correct dimensionality and positive values.
- Fast-3D context counts are positive odd values and map to the selected tier.
- Model construction and a small synthetic forward pass work for all twelve
  identifiers.
- At least Small 2D, Small Fast 3D, and Small True 3D complete a one-epoch CPU
  smoke training run.
- Big and Large model construction is tested without requiring full-size CPU
  training in CI.
- Resolved microbatch never exceeds the preset value.
- Existing legacy checkpoints continue to load.
- Java-generated configuration selects the same resolved preset as direct
  Python configuration.
