# ISSUE: Fixed Training Steps Per Epoch

## Problem

Training currently derives epoch length from the number of dataset items exposed
by the PyTorch `Dataset` and `DataLoader`. This is a poor fit for patch-based
segmentation and becomes especially problematic for 2.5D volumes:

- datasets with few large images or volumes produce too few optimizer steps;
- datasets with many small inputs produce disproportionately long epochs;
- defining each `(volume, center_z)` pair as an item improves coverage but still
  couples optimization length to dataset geometry;
- augmentation diversity depends on dataset size instead of an explicit
  training budget;
- learning-rate schedules currently derive their total step count from the
  resulting loader length.

## Intended Direction

Follow the nnU-Net-style iteration model: configure a fixed number of training
steps per epoch and repeatedly sample random source images/volumes, center
slices, patches, scales, and augmentations until that budget is reached.

Validation should remain deterministic and finite. It should not reuse the
infinite/random training sampler, and its case/slice coverage must be defined
explicitly.

## Required Design Work

- Add a portable `steps_per_epoch` training option with a conservative default.
- Decide whether source images/volumes are sampled uniformly or with weighting
  based on foreground, slice count, or usable annotated area.
- For 2.5D, sample a volume first and then a center Z plane, while preserving the
  empty-patch policy and volume-level scale estimate.
- Define worker-safe, reproducible random-number generation across epochs and
  resumed checkpoints.
- Update progress callbacks, scheduler `total_steps`, checkpoint resume state,
  and cancellation behavior.
- Keep validation sampling deterministic, retain empty validation slices, and
  report exactly which slices or patches contribute to validation metrics.
- Test tiny datasets, highly unequal volume depths, multiple workers, resumed
  training, and deterministic seeded runs.

## Current Scope

Do not change epoch semantics as part of the initial 2.5D context-stack and
scale-normalization implementation. Address this issue as a separate training
sampler change so its optimization and reproducibility effects can be tested in
isolation.
