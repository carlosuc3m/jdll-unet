# ISSUE: Gaussian Inference and OOM Fallback

## Problem

Sliding-window inference currently averages overlapping tile logits uniformly.
This can expose tile-edge artifacts, and a tile that exceeds available GPU or
CPU memory fails instead of retrying with a smaller spatial plan.

## Deferred Work

- Blend overlapping logits with a cached Gaussian importance map.
- Keep accumulation memory bounded and support CPU accumulation when required.
- Catch genuine out-of-memory failures without hiding unrelated runtime errors.
- Retry with progressively smaller, architecture-compatible tile sizes.
- Preserve the model planner's per-axis divisibility requirements.
- Report resolved tile size, retries, and accumulation device in inference
  metadata and callbacks.
- Add optional mirrored test-time augmentation only after deterministic Gaussian
  inference is stable.
- Test 2D, 2.5D, isotropic 3D, anisotropic 3D, CPU-only execution, and forced
  low-memory paths.

## Current Scope

Do not make this a prerequisite for the initial spacing-aware 3D architecture,
resampling, training sampler, augmentation, validation, or instance pipeline.
