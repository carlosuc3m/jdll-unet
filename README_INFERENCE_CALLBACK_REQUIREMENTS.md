# Whole-Image Inference and Callback Requirements

This document specifies the `jdll-unet` work required for JDLL/DeepIcy to run
UNet inference entirely in Python while preserving the inference progress and
cancellation behavior already used by the StarDist and YOLO integrations.

The Java side will transfer one complete image or volume through shared memory,
call the public inference API once, and retrieve the final label image through
shared memory. Python owns normalization, scale normalization, spacing handling,
tiling, model execution, reconstruction, and postprocessing.

## Scope

Provide a framework-neutral public API:

```python
from jdll_unet.infer import infer

result = infer(config, inputs, callback=callback)
```

Do not require Java to call private preprocessing, tile prediction, or
postprocessing functions. Do not add a runtime or inference-session API.
`load_model()` and its existing model cache remain the model-lifetime mechanism.

The callback must not import, reference, or require Appose. JDLL is responsible
for adapting backend events to its process/task system.

The implementation must support:

- 2D models.
- Fast 3D/2.5D models.
- True 3D models.
- Binary semantic segmentation.
- Multiclass semantic segmentation.
- Instance-friendly segmentation.
- CPU, CUDA, and MPS.

Do not add new dependencies for this work.

## Pipeline Ownership

One call to `infer()` must perform this complete sequence:

1. Load or retrieve the cached model.
2. Load and validate the complete input.
3. Normalize the complete input, never each tile independently.
4. Resolve input spacing and 2.5D context stride.
5. Apply semantic or instance scale normalization when requested.
6. Resolve tile size, overlap, tile starts, and total tile count.
7. Run tiled model prediction.
8. Reconstruct complete raw logits.
9. Restore continuous maps to the original spatial geometry when required.
10. Apply sigmoid or softmax.
11. Run task-specific postprocessing once on the reconstructed maps.
12. Return outputs and metadata.

The Java integration must not need to know how Python tiles or reconstructs the
prediction.

## Existing Functionality to Reuse

Preserve and reuse the current implementations where possible:

- `jdll_unet.infer.load_model`
- `jdll_unet.infer.tiled_predict`
- `jdll_unet.infer._predict_25d`
- `jdll_unet.infer._context_stack`
- `jdll_unet.io.normalize_image`
- `jdll_unet.planning.resolve_context_stride`
- `jdll_unet.planning.resample_image_mask`
- `jdll_unet.planning.restore_continuous_maps`
- `jdll_unet.scale.resize_2d_channels`
- `jdll_unet.postprocess.postprocess_binary`
- `jdll_unet.postprocess.postprocess_multiclass`
- `jdll_unet.postprocess.postprocess_instance`
- `jdll_unet.callbacks.CallbackDispatcher`

Refactor internal signatures as needed, but keep the public `infer()` behavior
backward compatible when no callback is supplied.

## Generic Callback Contract

Use the existing package-owned `CallbackDispatcher(callback)` and emit inference
progress with event type `inference_progress`.

The backend callback protocol is:

```python
def callback(event: dict[str, object]) -> bool | None:
    """Return False to request cooperative cancellation."""
```

This is a plain Python callable. It must work without Appose, JDLL, Java, or any
additional dependency. A caller may also pass `None`.

Every inference progress event must contain:

```python
callbacks.emit(
    "inference_progress",
    message="...",
    current=completed_tiles,
    maximum=total_tiles,
    phase="...",
    patch_index=patch_index,
    total_patches=total_tiles,
)
```

All callback payloads must be JSON serializable. Never put NumPy arrays, Torch
tensors, model objects, or exceptions directly in callback payloads.

### Required Phases

Emit phases in this order:

1. `inference_start`
2. Repeated `patch_start`, `patch_end`
3. `merge_start`
4. `inference_end`

These phase names map directly to JDLL's existing `InferenceProgress.Phase`
values.

### `inference_start`

Emit after preprocessing and scale resolution, when the final prepared shape
and exact number of model tiles are known, but before the first model tile is
executed.

```python
callbacks.emit(
    "inference_progress",
    message=f"Starting UNet inference on {total_tiles} patch(es)",
    current=0,
    maximum=total_tiles,
    phase="inference_start",
    patch_index=0,
    total_patches=total_tiles,
    dimensions=dimensions,
    original_shape=list(original_spatial_shape),
    prepared_shape=list(prepared_spatial_shape),
    tile_size=list(tile_size),
    tile_overlap=overlap,
)
```

`total_tiles` must represent actual model forward passes:

- 2D: number of Y/X tiles.
- True 3D: number of Z/Y/X tiles.
- Fast 3D/2.5D: number of Z output planes multiplied by the number of Y/X
  tiles per plane.

### `patch_start`

Emit immediately before each model forward pass.

`patch_index` is one-based. `current` is the number of already completed tiles,
therefore `patch_index - 1`.

```python
callbacks.emit(
    "inference_progress",
    message=f"UNet inference patch {patch_index}/{total_tiles}",
    current=patch_index - 1,
    maximum=total_tiles,
    phase="patch_start",
    patch_index=patch_index,
    total_patches=total_tiles,
)
```

Optional JSON-safe tile information may be included, such as spatial starts,
but Java must not depend on it.

### `patch_end`

Emit after the model output has been incorporated into the reconstruction
accumulator.

```python
callbacks.emit(
    "inference_progress",
    message=f"Finished UNet inference patch {patch_index}/{total_tiles}",
    current=patch_index,
    maximum=total_tiles,
    phase="patch_end",
    patch_index=patch_index,
    total_patches=total_tiles,
)
```

### `merge_start`

Emit after all model tiles are complete and immediately before final map
restoration and postprocessing. Reconstruction may already have been accumulated
incrementally; this event still marks the transition from tile prediction to
final output generation.

```python
callbacks.emit(
    "inference_progress",
    message="Reconstructing and postprocessing UNet prediction",
    current=total_tiles,
    maximum=total_tiles,
    phase="merge_start",
    patch_index=total_tiles,
    total_patches=total_tiles,
)
```

### `inference_end`

Emit only after final outputs have been produced successfully.

```python
callbacks.emit(
    "inference_progress",
    message="UNet inference finished",
    current=total_tiles,
    maximum=total_tiles,
    phase="inference_end",
    patch_index=total_tiles,
    total_patches=total_tiles,
)
```

Keep the existing `complete` event after `inference_end`. Its metadata must
continue to include at least:

- task.
- model path.
- original input shape.
- prepared/scaled input shape.
- output keys.
- instance scale factor.
- semantic scale factor and comparison.
- input and target spacing.
- 2.5D context stride.
- total patch count.
- tile size and overlap.

## Warnings and Errors

Keep the existing `warning` events for recoverable situations. Warning payloads
must include `type="warning"` and a useful `message`.

The public `infer()` callback path must emit an `error` event before propagating
an unexpected exception. Include:

```python
{
    "type": "error",
    "message": str(exception),
    "error_class": exception.__class__.__name__,
    "stage": current_stage,
}
```

Use a meaningful stage such as `input`, `preprocessing`, `planning`,
`prediction`, `reconstruction`, or `postprocessing`.

Do not emit `inference_end` or `complete` after failure or cancellation.

## Cooperative Cancellation

Cancellation must stop inference promptly without closing the Python service or
removing the cached model. A subsequent inference call must work in the same
process without reloading the model.

Use one `CallbackDispatcher` for the entire inference call and check
`callbacks.cancel_requested()`:

- before preprocessing;
- after preprocessing;
- before every tile;
- after every tile;
- before map restoration;
- before postprocessing;
- before returning outputs.

If cancellation is detected:

1. Stop scheduling new tiles.
2. Release temporary tensors and reconstruction buffers.
3. Do not call `clear_model_cache()`.
4. Emit one `cancelled` event.
5. Do not emit `inference_end`, `complete`, or a generic `error` event.
6. End the current inference call without changing model-cache state.

Recommended payload:

```python
callbacks.emit(
    "cancelled",
    message="UNet inference cancelled",
    current=completed_tiles,
    maximum=total_tiles,
    stage=current_stage,
    completed_patches=completed_tiles,
    total_patches=total_tiles,
)
```

Introduce a dedicated `InferenceCancelled` exception if needed. If used, the
public `infer()` callback path must handle it separately from ordinary errors so
it does not emit an `error` event.

Cancellation is requested only through the generic callback contract: when the
callback returns `False`, `CallbackDispatcher.emit()` returns `False` and the
backend stops at the next defined cancellation point. The backend must not know
how the caller detected cancellation.

## JDLL Adapter, Outside the Backend

JDLL will generate the process-specific adapter. This belongs in JDLL's Python
script generation, not in `jdll-unet`:

```python
def _jdll_unet_callback(event):
    message = str(event.get("message", ""))
    current = event.get("current")
    maximum = event.get("maximum")
    task.update(
        message=message,
        current=current,
        maximum=maximum,
        info=event,
    )
    return not _jdll_task_cancelled(task)

result = jdll_unet.infer.infer(
    config,
    inputs,
    callback=_jdll_unet_callback,
)
```

`_jdll_task_cancelled()` is also JDLL-owned. It may inspect Appose state, a Java
cancellation signal, or another process-specific mechanism. None of those
details may leak into `jdll-unet`.

## Output Contract

Inference outputs are returned by `infer()`; they are not sent through callback
events.

Preserve these output conventions:

### Binary semantic

```python
outputs = {
    "foreground_probability": probability,
    "mask": binary_mask,
    # "labels" when connected-components output is enabled
}
```

### Multiclass semantic

```python
outputs = {
    "probabilities": class_probabilities,
    "mask": class_label_image,
}
```

### Instance-friendly

```python
outputs = {
    "foreground_probability": foreground,
    "boundary_probability": boundary,
    "distance_probability": distance,
    "labels": instance_labels,
}
```

Final label output shapes must be:

- 2D: `Y, X`.
- Fast 3D/2.5D: `Z, Y, X`.
- True 3D: `Z, Y, X`.

The final labels must match the original input spatial shape, not the scaled or
spacing-normalized working shape.

## Resource and Model Lifetime

After success, cancellation, or failure:

- delete temporary Torch tensors and large NumPy reconstruction arrays;
- run Python garbage collection where useful;
- call `torch.cuda.empty_cache()` only on CUDA;
- call `torch.cuda.ipc_collect()` only when supported and safe;
- call `torch.mps.empty_cache()` only on MPS when available;
- never clear the loaded model cache automatically;
- never terminate the Python worker from backend inference code.

Cleanup failures must not hide the original inference result or exception.

## Implementation Guidance

Avoid duplicating callback logic in `tiled_predict()` and `_predict_25d()`.
Create a small internal progress object or callbacks-aware iterator that owns:

- total tile count;
- next one-based tile index;
- `patch_start` emission;
- `patch_end` emission;
- cancellation checks.

For 2.5D, share the same progress counter across all Z planes so patch indices
remain monotonic from `1` through `total_tiles`.

Calculate tile starts once and reuse them both for total-count calculation and
execution. Do not estimate the total separately, because the last aligned tile
may add another start position.

When no callback is supplied, inference output and performance should stay
equivalent to the current implementation except for negligible branching.

## Required Tests

Add tests covering all of the following:

1. A one-tile 2D inference emits the exact phase order:
   `inference_start`, `patch_start`, `patch_end`, `merge_start`,
   `inference_end`, followed by `complete`.
2. Multi-tile 2D inference reports the exact total and one-based monotonic patch
   indices.
3. Fast 3D/2.5D total work equals Z planes multiplied by Y/X tiles.
4. True 3D total work equals the product of Z/Y/X tile-start counts.
5. Callback `current` and `maximum` values follow this document.
6. Inference without a callback returns the same outputs as before.
7. Cancellation before the first tile emits `cancelled` and no completion.
8. Cancellation after an intermediate tile stops before the next model forward.
9. A second inference succeeds after cancellation using the same cached model.
10. Backend failure emits `error`, does not emit completion, and preserves the
    original exception.
11. Every callback payload is JSON serializable.
12. Binary, multiclass, and instance outputs retain their documented keys and
    original-resolution shapes.
13. Instance inference continues to use foreground, boundary, and distance maps
    for final postprocessing.
14. CPU tests are mandatory; CUDA and MPS tests may be conditional on device
    availability.

Use small synthetic models and arrays so the callback test suite remains fast.

## Acceptance Criteria

The work is complete when:

- Java can call the framework-neutral `infer()` once with a whole image or
  volume.
- Python performs complete 2D, 2.5D, or 3D inference internally.
- Java can reproduce its existing inference progress phases by adapting generic
  backend events to its task-update mechanism.
- Cancellation stops between tiles without unloading the model or killing the
  worker.
- Final labels have the original input geometry.
- Existing callers without callbacks remain compatible.
- The full test suite passes without new dependencies.
- Importing and using the callback API works in an environment where Appose and
  JDLL are not installed.
