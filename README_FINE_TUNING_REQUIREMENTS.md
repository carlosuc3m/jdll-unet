# Fine-Tuning Compatibility and Learning-Rate Requirements

## Context

JDLL provides a simplified UNet training interface. Training from scratch may
use a built-in preset or a custom `config.json`, including an explicit
`learning_rate`. Fine-tuning is different: the source model defines the
architecture, while the new dataset defines the required input channels and
prediction targets.

Java will send a fine-tuning request containing:

```json
{
  "starting_point": "fine_tune",
  "base_model": "/path/to/source/model",
  "learning_rate": "auto"
}
```

It will intentionally omit `architecture`. `jdll-unet` must own architecture
recovery, checkpoint compatibility, weight adaptation, and learning-rate
resolution.

## Required Behavior

### Training From Scratch

- Preserve the current built-in architecture selection.
- If `learning_rate` is numeric, use it exactly.
- If `learning_rate` is `auto`, use the package scratch default, currently
  `1e-3`.
- Use one learning rate for all parameters.

### Source Model Resolution

For fine-tuning:

1. Resolve `base_model` as either a model directory or checkpoint file.
2. Read the source model's complete `architecture_config` from `config.json` or
   checkpoint metadata.
3. Reconstruct the exact source backbone, including dimensionality, channels,
   depth, blocks, normalization, activation, kernels, strides, context, and
   deep-supervision structure.
4. Determine the target task, class mapping, input channels, and output channels
   from the new dataset.

Do not use the request's default architecture. Do not silently fall back to a
built-in preset when source metadata is missing or invalid.

### Compatible Weight Loading

Load all unchanged tensors exactly. Adapt only the input convolution and output
heads.

Input-convolution adaptation must follow the existing JDLL StarDist rules:

- `1 -> N`: repeat the source kernel across input channels and divide by `N`.
- `N -> 1`: sum the source kernel over its input-channel axis.
- Other `N -> M`: copy the shared leading channels and initialize additional
  target channels to zero.

For 2.5D models, a context-plane change is an input-channel adaptation. Preserve
centered overlapping planes and zero-initialize added outer planes. Apply this
per image channel rather than treating context planes and image channels as an
unordered flat list.

Output adaptation applies to:

- the primary output head;
- every deep-supervision output head.

Use these rules:

- If task semantics, class identities, class order, and output count are
  unchanged, load every head tensor unchanged.
- If the same semantic task has a known class mapping, copy matching output
  channels and initialize new output channels.
- If output semantics change, such as binary semantic to instance-friendly
  segmentation, reinitialize all output heads.
- If class counts match but known class identities or ordering differ, do not
  treat the heads as compatible.

After adaptation, assert that every missing or shape-different tensor belongs to
an explicitly adapted input layer or output head. Raise `ModelLoadError` for
every other mismatch. Do not use unrestricted `strict=False` as the
compatibility policy.

Fine-tuning must reject, with a clear message:

- missing or unrecoverable source architecture metadata;
- disagreement between source configuration and checkpoint tensors;
- incompatible model and dataset dimensionality;
- unsupported source schema or model implementation;
- any unexplained backbone mismatch.

### Fine-Tuning Learning Rates

Read the source model's recorded initial/resolved training learning rate.

With `learning_rate="auto"`:

- preserved backbone parameters use `source_learning_rate * 0.1`;
- adapted or reinitialized input/output parameters use
  `source_learning_rate`;
- if the source learning rate is unavailable, use `1e-4` for the backbone and
  `1e-3` for adapted parameters.

If no input or output adaptation is needed, use only the reduced backbone rate
for the complete model.

Create optimizer parameter groups only when adapted parameters exist. The
learning-rate scheduler must preserve the ratio between parameter groups.
Create a fresh optimizer and scheduler; do not restore optimizer state from the
source checkpoint.

Fine-tuning requests from JDLL do not provide a new explicit learning rate.
Numeric `learning_rate` remains an input for scratch training only.

## Saved Configuration

The new model's `config.json` and checkpoint metadata must contain:

- the complete resolved target `architecture_config`;
- source model and checkpoint paths;
- source learning rate, when available;
- resolved backbone learning rate;
- resolved adapted-layer learning rate, or `null` when no adaptation occurred;
- input-channel adaptation summary;
- output-head adaptation summary;
- copied, adapted, reinitialized, missing, and unexpected tensor names;
- `starting_point`.

This metadata must be JSON serializable and sufficient to explain exactly how
the new model was initialized.

## Callback Contract

Extend the existing `training_plan` event with:

```python
callbacks.emit(
    "training_plan",
    message="UNet training plan resolved",
    starting_point="fine_tune",
    source_model="/path/to/source/model",
    source_architecture="resenc-medium-2d",
    architecture="resenc-medium-2d",
    source_input_channels=1,
    target_input_channels=3,
    source_output_channels=1,
    target_output_channels=3,
    input_adaptation="Adapted input convolution from 1 to 3 channels",
    output_adaptation="Reinitialized primary and deep-supervision heads",
    source_learning_rate=0.001,
    backbone_learning_rate=0.0001,
    adapted_layers_learning_rate=0.001,
)
```

For scratch training, emit the same applicable fields with:

- `starting_point="scratch"`;
- `source_model=null`;
- `source_architecture=null`;
- `backbone_learning_rate` equal to the resolved scratch learning rate;
- `adapted_layers_learning_rate=null`;
- adaptation summaries set to `null`.

Field names and value types are part of the Java integration contract. Omit
large state-dict lists from callbacks; keep complete tensor details in
`config.json`.

## Tests

Add tests covering:

- scratch `auto` learning rate;
- explicit scratch learning rate;
- exact fine-tuning load with unchanged inputs and outputs;
- `1 -> N`, `N -> 1`, and general `N -> M` input adaptation for 2D and 3D;
- centered 2.5D context adaptation;
- primary and deep-supervision head adaptation;
- complete head reinitialization when task semantics change;
- strict rejection of an unrelated backbone mismatch;
- source LR reduction and fallback rates;
- optimizer parameter groups and scheduler ratio preservation;
- saved configuration and `training_plan` callback fields;
- missing, malformed, and checkpoint-inconsistent source metadata.

