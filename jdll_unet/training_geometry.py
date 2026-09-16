"""Resolve one source/domain, architecture and sampling plan before training."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np
import torch

from .config import (
    AUTO,
    ArchitectureConfig,
    TrainingConfig,
    default_batch_size,
    default_deep_supervision,
    default_patch_size,
)
from .dataset import DatasetInfo, inspect_dataset
from .errors import DatasetError
from .geometry import (
    assert_disjoint,
    case_record,
    eligible_centers,
    inspect_sources,
    load_domain_mask,
    padding_extents,
    spatial_holdout,
    split_sources,
    stable_seed,
)
from .io import ImageMaskPair, discover_dataset, read_class_labels
from .planning import (
    DatasetPlan,
    RuntimeMemoryPlan,
    build_dataset_plan,
    derive_stage_geometry,
    plan_patch_and_microbatch,
    resample_mask,
    resolve_context_stride,
    validate_network_shape,
)
from .scale import (
    InstanceSizeEstimate,
    estimate_3d_instance_size,
    estimate_instance_size,
    estimate_volume_instance_size,
)
from .targets import target_output_channels
from .task_detect import detect_task_from_pairs


@dataclass
class TrainingGeometry:
    train: list[ImageMaskPair]
    val: list[ImageMaskPair]
    info: DatasetInfo
    task: str
    spacing: DatasetPlan
    memory: RuntimeMemoryPlan
    architecture: ArchitectureConfig
    context_policy: str
    context_spacing: float | None
    records: list[dict[str, Any]]
    network_shape: dict[str, Any]
    instance_estimates: dict[tuple[str, tuple], tuple[InstanceSizeEstimate | None, int]]
    case_sampling: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "dimensions": self.architecture.dimensions,
            "patch_size": self.memory.resolved_patch,
            "network_shape": self.network_shape,
            "case_sampling": self.case_sampling,
            "spacing": self.spacing.to_dict(),
            "sources": self.records,
            "training_domains": [case_record(pair) for pair in self.train],
            "validation_domains": [case_record(pair) for pair in self.val],
        }


def measure_case_instances(
    pair: ImageMaskPair, cfg: TrainingConfig, dimensions: str, spacing: tuple[float, float, float]
) -> tuple[InstanceSizeEstimate | None, int]:
    options = cfg.instance_scale_normalization
    mask = load_domain_mask(pair, dimensions)
    seed = stable_seed(cfg.seed, 0, str(pair.image.resolve()) + str(pair.region))
    kwargs: dict[str, Any] = {
        "max_instances": options.max_instances_per_image,
        "seed": seed,
        "measure": options.object_size_measure,
    }
    repairs = 0
    if dimensions == "3d":
        estimate, repair = estimate_3d_instance_size(
            mask,
            spacing,
            exclude_border=options.exclude_border_instances,
            min_instance_voxels=options.min_instance_area,
            **kwargs,
        )
        repairs = repair.repaired_components
    elif mask.ndim == 3:
        estimate, repair = estimate_volume_instance_size(
            mask,
            exclude_xy_border=options.exclude_border_instances,
            min_instance_area=options.min_instance_area,
            **kwargs,
        )
        repairs = repair.repaired_components
    else:
        estimate = estimate_instance_size(
            mask, exclude_border=options.exclude_border_instances, min_instance_area=options.min_instance_area, **kwargs
        )
    return estimate, repairs


def resolve_training_geometry(
    cfg: TrainingConfig,
    source_arch: ArchitectureConfig,
    *,
    inherited: bool,
    device: torch.device,
    available_memory: int | None,
    emit: Callable[..., Any],
    check_cancel: Callable[[], None],
) -> TrainingGeometry:
    discovered = discover_dataset(cfg.dataset_path)
    dimensions = source_arch.dimensions
    train, train_records = inspect_sources(discovered.train, dimensions, emit, check_cancel)
    val, val_records = inspect_sources(discovered.val, dimensions, emit, check_cancel)
    records = [dict(record, requested_split="train") for record in train_records] + [
        dict(record, requested_split="val") for record in val_records
    ]
    if dimensions == "2d" and not any(pair.source_kind != "volume" for pair in train + val):
        message = (
            "2D training and fine-tuning require at least one valid standalone 2D image/mask pair "
            "in the supplied dataset (a singleton Z=1 stack also qualifies). Volume-only datasets "
            "require a 2.5D or 3D model; extracted volume planes do not satisfy this requirement."
        )
        emit("warning", message=message, reason="missing_standalone_2d_source")
        raise DatasetError(message)
    assert_disjoint(train, val)
    # Aliases within a split reference one sampling source. Different cases with
    # identical basenames need distinct plan keys, including across explicit splits.
    seen_names: dict[str, str] = {}
    for collection in (train, val):
        unique: dict[str, ImageMaskPair] = {}
        for pair in collection:
            stem = pair.stem
            if stem in seen_names and seen_names[stem] != pair.source_id:
                import hashlib

                stem += "-" + hashlib.sha256(str(pair.image.resolve()).encode()).hexdigest()[:10]
            seen_names[stem] = pair.source_id
            unique.setdefault(pair.source_id, replace(pair, stem=stem))
        collection[:] = unique.values()
    if cfg.skip_empty_images:
        empty = [pair for pair in train if not any(pair.plane_positive_counts)]
        for pair in empty:
            records.append({**case_record(pair), "status": "skipped", "reason": "empty_training_source"})
        if empty:
            emit(
                "warning",
                message=f"Empty training masks: {len(empty)} source(s) skipped.",
                reason="empty_training_source",
                count=len(empty),
            )
        train = [pair for pair in train if any(pair.plane_positive_counts)]
    if not train or not any(any(pair.plane_positive_counts) for pair in train):
        raise DatasetError("All training masks are empty or no usable training sources remain")
    runtime_name = cfg.architecture if cfg.architecture.endswith(dimensions) else f"resenc-tiny-{dimensions}"
    preferred = default_patch_size(runtime_name)
    requested_patch = cfg.patch_size
    batch_cap = default_batch_size(runtime_name, device) if cfg.batch_size == AUTO else int(cfg.batch_size)
    initial_patch = (
        cast(tuple[int, ...], requested_patch)
        if requested_patch != AUTO
        else tuple(
            min(p, s)
            for p, s in zip(
                preferred, train[0].domain_shape if dimensions == "3d" else train[0].domain_shape[-2:], strict=True
            )
        )
    )

    def initial_feasible(pair: ImageMaskPair, training: bool) -> bool:
        try:
            shape = pair.domain_shape if dimensions == "3d" else pair.domain_shape[-2:]
            if dimensions == "3d" and shape[0] <= 1:
                return False
            padding_extents(shape, initial_patch, cfg.max_padding_ratio)
            if dimensions == "2.5d":
                return bool(
                    eligible_centers(
                        pair.domain_shape[0],
                        int(cfg.context_slices),
                        cfg.context.stride if cfg.context.stride_policy == "fixed_stride" else 1,
                        cfg.max_padding_ratio,
                    )
                )
            return True
        except DatasetError:
            return False

    automatic_split = not discovered.explicit_val or not val
    if automatic_split:
        if discovered.explicit_val:
            emit(
                "warning",
                message="Validation became empty after geometry filtering; repairing from eligible training sources.",
                reason="split_repaired",
            )
        if len(train) == 1:
            train, val = spatial_holdout(
                train[0], cfg.validation_fraction, cfg.seed, initial_feasible, check_cancel=check_cancel
            )
            emit(
                "warning",
                message="Created a within-source validation holdout; this is less independent than another specimen.",
                reason="spatial_holdout",
            )
        else:
            train, val = split_sources(train, cfg.validation_fraction, cfg.seed)
    excluded: set[str] = set()
    estimate_cache: dict[tuple, tuple[InstanceSizeEstimate | None, int]] = {}
    failed_holdouts: set[tuple] = set()
    while True:
        check_cancel()
        if not any(any(pair.plane_positive_counts) for pair in train):
            raise DatasetError("All training masks in the assigned training split are empty")
        info = inspect_dataset(train, dimensions)
        detection = detect_task_from_pairs(train, cfg.dataset_path, requested_task=cfg.task, dimensions=dimensions)
        task = str(detection["task"])
        if detection.get("ambiguous") or task not in {"binary_semantic", "multiclass_semantic", "instance_friendly"}:
            raise DatasetError(
                f"Dataset task is ambiguous or unsupported: {detection.get('reason')}; specify the prediction task"
            )
        class_labels = read_class_labels(cfg.dataset_path) if task == "multiclass_semantic" else None
        if class_labels is not None:
            if set(info.label_values) - set(class_labels):
                raise DatasetError("Training labels disagree with the declared class identities")
            info.label_values = class_labels
        labels = [1] if task == "binary_semantic" else info.label_values
        spacing_cfg = cfg.spacing
        spacing_plan = build_dataset_plan(
            train,
            dimensions,
            default_spacing=spacing_cfg.default_spacing,
            known_fraction_threshold=spacing_cfg.known_fraction_threshold,
            target_spacing=spacing_cfg.target_spacing,
            anisotropy_threshold=spacing_cfg.anisotropy_threshold,
            max_upsampling=spacing_cfg.max_upsampling,
            validation_pairs=val,
        )
        spacings = {case.case: case.spacing for case in spacing_plan.cases}
        context_spacing = (
            spacing_plan.context_spacing
            if cfg.context.spacing == AUTO
            else float(cfg.context.spacing)
            if cfg.context.spacing is not None
            else None
        )
        context_policy = cfg.context.stride_policy
        if context_policy == "nearest_physical" and (
            context_spacing is None or not spacing_plan.context_spacing_reliable
        ):
            context_policy = "adjacent"
        planning_shapes = []
        for pair in train:
            shape = np.array(pair.domain_shape if dimensions == "3d" else pair.domain_shape[-2:])
            if dimensions == "3d":
                shape = np.maximum(
                    1, np.rint(shape * np.asarray(spacings[pair.stem]) / np.asarray(spacing_plan.target_spacing))
                ).astype(int)
            planning_shapes.append(shape)
        planning_shape = tuple(int(v) for v in np.median(planning_shapes, axis=0))
        modalities = info.input_channels if cfg.input_channels == AUTO else int(cfg.input_channels)
        if modalities != info.input_channels:
            raise DatasetError(f"Requested {modalities} input modalities but sources have {info.input_channels}")
        input_channels = modalities * int(cfg.context_slices) if dimensions == "2.5d" else modalities
        arch = replace(
            source_arch,
            input_channels=input_channels,
            output_channels=target_output_channels(task, labels),
            context_slices=int(cfg.context_slices),
            deep_supervision=source_arch.deep_supervision
            if inherited
            else default_deep_supervision(cfg.architecture)
            if cfg.deep_supervision == AUTO
            else bool(cfg.deep_supervision),
        )
        if cfg.output_classes != AUTO and task == "multiclass_semantic":
            if int(cfg.output_classes) < arch.output_channels:
                raise DatasetError("output_classes cannot represent all training labels")
            arch.output_channels = int(cfg.output_classes)
        channels = arch.channels or tuple(arch.base_channels * 2**i for i in range(arch.depth))
        blocks = arch.encoder_blocks or (arch.convs_per_level,) * arch.depth
        arch.channels = channels
        arch.encoder_blocks = blocks
        if requested_patch == AUTO:
            memory = plan_patch_and_microbatch(
                preferred,
                planning_shape,
                channels,
                blocks,
                arch.reference_memory_gb,
                min(batch_cap, cfg.effective_batch_size),
                effective_batch_size=cfg.effective_batch_size,
                available_memory_bytes=available_memory,
                memory_fraction=cfg.memory_fraction,
                input_channels=input_channels,
                deep_supervision=arch.deep_supervision,
            )
        else:
            assert isinstance(requested_patch, tuple)
            microbatch = max(
                value
                for value in range(1, min(batch_cap, cfg.effective_batch_size) + 1)
                if cfg.effective_batch_size % value == 0
            )
            budget = (
                min(arch.reference_memory_gb * 1024**3, available_memory)
                if available_memory
                else arch.reference_memory_gb * 1024**3
            )
            memory = RuntimeMemoryPlan(
                preferred,
                requested_patch,
                batch_cap,
                microbatch,
                arch.reference_memory_gb,
                available_memory / 1024**3 if available_memory else None,
                budget * cfg.memory_fraction / 1024**3,
                ("user_patch_override",),
            )
        patch = memory.resolved_patch
        if not inherited:
            arch.kernels, arch.strides = derive_stage_geometry(
                patch,
                cast(tuple[float, ...], spacing_plan.target_spacing) if dimensions == "3d" else (1, 1),
                arch.depth,
                anisotropy_threshold=spacing_cfg.kernel_anisotropy_threshold,
                minimum_feature_map=spacing_cfg.minimum_feature_map_size,
            )
        else:
            ndim = 3 if dimensions == "3d" else 2
            arch.kernels = arch.kernels or ((3,) * ndim,) * arch.depth
            arch.strides = arch.strides or ((2,) * ndim,) * (arch.depth - 1)
        network_shape = validate_network_shape(arch, patch, memory.resolved_microbatch)
        nominal_scales = {}
        estimates = {}
        if task == "instance_friendly" and cfg.instance_scale_normalization.enabled:
            extent = (
                min(p * s for p, s in zip(patch, cast(tuple[float, ...], spacing_plan.target_spacing), strict=True))
                if dimensions == "3d"
                else min(patch)
            )
            for pair in train + val:
                check_cancel()
                case_spacing = spacings.get(pair.stem, (1, 1, 1))
                key = (pair.stem, pair.region, case_spacing)
                if key not in estimate_cache:
                    estimate_cache[key] = measure_case_instances(pair, cfg, dimensions, case_spacing)
                estimates[(pair.stem, pair.region)] = estimate_cache[key]
            known = [
                estimate.median_diameter_px
                for pair in train
                if (estimate := estimates[(pair.stem, pair.region)][0]) is not None
            ]
            if not known:
                raise DatasetError(
                    "Instance scale normalization could not measure valid training instances; check masks or disable border exclusion"
                )
            fallback = float(np.median(np.asarray(known, dtype=float)))
            options = cfg.instance_scale_normalization
            nominal_scales = {
                key: float(
                    np.clip(
                        options.target_object_fraction
                        * extent
                        / (value[0].median_diameter_px if value[0] else fallback),
                        options.min_effective_scale,
                        options.max_effective_scale,
                    )
                )
                for key, value in estimates.items()
            }

        case_sampling: list[dict[str, Any]] = []

        def resolve_case(
            pair: ImageMaskPair,
            training: bool,
            spacings: dict = spacings,
            spacing_plan: DatasetPlan = spacing_plan,
            nominal_scales: dict = nominal_scales,
            patch: tuple = patch,
            context_policy: str = context_policy,
            context_spacing: float | None = context_spacing,
            case_sampling: list = case_sampling,
        ) -> ImageMaskPair:
            shape = np.asarray(pair.domain_shape if dimensions == "3d" else pair.domain_shape[-2:])
            if dimensions == "3d":
                if pair.domain_shape[0] <= 1:
                    raise DatasetError("A 3D sampling domain requires more than one real Z plane")
                shape = np.maximum(
                    1, np.rint(shape * np.asarray(spacings[pair.stem]) / np.asarray(spacing_plan.target_spacing))
                ).astype(int)
            scale = nominal_scales.get((pair.stem, pair.region), 1.0)
            source_patch = tuple(max(1, round(p / scale)) for p in patch)
            pads = padding_extents(tuple(shape), source_patch, cfg.max_padding_ratio)
            centers = None
            stride = None
            if dimensions == "2.5d":
                stride = resolve_context_stride(
                    context_policy,
                    fixed_stride=cfg.context.stride,
                    target_spacing=context_spacing,
                    z_spacing=spacings[pair.stem][0],
                )
                centers = eligible_centers(pair.domain_shape[0], int(cfg.context_slices), stride, cfg.max_padding_ratio)
                if not centers:
                    raise DatasetError(
                        f"{pair.image.name}: no eligible real centers for depth {pair.domain_shape[0]}, context {cfg.context_slices}, stride {stride}"
                    )
            elif dimensions == "2d" and len(pair.domain_shape) == 3:
                centers = tuple(range(pair.domain_shape[0]))
            if training and centers is not None and not any(pair.plane_positive_counts[z] for z in centers):
                raise DatasetError(f"{pair.image.name}: no positive eligible planes to anchor the empty-plane quota")
            if training and dimensions == "3d" and cfg.skip_empty_patches:
                target_spacing = cast(tuple[float, float, float], spacing_plan.target_spacing)
                if not np.allclose(spacings[pair.stem], target_spacing):
                    resampled = resample_mask(load_domain_mask(pair, raw=True), spacings[pair.stem], target_spacing)
                    if not np.any(resampled > 0):
                        raise DatasetError(f"{pair.image.name}: no foreground remains after target-spacing resampling")
            case_sampling.append(
                {
                    "source": pair.stem,
                    "split": "train" if training else "val",
                    "region": pair.region,
                    "real_resampled_shape": tuple(int(v) for v in shape),
                    "source_to_training_scale": scale,
                    "nominal_extraction_shape": source_patch,
                    "spatial_padding": pads,
                    "context_stride": stride,
                    "eligible_centers": centers,
                    "skipped_centers": pair.domain_shape[0] - len(centers) if centers is not None else 0,
                    "context_padding_extents": [
                        (
                            max(0, (int(cfg.context_slices) // 2) * stride - z),
                            max(0, z + (int(cfg.context_slices) // 2) * stride - (pair.domain_shape[0] - 1)),
                        )
                        for z in centers or ()
                    ]
                    if stride is not None
                    else None,
                }
            )
            return replace(pair, eligible_centers=centers)

        filtered = []
        removed = []
        for is_train, cases in ((True, train), (False, val)):
            kept = []
            for pair in cases:
                try:
                    kept.append(resolve_case(pair, is_train))
                except DatasetError as exc:
                    removed.append(pair)
                    record = {
                        **case_record(pair),
                        "status": "skipped",
                        "reason": "sampling_ineligible",
                        "message": str(exc),
                    }
                    records.append(record)
                    emit("warning", **record)
            filtered.append(kept)
        previous_train, previous_val = train, val
        train, val = filtered
        if (
            removed
            and len(previous_train) == len(previous_val) == 1
            and previous_train[0].split_origin == "spatial_holdout"
        ):
            parent = replace(previous_train[0], region=(), eligible_centers=None)
            failed_holdouts.add(previous_train[0].region)
            train, val = spatial_holdout(
                parent,
                cfg.validation_fraction,
                cfg.seed,
                initial_feasible,
                excluded_regions=failed_holdouts,
                check_cancel=check_cancel,
            )
            emit(
                "warning",
                message="Trying another disjoint spatial holdout after final eligibility filtering; refitting the training plan.",
                reason="spatial_holdout_repaired",
            )
            continue
        if not train:
            raise DatasetError("No usable training sources remain after fixed-patch/context/padding checks")
        if removed:
            identities = {pair.source_id for pair in removed}
            if identities <= excluded:
                raise DatasetError("No stable eligible training/validation plan can be resolved")
            excluded |= identities
            if not val:
                if len(train) > 1:
                    train, val = split_sources(train, cfg.validation_fraction, cfg.seed)
                else:

                    def final_feasible(pair: ImageMaskPair, training: bool) -> bool:
                        try:
                            resolve_case(pair, training)
                            return True
                        except DatasetError:
                            return False

                    train, val = spatial_holdout(
                        train[0], cfg.validation_fraction, cfg.seed, final_feasible, check_cancel=check_cancel
                    )
                emit(
                    "warning",
                    message="Repaired validation after final eligibility filtering; refitting training-only statistics.",
                    reason="split_repaired",
                )
            continue
        assert_disjoint(train, val)
        if task == "multiclass_semantic":
            unknown = set().union(*(set(pair.label_values) for pair in val)) - set(labels)
            if unknown:
                raise DatasetError(
                    f"Validation contains unsupported classes {sorted(unknown)} absent from the training task; supply explicit class metadata"
                )
        for pair in val:
            if not any(pair.plane_positive_counts):
                emit(
                    "warning",
                    message=f"Empty validation content retained: {pair.image.name}; foreground performance cannot be measured on this case.",
                    reason="empty_validation_source",
                )
        return TrainingGeometry(
            train,
            val,
            info,
            task,
            spacing_plan,
            memory,
            arch,
            context_policy,
            context_spacing,
            records,
            network_shape,
            estimates,
            case_sampling,
        )
