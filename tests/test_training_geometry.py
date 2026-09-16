import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch

from jdll_unet.augment import apply_augmentation, make_augmentation_config, sample_patch
from jdll_unet.config import ArchitectureConfig, TrainingConfig, parse_training_config
from jdll_unet.dataset import make_dataset
from jdll_unet.errors import ConfigError, DatasetError, ModelLoadError
from jdll_unet.finetune import recover_base_learning_rate
from jdll_unet.geometry import (
    assert_disjoint,
    eligible_centers,
    inspect_pair,
    load_domain_image,
    load_domain_mask,
    padding_extents,
    spatial_holdout,
    split_sources,
    with_region,
)
from jdll_unet.io import ImageMaskPair
from jdll_unet.losses import compute_loss
from jdll_unet.metrics import compute_metrics
from jdll_unet.model import build_unet
from jdll_unet.planning import build_dataset_plan, validate_network_shape
from jdll_unet.targets import boundary_target, prepare_target
from jdll_unet.trainer import train


def pair_files(root, shape, name="case", image_axes=None, mask_axes=None, value=1):
    image = np.full(shape, value, dtype=np.float32)
    mask = np.ones(shape, dtype=np.uint16)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    paths = ImageMaskPair(root / "images" / f"{name}.tif", root / "masks" / f"{name}.tif", name)
    axes = "ZYX" if len(shape) == 3 else "YX"
    tifffile.imwrite(paths.image, image, photometric="minisblack", metadata={"axes": image_axes or axes})
    tifffile.imwrite(paths.mask, mask, photometric="minisblack", metadata={"axes": mask_axes or axes})
    return paths


def dataset_for(
    pairs,
    *,
    dimensions="2d",
    training=True,
    samples=100,
    skip_empty=False,
    context=3,
    patch=(8, 8),
    max_empty_plane_fraction=0.20,
):
    return make_dataset(
        pairs,
        "binary_semantic",
        [1],
        None,
        "fast",
        patch,
        False,
        0,
        {"skip_empty_patches": skip_empty},
        training,
        dimensions,
        17,
        sample_count=samples,
        context_slices=context,
        max_empty_plane_fraction=max_empty_plane_fraction,
    )


@pytest.mark.parametrize(
    "length,patch,expected", [(20, 16, (0, 0)), (8, 16, (4, 4)), (4, 16, None), (4, 8, (2, 2)), (4, 12, (4, 4))]
)
def test_padding_numeric_examples(length, patch, expected):
    if expected is None:
        with pytest.raises(DatasetError, match="maximum"):
            padding_extents((length, 16, 16), (patch, 16, 16))
    else:
        assert padding_extents((length, 16, 16), (patch, 16, 16))[0] == expected
        image, mask, valid = sample_patch(
            np.ones((1, length, 16, 16)),
            np.ones((length, 16, 16)),
            (patch, 16, 16),
            np.random.default_rng(1),
            return_validity=True,
        )
        assert image.shape == (1, patch, 16, 16)
        assert valid.sum() == min(length, patch) * 16 * 16


@pytest.mark.parametrize(
    "depth,context,stride,centers",
    [(4, 11, 1, (1, 2)), (4, 9, 1, (0, 1, 2, 3)), (4, 13, 1, ()), (4, 9, 2, ()), (1, 3, 1, ())],
)
def test_context_extent_examples(depth, context, stride, centers):
    assert eligible_centers(depth, context, stride) == centers


def test_rgb_and_grayscale_stack_geometry_and_channel_masks(tmp_path):
    volume = pair_files(tmp_path, (3, 12, 13), "volume")
    resolved, _ = inspect_pair(volume)
    assert resolved.source_kind == "volume"
    assert load_domain_image(resolved).shape == (1, 3, 12, 13)
    tifffile.imwrite(volume.image, np.ones((12, 13, 3), dtype=np.uint8), photometric="rgb")
    tifffile.imwrite(
        volume.mask, np.ones((12, 13, 2), dtype=np.uint8), photometric="minisblack", metadata={"axes": "YXC"}
    )
    with pytest.warns(RuntimeWarning, match="first channel"):
        rgb, _ = inspect_pair(volume)
    assert rgb.source_kind == "image_2d" and rgb.image_channels == 3
    assert load_domain_mask(rgb).shape == (12, 13)


@pytest.mark.parametrize("reverse", [False, True])
def test_volume_plane_mask_mismatch_is_not_reinterpreted(tmp_path, reverse):
    pair = pair_files(tmp_path, (3, 12, 13))
    path = pair.image if reverse else pair.mask
    tifffile.imwrite(path, np.ones((12, 13), dtype=np.uint8))
    with pytest.raises(DatasetError, match="Incompatible"):
        inspect_pair(pair)


def test_singleton_stack_is_equivalent_to_2d(tmp_path):
    pair = pair_files(tmp_path, (1, 12, 13))
    tifffile.imwrite(pair.mask, np.ones((12, 13), dtype=np.uint8))
    resolved, _ = inspect_pair(pair)
    assert resolved.source_kind == "singleton_stack"
    assert load_domain_image(resolved).shape == (1, 12, 13)


def test_ambiguous_and_conflicting_axes_fail(tmp_path):
    pair = pair_files(tmp_path, (2, 12, 13))
    tifffile.imwrite(pair.image, np.ones((2, 2, 12, 13), dtype=np.uint8), photometric="minisblack", metadata={})
    with pytest.raises(DatasetError, match="Ambiguous"):
        inspect_pair(pair)
    tifffile.imwrite(
        pair.image, np.ones((2, 12, 13), dtype=np.uint8), photometric="minisblack", metadata={"axes": "CYX"}
    )
    with pytest.raises(DatasetError, match="Incompatible"):
        inspect_pair(pair)


def test_aliases_do_not_leak_across_splits(tmp_path):
    pair, _ = inspect_pair(pair_files(tmp_path, (16, 16)))
    alias = tmp_path / "alias.tif"
    alias.symlink_to(pair.image)
    with pytest.raises(DatasetError, match="split conflict"):
        assert_disjoint([pair], [replace(pair, image=alias)])
    other, _ = inspect_pair(pair_files(tmp_path, (16, 16), "other"))
    train_pairs, val_pairs = split_sources([pair, replace(pair, image=alias), other], 0.5, 1)
    assert len(train_pairs) == len(val_pairs) == 1
    assert_disjoint(train_pairs, val_pairs)


def test_holdout_domain_enforced_before_transforms_and_statistics(tmp_path):
    original, _ = inspect_pair(pair_files(tmp_path, (64, 64)))

    def feasible(pair, training):
        try:
            padding_extents(pair.domain_shape, (16, 16), 0)
            return True
        except DatasetError:
            return False

    training, validation = spatial_holdout(original, 0.25, 42, feasible)
    assert_disjoint(training, validation)
    assert spatial_holdout(original, 0.25, 42, feasible) == (training, validation)
    image = np.full((64, 64), 3, dtype=np.float32)
    image[tuple(slice(a, b) for a, b in validation[0].region)] = 9000
    tifffile.imwrite(original.image, image, metadata={"axes": "YX"})
    domain = load_domain_image(training[0])
    assert np.all(domain == 3)
    cfg = make_augmentation_config(
        "strong",
        (16, 16),
        True,
        1,
        {
            "brightness_probability": 0,
            "contrast_probability": 0,
            "gamma_probability": 0,
            "noise_probability": 0,
            "shift_probability": 0,
            "affine_probability": 1,
            "elastic_probability": 1,
        },
    )
    for seed in range(10):
        sampled, _, valid = apply_augmentation(
            domain, load_domain_mask(training[0]), cfg, np.random.default_rng(seed), return_validity=True
        )
        np.testing.assert_allclose(sampled[:, valid], 3, atol=1e-5)
    with pytest.raises(DatasetError, match="No feasible"):
        spatial_holdout(original, 0.25, 42, lambda pair, training: False)


def test_spacing_statistics_use_training_only(tmp_path):
    train_pair = pair_files(tmp_path, (8, 16, 16), "train")
    val_pair = pair_files(tmp_path, (8, 16, 16), "val")
    train_pair.image.with_suffix(".json").write_text(json.dumps({"spacing": [2, 1, 1]}))
    val_pair.image.with_suffix(".json").write_text(json.dumps({"spacing": [99, 7, 7]}))
    plan = build_dataset_plan([train_pair], "3d", validation_pairs=[val_pair])
    assert plan.target_spacing == (2, 1, 1)
    assert plan.cases[1].spacing == (99, 7, 7)


@pytest.mark.parametrize(
    "task,channels", [("binary_semantic", 1), ("multiclass_semantic", 3), ("instance_friendly", 3)]
)
def test_padding_predictions_cannot_change_losses_metrics_or_gradients(task, channels):
    mask = np.zeros((16, 16), dtype=np.int64)
    mask[4:8, 4:8] = 1
    valid = np.zeros_like(mask, dtype=bool)
    valid[2:14, 2:14] = True
    arrays = prepare_target(task, mask, label_values=[1, 2], validity=valid)
    target = {key: torch.from_numpy(value[None]) for key, value in arrays.items()}
    torch.manual_seed(1)
    logits = torch.randn(1, channels, 16, 16, requires_grad=True)
    changed = logits.detach().clone()
    changed[:, :, ~torch.from_numpy(valid)] = 100
    weights = {"focal": 0.5, "boundary_focal": 0.5}
    loss, _ = compute_loss(task, logits, target, weights)
    other, _ = compute_loss(task, changed, target, weights)
    torch.testing.assert_close(loss, other)
    assert compute_metrics(task, logits, target) == compute_metrics(task, changed, target)
    loss.backward()
    assert not logits.grad[:, :, ~torch.from_numpy(valid)].any()
    aux = torch.randn(1, channels, 8, 8)
    deep, _ = compute_loss(task, [logits.detach(), aux], target, weights)
    assert torch.isfinite(deep)
    target["valid"].zero_()
    with pytest.raises(ValueError, match="nonempty real support"):
        compute_loss(task, logits, target)


def test_padding_does_not_create_instance_boundary():
    mask = np.pad(np.ones((8, 8), dtype=int), 4)
    valid = np.pad(np.ones((8, 8), dtype=bool), 4)
    assert not boundary_target(mask, validity=valid).any()


@pytest.mark.parametrize("fraction,quota", [(0, 0), (0.20, 5), (0.40, 13)])
def test_actual_epoch_draws_respect_empty_plane_cap(tmp_path, fraction, quota):
    paths = pair_files(tmp_path, (100, 8, 8))
    mask = np.zeros((100, 8, 8), dtype=np.uint8)
    mask[:20] = 1
    tifffile.imwrite(paths.mask, mask, metadata={"axes": "ZYX"}, photometric="minisblack")
    pair, _ = inspect_pair(paths)
    dataset = dataset_for([pair], samples=37, max_empty_plane_fraction=fraction)
    subsets = []
    for epoch in range(4):
        dataset.set_epoch(epoch)
        selected = dataset.sampling_summary[0]
        assert selected["retained_empty_quota"] == quota
        assert selected["empty_plane_draws"] <= np.floor(37 * fraction)
        subsets.append(selected["selected_empty_planes"])
        schedule = dataset.epoch_indices.copy()
        dataset.set_epoch(epoch)
        np.testing.assert_array_equal(schedule, dataset.epoch_indices)
    if quota:
        assert subsets[0] != subsets[1]
    dataset.augmentation.skip_empty_patches = True
    dataset.set_epoch(0)
    assert dataset.sampling_summary[0]["retained_empty_quota"] == 0
    validation = dataset_for([pair], training=False)
    assert len(validation) == 100


def test_context_centers_and_modality_order(tmp_path):
    paths = pair_files(tmp_path, (4, 8, 8))
    image = np.broadcast_to(np.arange(1, 5)[:, None, None], (4, 8, 8)).astype(np.float32)
    tifffile.imwrite(paths.image, image, metadata={"axes": "ZYX"}, photometric="minisblack")
    pair, _ = inspect_pair(paths)
    dataset = dataset_for([pair], dimensions="2.5d", training=False, context=11)
    dataset.normalization = {"type": "none"}
    assert dataset.items == [(0, 1), (0, 2)]
    _, stack, mask = dataset._load_item(0)
    assert stack.shape == (11, 8, 8)
    np.testing.assert_array_equal(stack[:, 0, 0], [0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0])
    np.testing.assert_array_equal(mask, load_domain_mask(pair)[1])


def test_network_constraints_follow_anisotropic_strides_and_normalization():
    arch = ArchitectureConfig(
        depth=3,
        channels=(4, 8, 16),
        strides=((1, 2, 2), (2, 2, 2)),
        kernels=((1, 3, 3), (3, 3, 3), (3, 3, 3)),
        dimensions="3d",
    )
    requirements = validate_network_shape(arch, (2, 8, 8), 1)
    assert requirements["cumulative_downsampling"] == (2, 4, 4)
    with pytest.raises(ConfigError, match="network minimum"):
        validate_network_shape(arch, (1, 8, 8), 1)
    arch.normalization = "instance"
    with pytest.raises(ConfigError, match="normalization"):
        validate_network_shape(arch, (2, 4, 4), 1)


def tiny_source(root, dimensions="2d", context=3):
    root.mkdir()
    arch = ArchitectureConfig(
        name=f"resenc-tiny-{dimensions}",
        dimensions=dimensions,
        depth=2,
        channels=(4, 8),
        encoder_blocks=(1, 1),
        context_slices=context,
        input_channels=context if dimensions == "2.5d" else 1,
        normalization="batch",
    )
    cfg = {
        "format": "jdll-unet",
        "format_version": 1,
        "task": "binary_semantic",
        "architecture_config": asdict(arch),
        "label_values": [1],
        "normalization": {"type": "none"},
        "training": {"starting_point": "scratch", "learning_rate": 0.003, "base_learning_rate": 0.003},
    }
    (root / "config.json").write_text(json.dumps(cfg))
    torch.save(
        {"state_dict": build_unet(arch).state_dict(), "architecture_config": asdict(arch), "model_config": cfg},
        root / "model.pt",
    )
    return root


@pytest.mark.parametrize(
    "dimensions,context,shape,patch",
    [("2d", 3, (4, 24, 24), (16, 16)), ("2.5d", 11, (4, 24, 24), (16, 16)), ("3d", 3, (8, 16, 16), (16, 16, 16))],
)
def test_cpu_training_uses_resolved_geometry_and_repeated_finetune(tmp_path, dimensions, context, shape, patch):
    data = tmp_path / "data"
    pair_files(data, shape, "a")
    pair_files(data, shape, "b")
    pair_files(data, (24, 24), "standalone")
    source = tiny_source(tmp_path / "source", dimensions, context)
    for generation in range(2):
        events = []
        output = tmp_path / f"generation{generation}"
        request = {
            "model_name": "fine",
            "dataset_path": data,
            "output_dir": output,
            "starting_point": "fine_tune",
            "base_model": source,
            "context_slices": "auto",
            "epochs": 1,
            "steps_per_epoch": 1,
            "patch_size": patch,
            "effective_batch_size": 1,
            "validation": {"mode": "full", "full_every": 1, "light_steps": 1},
            "preview_count": 1,
            "task": "binary_semantic",
        }
        if generation:
            request["max_empty_plane_fraction"] = 0.35
            request = TrainingConfig(**request)
        result = train(request, task=events.append)
        cfg = result["config"]
        assert cfg["architecture_config"]["context_slices"] == context
        assert cfg["training"]["base_learning_rate"] == 0.003
        assert cfg["training"]["learning_rate"] == pytest.approx(0.0003)
        assert cfg["training"]["adapted_layers_learning_rate"] is None
        expected_fraction = 0.35 if generation else 0.20
        assert cfg["training"]["max_empty_plane_fraction"] == expected_fraction
        assert cfg["normalization"]["type"] == "none"
        plan = json.loads((output / "dataset_plan.json").read_text())
        assert plan["patch_size"] == list(patch)
        assert plan["sampling_policy"]["max_empty_plane_fraction"] == expected_fraction
        if dimensions == "2.5d":
            assert all(
                pair["eligible_centers"] == [1, 2] for pair in plan["training_domains"] + plan["validation_domains"]
            )
        saved = torch.load(output / "model.pt", weights_only=False)
        assert saved["model_config"] == cfg
        assert any(event["type"] == "dataset_summary" for event in events)
        training_plan = next(event for event in events if event["type"] == "training_plan")
        assert training_plan["max_empty_plane_fraction"] == expected_fraction
        assert json.loads((output / "previews/latest.json").read_text())["items"][0]["source_image"]
        source = output


def test_base_lr_recovery_rejects_corrupt_and_does_not_guess():
    assert recover_base_learning_rate({"training": {"starting_point": "scratch", "learning_rate": 0.007}}) == (
        0.007,
        "legacy_scratch_initial_learning_rate",
    )
    with pytest.warns(RuntimeWarning, match="fallback"):
        assert (
            recover_base_learning_rate({"training": {"learning_rate": 0.005, "adapted_layers_learning_rate": 0.9}})[0]
            == 0.001
        )
    for value in (-1, float("nan"), True):
        with pytest.raises(ModelLoadError, match="Corrupt"):
            recover_base_learning_rate({"training": {"base_learning_rate": value}})


def test_padding_parameter_is_validated(tmp_path):
    with pytest.raises(ConfigError):
        parse_training_config(
            {"model_name": "test", "output_dir": tmp_path, "dataset_path": tmp_path, "max_padding_ratio": -1}
        )


@pytest.mark.parametrize("as_object", [False, True])
@pytest.mark.parametrize("value", [None, False, True, -0.1, 1, 2, float("nan"), float("inf"), "0.2", "invalid", []])
def test_empty_plane_fraction_rejects_invalid_values(tmp_path, as_object, value):
    request = {
        "model_name": "test",
        "output_dir": tmp_path,
        "dataset_path": tmp_path,
        "max_empty_plane_fraction": value,
    }
    if as_object:
        request = TrainingConfig(**request)
    with pytest.raises(ConfigError, match="max_empty_plane_fraction"):
        parse_training_config(request)


@pytest.mark.parametrize("as_object", [False, True])
@pytest.mark.parametrize(
    "options,expected", [({}, 0.20), ({"max_empty_plane_fraction": 0}, 0), ({"max_empty_plane_fraction": 0.35}, 0.35)]
)
def test_empty_plane_fraction_defaults_and_overrides(tmp_path, as_object, options, expected):
    request = {"model_name": "test", "output_dir": tmp_path, "dataset_path": tmp_path, **options}
    if as_object:
        request = TrainingConfig(**request)
    parsed = parse_training_config(request)
    assert parsed.max_empty_plane_fraction == expected
    assert asdict(parsed)["max_empty_plane_fraction"] == expected


@pytest.mark.parametrize("fine_tune", [False, True])
@pytest.mark.parametrize("invalid_2d", [False, True])
@pytest.mark.parametrize("source_count", [1, 2])
def test_2d_rejects_volume_only_datasets_before_splitting(tmp_path, fine_tune, invalid_2d, source_count):
    data = tmp_path / "data"
    for index in range(source_count):
        pair_files(data, (4, 24, 24), f"volume{index}")
    if invalid_2d:
        invalid = pair_files(data, (24, 24), "invalid_2d")
        tifffile.imwrite(invalid.mask, np.ones((20, 24), dtype=np.uint8), metadata={"axes": "YX"})
    request = {"model_name": "volume-only", "dataset_path": data, "output_dir": tmp_path / "out"}
    if fine_tune:
        request.update(starting_point="fine_tune", base_model=tiny_source(tmp_path / "source"))
    events = []
    with pytest.raises(DatasetError, match="at least one valid standalone 2D"):
        train(request, task=events.append)
    assert any(event.get("reason") == "missing_standalone_2d_source" for event in events)
    assert not any(event.get("reason") == "spatial_holdout" for event in events)
    assert not any(event["type"] in {"training_plan", "completed"} for event in events)


@pytest.mark.parametrize("standalone_shape", [(24, 24), (1, 24, 24)])
def test_2d_standalone_requirement_is_dataset_level_not_per_split(tmp_path, standalone_shape):
    data = tmp_path / "data"
    pair_files(data / "train", (4, 24, 24), "volume")
    pair_files(data / "val", standalone_shape, "standalone")
    result = train(
        {
            "model_name": "mixed",
            "dataset_path": data,
            "output_dir": tmp_path / "out",
            "starting_point": "fine_tune",
            "base_model": tiny_source(tmp_path / "source"),
            "epochs": 1,
            "steps_per_epoch": 1,
            "patch_size": [16, 16],
            "effective_batch_size": 1,
            "task": "binary_semantic",
            "preview_count": 0,
            "validation": {"mode": "light", "light_steps": 1},
        }
    )
    plan = json.loads(Path(result["dataset_plan_path"]).read_text())
    assert len(plan["training_domains"][0]["spatial_shape"]) == 3
    assert len(plan["validation_domains"][0]["spatial_shape"]) == 2


@pytest.mark.parametrize(
    "shape,dimensions,context,patch",
    [((48, 48), "2d", 3, (16, 16)), ((4, 48, 48), "2.5d", 11, (16, 16)), ((8, 32, 32), "3d", 3, (16, 16, 16))],
)
def test_single_source_cpu_holdout_and_preview_provenance(tmp_path, shape, dimensions, context, patch):
    data = tmp_path / "single"
    pair_files(data, shape)
    source = tiny_source(tmp_path / "source", dimensions, context)
    output = tmp_path / "result"
    result = train(
        {
            "model_name": "single",
            "dataset_path": data,
            "output_dir": output,
            "starting_point": "fine_tune",
            "base_model": source,
            "epochs": 1,
            "steps_per_epoch": 1,
            "effective_batch_size": 1,
            "patch_size": patch,
            "validation_fraction": 0.25,
            "task": "binary_semantic",
            "preview_count": 1,
            "validation": {"mode": "full", "full_every": 1, "light_steps": 1},
        }
    )
    plan = json.loads((output / "dataset_plan.json").read_text())
    train_domain, val_domain = plan["training_domains"][0], plan["validation_domains"][0]
    assert train_domain["split_origin"] == val_domain["split_origin"] == "spatial_holdout"
    assert any(a[1] <= b[0] or b[1] <= a[0] for a, b in zip(train_domain["region"], val_domain["region"], strict=True))
    preview = json.loads((output / "previews/latest.json").read_text())["items"][0]
    assert preview["region"] == val_domain["region"]
    assert result["metrics"]["full_validation"]["per_case_dice"]


def test_saved_config_resolves_nested_auto_and_scratch_custom_lr(tmp_path):
    data = tmp_path / "auto-data"
    pair_files(data, (24, 24), "a")
    pair_files(data, (24, 24), "b")
    result = train(
        {
            "model_name": "auto",
            "dataset_path": data,
            "output_dir": tmp_path / "auto-model",
            "epochs": 1,
            "steps_per_epoch": 1,
            "normalization": {"type": "auto", "low": "auto"},
            "augmentation": {"flip_probability": "auto"},
            "learning_rate": 0.007,
            "effective_batch_size": 1,
            "preview_count": 0,
            "validation": {"mode": "light", "light_steps": 1},
        }
    )
    config = result["config"]

    def check(value, key=""):
        if isinstance(value, dict):
            for name, item in value.items():
                check(item, name)
        elif isinstance(value, list):
            for item in value:
                check(item, key)
        elif key != "model_name":
            assert value != "auto", key

    check(config)
    assert config["training"]["base_learning_rate"] == config["training"]["learning_rate"] == 0.007
    assert config["training"]["device"] == "cpu"


def test_finetune_generations_with_and_without_adapted_layers(tmp_path):
    source = tiny_source(tmp_path / "source")
    data = tmp_path / "rgb"
    for name in ("a", "b"):
        paths = pair_files(data, (24, 24), name)
        tifffile.imwrite(paths.image, np.ones((24, 24, 3), dtype=np.float32), photometric="rgb")
    for index in range(2):
        result = train(
            {
                "model_name": "adapt",
                "dataset_path": data,
                "output_dir": tmp_path / f"adapt{index}",
                "starting_point": "fine_tune",
                "base_model": source,
                "epochs": 1,
                "steps_per_epoch": 1,
                "patch_size": [16, 16],
                "effective_batch_size": 1,
                "preview_count": 0,
                "validation": {"mode": "light", "light_steps": 1},
            }
        )
        config = result["config"]["training"]
        assert config["base_learning_rate"] == 0.003
        assert config["learning_rate"] == pytest.approx(0.0003)
        assert config["adapted_layers_learning_rate"] == (0.003 if index == 0 else None)
        checkpoint = torch.load(result["model_path"], weights_only=False)
        groups = checkpoint["optimizer_state_dict"]["param_groups"]
        assert len(groups) == (2 if index == 0 else 1)
        identifiers = [identifier for group in groups for identifier in group["params"]]
        assert len(identifiers) == len(set(identifiers))
        source = Path(result["model_dir"])


def test_legacy_finetune_lineage_recovery(tmp_path):
    parent = tmp_path / "scratch"
    parent.mkdir()
    (parent / "config.json").write_text(json.dumps({"training": {"starting_point": "scratch", "learning_rate": 0.006}}))
    old = {
        "training": {
            "starting_point": "fine_tune",
            "learning_rate": 0.0006,
            "fine_tuning_initialization": {
                "source_model": str(parent),
                "source_learning_rate": 0.006,
                "backbone_learning_rate": 0.0006,
            },
        }
    }
    base, provenance = recover_base_learning_rate(old)
    assert base == 0.006 and provenance.startswith("legacy_source_lineage")


def test_cancellation_during_planning_has_no_success_event(tmp_path):
    data = tmp_path / "data"
    pair_files(data, (16, 16), "a")
    pair_files(data, (16, 16), "b")

    class Callback:
        cancelled = True

        def __init__(self):
            self.events = []

        def __call__(self, event):
            self.events.append(event)

    callback = Callback()
    result = train({"model_name": "cancel", "dataset_path": data, "output_dir": tmp_path / "model"}, task=callback)
    assert result["cancelled"]
    assert [event["type"] for event in callback.events] == ["cancelled"]


def test_post_filter_validation_repair_and_fixed_shape_rejection(tmp_path):
    data = tmp_path / "data"
    for name in ("a", "b"):
        pair_files(data / "train", (8, 16, 16), name)
    pair_files(data / "train", (4, 16, 16), "too_shallow")
    pair_files(data / "val", (16, 16), "incompatible")
    source = tiny_source(tmp_path / "source", "3d")
    events = []
    result = train(
        {
            "model_name": "repair",
            "dataset_path": data,
            "output_dir": tmp_path / "out",
            "starting_point": "fine_tune",
            "base_model": source,
            "epochs": 1,
            "steps_per_epoch": 1,
            "patch_size": [16, 16, 16],
            "effective_batch_size": 1,
            "preview_count": 0,
            "validation": {"mode": "light", "light_steps": 1},
        },
        task=events.append,
    )
    plan = json.loads(Path(result["dataset_plan_path"]).read_text())
    assert all(pair["spatial_shape"][0] == 8 for pair in plan["training_domains"] + plan["validation_domains"])
    assert any(event.get("reason") == "split_repaired" for event in events)
    assert any("padding 6+6" in event.get("message", "") for event in events)


def test_small_empty_quota_and_wholly_empty_plane_source(tmp_path):
    paths = pair_files(tmp_path, (9, 8, 8))
    labels = np.zeros((9, 8, 8), dtype=np.uint8)
    labels[:3] = 1
    tifffile.imwrite(paths.mask, labels, metadata={"axes": "ZYX"}, photometric="minisblack")
    pair, _ = inspect_pair(paths)
    dataset = dataset_for([pair], samples=3)
    dataset.set_epoch(1)
    assert dataset.sampling_summary[0]["retained_empty_quota"] == 0
    empty = with_region(pair, ((3, 9), (0, 8), (0, 8)), "provided")
    dataset = dataset_for([empty])
    with pytest.raises(DatasetError, match="No eligible"):
        dataset.set_epoch(1)


def test_explicit_class_metadata_and_validation_only_class(tmp_path):
    data = tmp_path / "data"
    pair_files(data / "train", (24, 24), "a")
    validation = pair_files(data / "val", (24, 24), "b")
    labels = np.full((24, 24), 42, dtype=np.uint16)
    tifffile.imwrite(validation.mask, labels, metadata={"axes": "YX"})
    source = tiny_source(tmp_path / "source")
    request = {
        "model_name": "classes",
        "dataset_path": data,
        "output_dir": tmp_path / "out",
        "starting_point": "fine_tune",
        "base_model": source,
        "task": "multiclass_semantic",
        "patch_size": [16, 16],
        "epochs": 1,
        "steps_per_epoch": 1,
        "effective_batch_size": 1,
        "preview_count": 0,
        "validation": {"mode": "light", "light_steps": 1},
    }
    with pytest.raises(DatasetError, match="unsupported classes"):
        train(request)
    (data / "classes.json").write_text(json.dumps({"1": "first", "42": "second"}))
    result = train(request)
    assert result["config"]["label_values"] == [1, 42]
    assert result["config"]["num_classes"] == 3


def test_configuration_object_preserves_explicit_overrides_and_auto(tmp_path):
    source = tiny_source(tmp_path / "source", "2.5d", 11)
    values = {
        "model_name": "object",
        "dataset_path": tmp_path / "data",
        "output_dir": tmp_path / "out",
        "starting_point": "fine_tune",
        "base_model": source,
        "context_slices": "auto",
    }
    parsed = parse_training_config(values)
    assert parsed.context_slices == "auto"
    assert "model_normalization" not in parsed.request_dict()
    explicit = parse_training_config({**values, "model_normalization": "none"})
    with pytest.raises(ModelLoadError, match="preserve source"):
        train(explicit)


@pytest.mark.parametrize("normalization", ["percentile", "minmax", "zscore", "none"])
def test_context_normalization_reuses_statistics_and_preserves_volume_policy(tmp_path, normalization):
    from jdll_unet.infer import _context_stack
    from jdll_unet.io import normalize_image

    paths = pair_files(tmp_path, (4, 12, 13))
    raw = np.arange(4 * 12 * 13, dtype=np.uint16).reshape(4, 12, 13)
    tifffile.imwrite(paths.image, raw, metadata={"axes": "ZYX"}, photometric="minisblack", compression="deflate")
    pair, _ = inspect_pair(paths)
    dataset = dataset_for([pair], dimensions="2.5d", training=False, context=3)
    dataset.normalization = {"type": normalization}
    reference = normalize_image(raw[None], dataset.normalization)
    for index in range(4):
        _, context, _ = dataset._load_item(index)
        np.testing.assert_allclose(context, _context_stack(reference, index, 3), rtol=1e-6, atol=1e-6)
    assert len(dataset._normalization_statistics) == 1
    assert load_domain_image(pair, reader=dataset.reader, raw=True).dtype == np.uint16
    assert dataset.reader.bytes <= dataset.reader.max_bytes


def test_foreground_lost_in_resampling_is_reported_during_planning(tmp_path):
    data = tmp_path / "data"
    for name in ("a", "b"):
        paths = pair_files(data, (8, 8, 8), name)
        mask = np.zeros((8, 8, 8), dtype=np.uint16)
        mask[1] = 1
        tifffile.imwrite(paths.mask, mask, metadata={"axes": "ZYX"}, photometric="minisblack")
    source = tiny_source(tmp_path / "source", "3d")
    events = []
    with pytest.raises(DatasetError, match="No usable training"):
        train(
            {
                "model_name": "lost",
                "dataset_path": data,
                "output_dir": tmp_path / "out",
                "starting_point": "fine_tune",
                "base_model": source,
                "patch_size": [2, 8, 8],
                "spacing": {"target_spacing": [4, 1, 1]},
                "task": "binary_semantic",
            },
            task=events.append,
        )
    assert any("no foreground remains" in event.get("message", "") for event in events)
