import importlib
import json
import math
from dataclasses import asdict

import numpy as np
import pytest
import tifffile
import torch

from jdll_unet import train
from jdll_unet.augment import AugmentationConfig, apply_augmentation, sample_scale_jitter
from jdll_unet.config import architecture_defaults, default_patch_size, parse_training_config
from jdll_unet.errors import ConfigError
from jdll_unet.image_reading import image_reading_session
from jdll_unet.planning import (
    derive_stage_geometry,
    estimate_training_memory_bytes,
    physical_patch_shape,
    plan_patch_and_microbatch,
)
from jdll_unet.scale import resize_3d_pair_to_shape, target_instance_diameter
from jdll_unet.training_geometry import resolve_training_geometry


@pytest.mark.parametrize("preset,side,diameter", [("tiny", 64, 40), ("medium", 96, 60),
                                               ("big", 128, 80), ("large", 160, 100)])
def test_isotropic_reference_budget_and_diameter(preset, side, diameter):
    reference = default_patch_size(f"resenc-{preset}-3d")
    assert reference == (side,) * 3
    assert physical_patch_shape(reference, (1, 1, 1)) == reference
    assert target_instance_diameter(.625, reference, spacing=(1, 1, 1), reference_patch=reference) == diameter


@pytest.mark.parametrize("spacing", [(4, 1, 1), (1, 4, 1), (1, 1, 4), (4, 2, 1), (1e9, 1, 1), (1, 1e9, 1e9)])
def test_anisotropic_shapes_preserve_budget_and_finite_axes(spacing):
    patch = physical_patch_shape((64,) * 3, spacing)
    assert math.prod(patch) <= 64**3
    assert math.prod(patch) >= 0.9 * 64**3
    assert all(n >= 8 and n % 8 == 0 for n in patch)
    assert physical_patch_shape((64,) * 3, tuple(s * 0.001 for s in spacing)) == patch


def test_extra_xy_context_does_not_enlarge_reference_diameter():
    reference = (64, 64, 64)
    spacing = (4, 1, 1)
    patch = physical_patch_shape(reference, spacing)
    assert patch == (24, 104, 104)
    assert target_instance_diameter(.625, patch, spacing=spacing, reference_patch=reference) == 40
    # A change in physical units changes the physical diameter, not its voxel representation.
    assert target_instance_diameter(.625, patch, spacing=(.4, .1, .1), reference_patch=reference) == 4
    assert target_instance_diameter(.625, (32, 32, 32), spacing=(1, 1, 1), reference_patch=reference) == 20
    # Legacy canonical models still use their actual physical patch extent.
    assert target_instance_diameter(.25, patch, spacing=spacing) == 24


@pytest.mark.parametrize("spacing", [(0, 1, 1), (-1, 1, 1), (float("nan"), 1, 1), (float("inf"), 1, 1)])
def test_invalid_patch_spacing_rejected(spacing):
    with pytest.raises(ConfigError):
        physical_patch_shape((64,) * 3, spacing)


def test_memory_estimation_uses_actual_anisotropic_strides():
    patch, spacing = (24, 104, 104), (4, 1, 1)
    channels, blocks = (16, 32, 64, 128), (1, 2, 2, 2)
    kernels, strides = derive_stage_geometry(patch, spacing, 4)
    isotropic = estimate_training_memory_bytes(patch, channels, blocks)
    actual = estimate_training_memory_bytes(patch, channels, blocks, strides=strides, kernels=kernels)
    assert actual > isotropic
    plan = plan_patch_and_microbatch(patch, patch, channels, blocks, 4, 2,
                                    available_memory_bytes=actual * 2, memory_fraction=.75, spacing=spacing)
    assert plan.resolved_microbatch == 1
    assert plan.resolved_patch == patch


def config(tmp_path, dimensions="3d", options=None):
    return parse_training_config({"model_name": "scale", "output_dir": tmp_path / "model", "dataset_path": tmp_path,
                                  "architecture": f"resenc-tiny-{dimensions}",
                                  **({"instance_scale_normalization": options} if options is not None else {})})


def test_new_3d_defaults_and_legacy_explicit_ranges(tmp_path):
    cfg = config(tmp_path)
    scale = cfg.instance_scale_normalization
    assert scale.target_object_fraction == .625
    assert scale.training_scale_jitter == (.8, 1.25)
    assert scale.aggressive_scale_jitter == (.35, 2)
    assert scale.aggressive_jitter_probability == .3
    assert scale.jitter_distribution == "log_uniform_mixture"
    assert asdict(parse_training_config(cfg).instance_scale_normalization) == asdict(scale)
    legacy = config(tmp_path, options={"target_object_fraction": .25, "training_scale_jitter": [.5, 2],
                                      "mode": "canonical"}).instance_scale_normalization
    assert legacy.jitter_distribution == "log_uniform"
    assert legacy.training_scale_jitter == (.5, 2)
    assert legacy.target_object_fraction == .25
    for dimensions in ("2d", "2.5d"):
        previous = config(tmp_path, dimensions).instance_scale_normalization
        assert previous.training_scale_jitter == (.5, 2)
        assert previous.target_object_fraction == .25
        assert previous.jitter_distribution == "log_uniform"


@pytest.mark.parametrize("options", [
    {"aggressive_jitter_probability": -0.1}, {"aggressive_jitter_probability": 1.1},
    {"aggressive_jitter_probability": float("nan")}, {"aggressive_scale_jitter": [0, 2]},
    {"aggressive_scale_jitter": [2, .35]}, {"aggressive_scale_jitter": [1, float("inf")]},
    {"training_scale_jitter": [1, float("inf")]}, {"jitter_distribution": "unknown"},
])
def test_invalid_mixture_settings(tmp_path, options):
    with pytest.raises(ConfigError):
        config(tmp_path, options=options)


def test_jitter_sampling_probability_ranges_and_reproducibility():
    cfg = AugmentationConfig(training_scale_jitter=(.8, 1.25), jitter_distribution="log_uniform_mixture",
                             aggressive_scale_jitter=(.35, 2), aggressive_jitter_probability=.3)
    rng = np.random.default_rng(45)
    reference = np.random.default_rng(45)
    aggressive_count = 0
    for _ in range(10000):
        aggressive = reference.random() < .3
        aggressive_count += aggressive
        lo, hi = cfg.aggressive_scale_jitter if aggressive else cfg.training_scale_jitter
        expected = np.exp(reference.uniform(np.log(lo), np.log(hi)))
        assert sample_scale_jitter(cfg, rng) == expected
    assert .28 < aggressive_count / 10000 < .32


@pytest.mark.parametrize("instances", [True, False])
@pytest.mark.parametrize("deferred", [True, False])
def test_3d_mixture_has_one_zoom_and_no_validation_jitter(monkeypatch, instances, deferred):
    module = importlib.import_module("jdll_unet.augment")
    from jdll_unet.spatial_augment import SpatialImagePlan

    cfg = AugmentationConfig(patch_size=(8, 16, 16), instance_scale_enabled=instances,
                             instance_scale_mode="downsample_only", target_object_diameter_px=8,
                             training_scale_jitter=(.5, .5), aggressive_scale_jitter=(2, 2),
                             jitter_distribution="log_uniform_mixture", aggressive_jitter_probability=0,
                             affine_probability=1, rotation_degrees=(0, 0), scale_range=(.1, 5),
                             flip_probability=0, rotate90_probability=0)
    mask = np.zeros((16, 32, 32), np.uint8)
    mask[4:12, 8:24, 8:24] = 1
    image = mask[None].astype(np.float32)
    observed = []
    original = module._spatial_affine

    def affine(image, mask, rng, degrees, scale_range, *args):
        observed.append(scale_range)
        return original(image, mask, rng, degrees, scale_range, *args)

    monkeypatch.setattr(module, "_spatial_affine", affine)
    plan = SpatialImagePlan() if deferred else None
    _, labels = apply_augmentation(image, mask, cfg, rng=np.random.default_rng(2), object_diameter_px=8,
                                  defer_photometric=deferred, image_plan=plan)
    assert labels.shape == cfg.patch_size
    assert observed == [(1., 1.)]
    assert np.count_nonzero(labels) == 4 * 8 * 8
    if deferred:
        assert plan.resize_shape == cfg.patch_size
    # Validation never draws either branch and never applies affine augmentation.
    monkeypatch.setattr(module, "sample_scale_jitter", lambda *_: pytest.fail("Validation drew jitter"))
    apply_augmentation(image, mask, cfg, training=False, object_diameter_px=8)


def test_3d_crop_resize_retains_large_integer_ids():
    mask = np.full((16, 16, 16), 2**40 + 3, np.int64)
    image = np.zeros((1, *mask.shape), np.float32)
    _, resized = resize_3d_pair_to_shape(image, mask, (8, 8, 8))
    assert np.all(resized == 2**40 + 3)
    assert resized.dtype == mask.dtype


def write_dataset(root, shape=(32, 128, 128)):
    labels = np.zeros(shape, np.uint8)
    slices = tuple(slice(n // 4, 3 * n // 4) for n in shape)
    labels[slices] = 1
    for split in ("train", "val"):
        for kind, array in (("images", labels.astype(np.float32)), ("masks", labels)):
            directory = root / split / kind
            directory.mkdir(parents=True)
            tifffile.imwrite(directory / f"{split}.tif", array, metadata={"axes": "ZYX"}, photometric="minisblack")


@pytest.mark.parametrize("explicit_patch", [None, (16, 32, 32)])
def test_complete_geometry_resolves_spacing_before_patch_and_diameter(tmp_path, explicit_patch):
    write_dataset(tmp_path / "data")
    cfg = parse_training_config({"model_name": "geometry", "dataset_path": tmp_path / "data", "output_dir": tmp_path,
                                 "architecture": "resenc-tiny-3d", "task": "instance_friendly",
                                 "spacing": {"default_spacing": [4, 1, 1]},
                                 **({"patch_size": explicit_patch} if explicit_patch else {})})
    with image_reading_session():
        geometry = resolve_training_geometry(cfg, architecture_defaults(cfg.architecture), inherited=False,
                                             device=torch.device("cpu"), available_memory=8 * 1024**3,
                                             emit=lambda *a, **k: None, check_cancel=lambda: None)
    assert geometry.reference_patch == (64, 64, 64)
    assert geometry.memory.preferred_patch == (24, 104, 104)
    assert geometry.memory.resolved_patch == (explicit_patch or (24, 104, 104))
    assert cfg.instance_scale_normalization.mode == "downsample_only"
    assert geometry.to_dict()["patch_planning"]["reference_voxel_budget"] == 64**3


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_training_exports_new_policy_and_physical_reference(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Requires CUDA hardware")
    write_dataset(tmp_path / "data", shape=(8, 24, 24))
    result = train({"model_name": "volume", "dataset_path": tmp_path / "data", "output_dir": tmp_path / "model",
                    "architecture": "resenc-tiny-3d", "task": "instance_friendly", "patch_size": [8, 16, 16], "device": device,
                    "spacing": {"default_spacing": [4, 1, 1]}, "epochs": 1, "steps_per_epoch": 1,
                    "batch_size": 1, "effective_batch_size": 1, "preview_count": 0,
                    "validation": {"minimum_batches": 1, "minimum_samples": 1}})
    exported = result["config"]["training"]
    assert exported["instance_scale_normalization"]["aggressive_jitter_probability"] == .3
    assert exported["augmentation"]["jitter_distribution"] == "log_uniform_mixture"
    assert exported["isotropic_reference_patch_size"] == [64, 64, 64]
    assert exported["target_object_size"] == 10
    assert asdict(parse_training_config(exported).instance_scale_normalization) == exported["instance_scale_normalization"]
    metadata = json.loads((tmp_path / "model/model_metadata.json").read_text())
    assert metadata["instance_scale"]["reference_spacing"] == 1
    assert metadata["instance_scale"]["diameter_reference"] == "isotropic_preset_finest_spacing"


def test_inference_uses_saved_physical_target_without_external_metadata(tmp_path):
    from jdll_unet import infer
    from jdll_unet.config import ArchitectureConfig, write_json
    from jdll_unet.model import build_unet

    arch = ArchitectureConfig(name="resenc-tiny-3d", dimensions="3d", depth=2, channels=(4, 8),
                              encoder_blocks=(1, 1), input_channels=1, output_channels=3)
    saved = {"format": "jdll-unet", "format_version": 1, "task": "instance_friendly",
             "architecture_config": asdict(arch), "normalization": {"type": "none"},
             "training": {"patch_size": [8, 16, 16], "target_object_size": 40,
                          "instance_scale_normalization": {"enabled": True, "mode": "downsample_only",
                                                           "target_object_fraction": .625}}}
    write_json(tmp_path / "config.json", saved)
    torch.save({"state_dict": build_unet(arch).state_dict(), "architecture_config": asdict(arch),
                "model_config": saved}, tmp_path / "model.pt")
    image = np.ones((8, 16, 16), np.float32)
    for diameter, expected in ((20, 1), (80, .5)):
        result = infer({"model_path": str(tmp_path), "object_size": diameter, "device": "cpu"},
                       {"image": image, "spacing": [4, 1, 1]})
        assert result["metadata"]["instance_scale_factor"] == expected
        assert result["outputs"]["labels"].shape == image.shape
