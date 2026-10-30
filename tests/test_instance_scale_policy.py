import json
from dataclasses import asdict

import numpy as np
import pytest
import tifffile
import torch

from jdll_unet import infer, train
from jdll_unet.augment import AugmentationConfig, EmptyPatchError, apply_augmentation, instance_crop_shape
from jdll_unet.config import ArchitectureConfig, parse_training_config, write_json
from jdll_unet.errors import ConfigError
from jdll_unet.model import build_unet
from jdll_unet.scale import instance_scale_factor, resolve_instance_scale_mode


@pytest.mark.parametrize("dimensions,expected", [
    ("2d", "downsample_only"), ("2.5d", "downsample_only"), ("3d", "downsample_only"),
])
def test_automatic_scale_mode(dimensions, expected):
    assert resolve_instance_scale_mode("auto", dimensions) == expected
    assert resolve_instance_scale_mode("canonical", dimensions) == "canonical"


@pytest.mark.parametrize("diameter,expected", [(8, 1), (32, 1), (64, 0.5), (256, 0.25)])
def test_nominal_scale_only_reduces_large_objects(diameter, expected):
    assert instance_scale_factor(32, diameter, mode="downsample_only") == expected


@pytest.mark.parametrize("diameter,jitter,expected", [(8, 2, 2), (8, 0.5, 0.5), (64, 2, 1), (64, 0.5, 0.25)])
def test_jitter_applies_after_nominal_cap(diameter, jitter, expected):
    assert instance_scale_factor(32, diameter, mode="downsample_only", jitter=jitter) == expected


@pytest.mark.parametrize("mode,expected", [("canonical", (32, 32)), ("downsample_only", (128, 128))])
def test_source_crop_uses_resolved_policy(mode, expected):
    cfg = AugmentationConfig(patch_size=(128, 128), target_object_diameter_px=32, instance_scale_mode=mode)
    assert instance_crop_shape(cfg, (512, 512), 8) == expected


def test_padding_cannot_silently_upsample_nominal_crop():
    cfg = AugmentationConfig(patch_size=(128, 128), target_object_diameter_px=32,
                             instance_scale_mode="downsample_only", max_padding_ratio=0)
    with pytest.raises(EmptyPatchError):
        instance_crop_shape(cfg, (64, 64), 8)
    assert instance_crop_shape(cfg, (64, 64), 8, jitter=2) == (64, 64)


@pytest.mark.parametrize("diameter", [4, 16])
def test_xy_scaling_keeps_context_channels_aligned(diameter):
    yy, xx = np.mgrid[:64, :64]
    image = np.stack([yy + xx + 10 * channel for channel in range(3)]).astype(np.float32)
    mask = np.zeros((64, 64), np.uint8)
    mask[30:34, 30:34] = 1
    cfg = AugmentationConfig(patch_size=(32, 32), instance_scale_enabled=True,
                             target_object_diameter_px=8, instance_scale_mode="downsample_only")
    actual, labels = apply_augmentation(image, mask, cfg, training=False, object_diameter_px=diameter)
    assert actual.shape == (3, 32, 32)
    np.testing.assert_allclose(actual[1] - actual[0], 10)
    np.testing.assert_allclose(actual[2] - actual[0], 20)
    assert np.count_nonzero(labels) == (16 if diameter == 4 else 4)


@pytest.mark.parametrize("options", [{"mode": "invalid"}, {"mode": "downsample_only", "min_effective_scale": 2},
                                    {"mode": "downsample_only", "max_effective_scale": 0.5}])
def test_invalid_scale_policy_rejected(tmp_path, options):
    with pytest.raises(ConfigError):
        parse_training_config({"model_name": "test", "output_dir": tmp_path, "dataset_path": tmp_path,
                               "instance_scale_normalization": options})


@pytest.mark.parametrize("dimensions", ["2d", "2.5d"])
def test_training_planning_and_validation_use_native_scale_for_small_instances(tmp_path, dimensions):
    shape = (64, 64) if dimensions == "2d" else (3, 64, 64)
    mask = np.zeros(shape, np.uint8)
    mask[..., 30:34, 30:34] = 1
    axes = "YX" if dimensions == "2d" else "ZYX"
    for split in ("train", "val"):
        for kind, value in (("images", mask.astype(np.float32)), ("masks", mask)):
            directory = tmp_path / "data" / split / kind
            directory.mkdir(parents=True)
            tifffile.imwrite(directory / f"{split}.tif", value, metadata={"axes": axes}, photometric="minisblack")
    output = tmp_path / "model"
    result = train({
        "model_name": "native", "dataset_path": tmp_path / "data", "output_dir": output,
        "architecture": f"resenc-tiny-{dimensions}", "task": "instance_friendly",
        "patch_size": [32, 32], "context_slices": 3, "epochs": 1, "steps_per_epoch": 1,
        "batch_size": 1, "effective_batch_size": 1, "preview_count": 0,
        "instance_scale_normalization": {"training_scale_jitter": [1, 1]},
        "validation": {"mode": "light", "light_steps": 1, "minimum_batches": 1, "minimum_samples": 2},
    })
    assert result["config"]["training"]["instance_scale_normalization"]["mode"] == "downsample_only"
    plan = json.loads((output / "dataset_plan.json").read_text())
    assert all(case["nominal_extraction_shape"] == [32, 32] for case in plan["case_sampling"])
    statistics = json.loads((output / "dataset_statistics.json").read_text())["instance_scale_statistics"]
    assert statistics["canonical_scale"]["maximum"] == 1
    if dimensions == "2.5d":
        validation = json.loads((output / "validation_plan.json").read_text())
        assert all(domain["shape"] == [64, 64] for domain in validation["domains"])


@pytest.mark.parametrize("dimensions", ["2d", "2.5d", "3d"])
@pytest.mark.parametrize("mode", [None, "canonical", "auto", "downsample_only"])
def test_inference_new_and_legacy_policies_preserve_output_geometry(tmp_path, dimensions, mode):
    patch = (8, 16, 16) if dimensions == "3d" else (16, 16)
    arch = ArchitectureConfig(
        name=f"resenc-tiny-{dimensions}", dimensions=dimensions, depth=2,
        channels=(4, 8), encoder_blocks=(1, 1), context_slices=3,
        input_channels=3 if dimensions == "2.5d" else 1, output_channels=3,
    )
    scale_config = {"enabled": True, "target_object_fraction": 0.25}
    if mode is not None:
        scale_config["mode"] = mode
    model_config = {
        "format": "jdll-unet", "format_version": 1, "task": "instance_friendly",
        "architecture_config": asdict(arch), "input_channels": arch.input_channels,
        "normalization": {"type": "none"},
        "training": {"patch_size": patch, "instance_scale_normalization": scale_config},
    }
    write_json(tmp_path / "config.json", model_config)
    torch.save({"state_dict": build_unet(arch).state_dict(), "model_config": model_config,
                "architecture_config": asdict(arch)}, tmp_path / "model.pt")
    native_shape = patch if dimensions != "2.5d" else (3, *patch)
    image = np.random.default_rng(9).random(native_shape, dtype=np.float32)
    target = min(patch) * 0.25
    resolved = resolve_instance_scale_mode(mode or "canonical", dimensions)
    for diameter, expected in ((target / 2, 1 if resolved == "downsample_only" else 2), (target * 2, 0.5)):
        result = infer({"model_path": str(tmp_path), "device": "cpu", "object_size": diameter}, {"image": image})
        assert result["metadata"]["instance_scale_factor"] == expected
        assert result["outputs"]["labels"].shape == native_shape
        assert result["outputs"]["foreground_probability"].shape == native_shape
