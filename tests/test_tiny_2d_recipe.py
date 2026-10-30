import functools
import json
import math
from dataclasses import asdict, replace

import numpy as np
import pytest
import tifffile
import torch
from torch import nn

from jdll_unet import train, trainer
from jdll_unet.augment import AugmentationConfig, _spatial_affine, apply_augmentation, make_augmentation_config
from jdll_unet.config import (
    ArchitectureConfig,
    LRSchedulerConfig,
    architecture_defaults,
    default_learning_rate,
    parse_training_config,
    resolve_architecture_training_defaults,
)
from jdll_unet.dataset import JdllSegmentationDataset
from jdll_unet.errors import ConfigError
from jdll_unet.infer import load_model
from jdll_unet.io import ImageMaskPair
from jdll_unet.model import build_unet
from jdll_unet.occlusion import plan_cutout
from jdll_unet.schedulers import LearningRateScheduler


def test_r02_architecture_structure_and_parameter_budget():
    model = build_unet("resenc-tiny-2d", input_channels=3, output_channels=3)
    assert model.config.channels == (16, 32, 64, 128, 256)
    assert model.config.encoder_blocks == (1, 2, 2, 2, 2)
    assert sum(p.numel() for p in model.parameters()) == 438243
    assert isinstance(model.encoders[0].body[0], nn.Conv2d)
    assert isinstance(model.decoders[-1].block[0], nn.Conv2d)
    for block in [*model.encoders[1:], *model.decoders[:-1]]:
        spatial = (block.body if hasattr(block, "body") else block.block)[0]
        assert spatial[0].groups == spatial[0].in_channels
        assert spatial[1].kernel_size == (1, 1)
    assert model.head_dropout.p == .05
    assert len([m for m in model.modules() if isinstance(m, nn.Dropout2d)]) == 1
    assert isinstance(model.encoders[-1].activation, nn.PReLU)


@pytest.mark.parametrize("inputs,outputs,deep", [(1, 1, False), (3, 3, False), (4, 5, True)])
def test_r02_channels_odd_shapes_and_deep_supervision(inputs, outputs, deep):
    torch.set_num_threads(2)
    model = build_unet(architecture_defaults("resenc-tiny-2d", inputs, outputs, deep_supervision=deep))
    value = model(torch.randn(2, inputs, 65, 67))
    predictions = value if isinstance(value, list) else [value]
    assert predictions[0].shape == (2, outputs, 65, 67)
    sum(p.square().mean() for p in predictions).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize("name", ["resenc-tiny-2.5d", "resenc-tiny-3d", "resenc-medium-2d", "resenc-big-2d", "resenc-large-2d", "tiny-2d"])
def test_other_presets_keep_dense_relu_and_no_recipe(name, tmp_path):
    arch = architecture_defaults(name)
    assert arch.convolution == "dense"
    assert arch.activation == "relu"
    assert arch.head_dropout == 0
    if "resenc-tiny" in name:
        assert arch.channels == (16, 32, 64, 128)
    cfg = parse_training_config({"model_name": "test", "dataset_path": tmp_path, "output_dir": tmp_path / "out", "architecture": name})
    resolve_architecture_training_defaults(cfg, arch, "instance_friendly")
    assert cfg.lr_scheduler.warmup_fraction == 0
    assert not cfg.augmentation
    assert default_learning_rate("adamw", arch) == .001


def test_new_defaults_and_explicit_overrides_roundtrip(tmp_path):
    arch = architecture_defaults("resenc-tiny-2d")
    request = {"model_name": "test", "dataset_path": tmp_path, "output_dir": tmp_path / "out"}
    cfg = parse_training_config(request)
    resolve_architecture_training_defaults(cfg, arch, "instance_friendly")
    assert default_learning_rate("adamw", arch) == pytest.approx(.0031622776601683794)
    assert cfg.lr_scheduler.warmup_fraction == .04
    assert cfg.augmentation == {"channel_dropout_probability": 0., "cutout_probability": .5,
                                "cutout_max_fraction": .2, "cutout_diameter_fraction": .5}
    override = parse_training_config({**request, "learning_rate": .002,
        "lr_scheduler": {"warmup_fraction": 0},
        "augmentation": {"channel_dropout_probability": .3, "cutout_max_fraction": 0}})
    resolve_architecture_training_defaults(override, arch, "instance_friendly")
    assert override.learning_rate == .002
    assert override.lr_scheduler.warmup_fraction == 0
    assert override.augmentation["cutout_max_fraction"] == 0
    assert override.augmentation["channel_dropout_probability"] == .3
    scheduler = LearningRateScheduler(torch.optim.AdamW([nn.Parameter(torch.zeros(1))], lr=.003), cfg.lr_scheduler, 100, 10)
    restored = parse_training_config({**request, "lr_scheduler": scheduler.config_dict()})
    assert restored.lr_scheduler == cfg.lr_scheduler


def test_legacy_architecture_metadata_remains_loadable(tmp_path):
    arch = replace(architecture_defaults("resenc-tiny-2d", 3, 3), depth=4, channels=(16, 32, 64, 128),
                   encoder_blocks=(1, 2, 2, 2), activation="relu", convolution="dense", head_dropout=0)
    old = build_unet(arch).eval()
    assert sum(p.numel() for p in old.parameters()) == 491715
    payload = asdict(arch)
    payload.pop("convolution")
    payload.pop("head_dropout")
    checkpoint = tmp_path / "old.pt"
    torch.save({"state_dict": old.state_dict(), "model_config": {"architecture_config": payload}}, checkpoint)
    loaded, _ = load_model(checkpoint)
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        torch.testing.assert_close(old(x), loaded(x), rtol=0, atol=0)
    assert ArchitectureConfig.from_dict(payload).convolution == "dense"


def test_warmup_resume_and_epoch_poly_match_recipe():
    def scheduler():
        optimizer = torch.optim.AdamW([nn.Parameter(torch.zeros(1))], lr=.003)
        return LearningRateScheduler(optimizer, LRSchedulerConfig(warmup_fraction=.04), 1000, 100)
    first = scheduler()
    assert first.warmup_steps == 40
    assert first.current_lr == pytest.approx(.00003)
    for step in range(1, 41):
        first.step_batch()
        if step % 10 == 0:
            first.step_epoch(0)
        assert first.current_lr == pytest.approx(.003 * (.01 + .99 * min(step / 39, 1)))
    for _ in range(10):
        first.step_batch()
    assert first.current_lr == .003
    first.step_epoch(0)
    assert first.current_lr == pytest.approx(.003 * (1 - 10 / 960) ** .9)
    resumed = scheduler()
    resumed.load_state_dict(first.state_dict())
    for _ in range(10):
        first.step_batch()
        resumed.step_batch()
    first.step_epoch(0)
    resumed.step_epoch(0)
    assert first.state_dict() == resumed.state_dict()


@pytest.mark.parametrize("value", [-.1, 1., float("nan"), float("inf")])
def test_bad_warmup_rejected(value):
    with pytest.raises(ConfigError, match="warmup"):
        LearningRateScheduler(torch.optim.AdamW([nn.Parameter(torch.zeros(1))]), LRSchedulerConfig(warmup_fraction=value), 100)


def test_old_scheduler_state_without_warmup_still_loads():
    optimizer = torch.optim.AdamW([nn.Parameter(torch.zeros(1))])
    scheduler = LearningRateScheduler(optimizer, LRSchedulerConfig(warmup_fraction=0), 100)
    saved = scheduler.state_dict()
    saved.pop("warmup_steps")
    saved["config"].pop("warmup_fraction")
    saved["config"].pop("warmup_start_factor")
    scheduler.load_state_dict(saved)


def test_cutout_budget_reproducibility_padding_and_no_inplace_changes():
    valid = np.ones((128, 128), dtype=bool)
    valid[:10] = False
    original = valid.copy()
    options = {"probability": .5, "max_fraction": .2, "diameter_fraction": .5, "max_attempts": 2048}
    occupied = []
    for seed in range(200):
        erase = plan_cutout(valid, 32., np.random.default_rng(seed), **options)
        np.testing.assert_array_equal(erase, plan_cutout(valid, 32., np.random.default_rng(seed), **options))
        assert not np.any(erase & ~valid)
        assert erase.sum() <= math.floor(.2 * valid.sum())
        occupied.append(bool(erase.any()))
    assert .3 < np.mean(occupied) < .6
    np.testing.assert_array_equal(valid, original)


def test_cutout_diameter_tracks_crop_and_affine():
    cfg = AugmentationConfig(patch_size=(64, 64), instance_scale_enabled=True, target_object_diameter_px=32,
                             training_scale_jitter=(1, 1), flip_probability=0, rotate90_probability=0,
                             affine_probability=1, rotation_degrees=(0, 0), scale_range=(2, 2))
    metadata = {}
    apply_augmentation(np.ones((1, 256, 256), np.float32), np.ones((256, 256), np.uint32), cfg,
                       np.random.default_rng(1), object_diameter_px=64, metadata=metadata)
    assert metadata["object_diameter_px"] == 16
    # Rejected geometry must not alter the diameter reference.
    metadata = {"object_diameter_px": 32.}
    _spatial_affine(np.ones((1, 64, 64), np.float32), np.ones((64, 64), np.uint32),
                    np.random.default_rng(1), (0, 0), (10, 10), np.ones((64, 64), bool),
                    max_padding_ratio=0, metadata=metadata)
    assert metadata["object_diameter_px"] == 32


@pytest.mark.parametrize("defer_photo,defer_spatial", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("training", [True, False])
def test_cutout_does_not_change_sampling_labels_or_validation(tmp_path, defer_photo, defer_spatial, training):
    image = np.random.default_rng(1).random((96, 96), dtype=np.float32)
    mask = np.zeros((96, 96), np.uint16)
    mask[20:40, 20:40] = 1
    mask[60:80, 60:80] = 2
    pair = ImageMaskPair(tmp_path / "image.tif", tmp_path / "mask.tif", "sample", spatial_shape=mask.shape)
    tifffile.imwrite(pair.image, image)
    tifffile.imwrite(pair.mask, mask)
    aug = AugmentationConfig(patch_size=(64, 64), skip_empty_patches=False, affine_probability=.5,
                             cutout_probability=1, cutout_max_fraction=.2)

    def dataset(config):
        return JdllSegmentationDataset([pair], "instance_friendly", None, None, config, training,
                                      seed=42, instance_sizes={pair.stem: 20},
                                      sample_count=8 if training else None,
                                      defer_photometric=defer_photo, defer_spatial=defer_spatial)

    baseline, occluded = dataset(replace(aug, cutout_max_fraction=0)), dataset(aug)
    changed_count = 0
    for index in range(8):
        original, expected = baseline[index]
        erased, actual = occluded[index]
        erase = actual.pop("cutout_mask", None)
        assert set(actual) == set(expected)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        if training and defer_spatial:
            torch.testing.assert_close(original.image, erased.image, rtol=0, atol=0)
            original, erased = original.image, erased.image
        if erase is not None:
            assert training and defer_photo
            torch.testing.assert_close(original, erased, rtol=0, atol=0)
            changed = erase
        else:
            changed = original != erased
            assert torch.all(erased[changed] == 0)
        if training:
            assert not torch.any(changed & ~actual["valid"])
            assert changed.sum() <= .2 * actual["valid"].sum()
            changed_count += int(changed.sum())
        else:
            assert not changed.any()
    assert (changed_count > 0) == training


@pytest.mark.parametrize("overrides", [{"cutout_max_fraction": 2}, {"cutout_probability": float("nan")},
                                      {"cutout_diameter_fraction": 0}, {"cutout_max_attempts": 0}])
def test_bad_cutout_configuration_rejected(overrides):
    with pytest.raises(ValueError, match="cutout"):
        make_augmentation_config("balanced", (128, 128), True, .5, overrides)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_tiny_recipe_train_save_reload_and_predict(tmp_path, device):
    torch.set_num_threads(2)
    images, masks = tmp_path / "data/images", tmp_path / "data/masks"
    images.mkdir(parents=True)
    masks.mkdir()
    yy, xx = np.mgrid[:64, :64]
    mask = np.zeros((64, 64), np.uint16)
    mask[(yy - 20)**2 + (xx - 20)**2 < 64] = 1
    mask[(yy - 44)**2 + (xx - 44)**2 < 81] = 2
    for index in range(4):
        tifffile.imwrite(images / f"{index}.tif", (mask > 0).astype(np.float32))
        tifffile.imwrite(masks / f"{index}.tif", mask)
    output = tmp_path / "output"
    result = train({"model_name": "tiny-r02", "dataset_path": images.parent, "output_dir": output, "architecture": "resenc-tiny-2d",
                    "device": device, "task": "instance_friendly", "epochs": 2, "steps_per_epoch": 3,
                    "batch_size": 2, "effective_batch_size": 2, "patch_size": [64, 64], "preview_count": 0,
                    "validation": {"mode": "light", "full_every": 0, "early_stopping_patience": 0}})
    cfg = json.loads((output / "config.json").read_text())
    assert cfg["architecture_config"]["activation"] == "prelu"
    assert cfg["training"]["learning_rate"] == pytest.approx(math.sqrt(1e-5))
    assert cfg["training"]["lr_scheduler"]["warmup_fraction"] == .04
    assert cfg["training"]["augmentation"]["cutout_max_fraction"] == .2
    saved = torch.load(output / "weights_last.pt", weights_only=False, map_location="cpu")
    assert saved["scheduler_state_dict"]["step_count"] == 6
    assert saved["scheduler_state_dict"]["current_lrs"] == [0.]
    assert "rng_state" in saved
    model, _ = load_model(result["model_path"], device="cpu")
    with torch.no_grad():
        prediction = model(torch.rand(1, 1, 64, 64))
    assert prediction.shape == (1, 3, 64, 64)
    assert torch.isfinite(prediction).all()


def test_r02_resume_restores_dropout_and_warmup_exactly(tmp_path, monkeypatch):
    images, masks = tmp_path / "data/images", tmp_path / "data/masks"
    images.mkdir(parents=True)
    masks.mkdir()
    mask = np.zeros((64, 64), np.uint16)
    mask[12:25, 12:25] = 1
    mask[35:48, 35:48] = 2
    for index in range(3):
        tifffile.imwrite(images / f"{index}.tif", (mask > 0).astype(np.float32))
        tifffile.imwrite(masks / f"{index}.tif", mask)
    config = {"model_name": "resume-r02", "dataset_path": images.parent, "architecture": "resenc-tiny-2d",
              "device": "cpu", "task": "instance_friendly", "epochs": 3, "steps_per_epoch": 2,
              "batch_size": 2, "effective_batch_size": 2, "patch_size": [64, 64], "preview_count": 0,
              "validation": {"mode": "light", "full_every": 0, "early_stopping_patience": 0}}
    train({**config, "output_dir": tmp_path / "full"})

    class InterruptedRun(Exception):
        pass

    def interrupt(event):
        if event.get("message") == "UNet validation epoch 1":
            raise InterruptedRun()

    with pytest.raises(InterruptedRun):
        train({**config, "output_dir": tmp_path / "interrupted"}, task=interrupt)
    checkpoint = tmp_path / "interrupted/weights_last.pt"
    monkeypatch.setattr(trainer, "_train", functools.partial(trainer._train, resume_from=checkpoint))
    train({**config, "output_dir": tmp_path / "resumed"})
    expected = torch.load(tmp_path / "full/weights_last.pt", map_location="cpu", weights_only=False)
    actual = torch.load(tmp_path / "resumed/weights_last.pt", map_location="cpu", weights_only=False)
    assert expected["scheduler_state_dict"] == actual["scheduler_state_dict"]
    for name in expected["state_dict"]:
        torch.testing.assert_close(expected["state_dict"][name], actual["state_dict"][name], rtol=0, atol=0)
