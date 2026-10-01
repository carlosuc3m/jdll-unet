from unittest.mock import Mock

import numpy as np
import pytest
import tifffile
import torch
from scipy import ndimage as ndi
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.data import DataLoader

from jdll_unet import augment, spatial_augment, trainer
from jdll_unet.augment import AugmentationConfig, apply_augmentation, apply_tensor_photometric_augmentation
from jdll_unet.spatial_augment import (
    SpatialImageBatch,
    SpatialImagePlan,
    SpatialImageSample,
    _gaussian_blur,
    apply_spatial_image_plan,
    collate_spatial_samples,
)
from jdll_unet.targets import complete_device_targets, prepare_target

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
]


class NoCudaScalarReads(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten._local_scalar_dense.default:
            assert args[0].device.type != "cuda"
        return func(*args, **(kwargs or {}))


def config(shape, operation="all"):
    options = {
        f"{name}_probability": int(operation in {name, "all"})
        for name in ("flip", "rotate90", "affine", "elastic", "lowres", "blur")
    }
    return AugmentationConfig(
        patch_size=shape,
        foreground_oversampling=False,
        skip_empty_patches=False,
        brightness_probability=0,
        shift_probability=0,
        contrast_probability=0,
        gamma_probability=0,
        noise_probability=0,
        channel_dropout_probability=0,
        rotation_degrees=(17, 17),
        scale_range=(1.05, 1.05),
        **options,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "shape,spacing,channels", [((24, 24), None, 1), ((24, 24), (0.5, 0.5), 7), ((5, 24, 24), (4, 1, 1), 2)]
)
@pytest.mark.parametrize("operation", ["flip", "rotate90", "affine", "elastic", "lowres", "blur", "all"])
def test_replay_matches_cpu_images_labels_and_validity(device, shape, spacing, channels, operation):
    image = np.random.default_rng(42).random((channels, *shape), dtype=np.float32)
    mask = np.zeros(shape, dtype=np.int64)
    mask[..., 4:12, 3:15] = 1
    mask[..., 12:21, 12:22] = 2
    cfg = config(shape, operation)
    options = {"cfg": cfg, "spacing": spacing, "defer_photometric": True, "return_validity": True}
    expected, expected_mask, valid = apply_augmentation(image.copy(), mask, rng=np.random.default_rng(7), **options)
    plan = SpatialImagePlan()
    raw, actual_mask, actual_valid = apply_augmentation(
        image.copy(), mask, rng=np.random.default_rng(7), image_plan=plan, **options
    )
    with NoCudaScalarReads():
        actual = apply_spatial_image_plan(torch.from_numpy(raw[None]).to(device), plan)
        actual = apply_tensor_photometric_augmentation(
            actual, torch.from_numpy(actual_valid[None, None]).to(device), cfg
        )
    np.testing.assert_allclose(actual.cpu().numpy()[0], expected, atol=1e-5, rtol=1e-5)
    np.testing.assert_array_equal(actual_mask, expected_mask)
    np.testing.assert_array_equal(actual_valid, valid)
    expected_target = prepare_target("instance_friendly", expected_mask, validity=valid, spacing=spacing)
    actual_target = prepare_target("instance_friendly", actual_mask, validity=actual_valid, spacing=spacing)
    for key in expected_target:
        np.testing.assert_array_equal(actual_target[key], expected_target[key])
    if operation in {"lowres", "all"} and len(shape) == 3:
        assert plan.lowres_shape[0] == shape[0]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape,patch", [((12, 12), (24, 24)), ((2, 12, 12), (4, 24, 24))])
def test_replay_preserves_zero_padding_and_per_sample_plans(device, shape, patch):
    samples = []
    expected = []
    for index, operation in enumerate(("all", "flip", "blur")):
        image = np.full((3, *shape), index + 1, dtype=np.float32)
        mask = np.ones(shape, dtype=np.int64)
        cfg = config(patch, operation)
        plan = SpatialImagePlan()
        raw, labels, valid = apply_augmentation(
            image.copy(),
            mask,
            cfg,
            rng=np.random.default_rng(17),
            defer_photometric=True,
            image_plan=plan,
            return_validity=True,
        )
        reference, _, _ = apply_augmentation(
            image.copy(), mask, cfg, rng=np.random.default_rng(17), defer_photometric=True, return_validity=True
        )
        expected.append(reference)
        target = {
            key: torch.from_numpy(value)
            for key, value in prepare_target("binary_semantic", labels, validity=valid).items()
        }
        samples.append((SpatialImageSample(torch.from_numpy(raw), plan), target))
    loader = DataLoader(
        samples, batch_size=3, num_workers=0, pin_memory=device == "cuda", collate_fn=collate_spatial_samples
    )
    batch, target = next(iter(loader))
    assert isinstance(batch, SpatialImageBatch)
    if device == "cuda":
        assert batch.images.is_pinned()
    with NoCudaScalarReads():
        result = batch.materialize(torch.device(device))
        result = apply_tensor_photometric_augmentation(result, target["valid"].to(device), config(patch))
    np.testing.assert_allclose(result.cpu().numpy(), np.stack(expected), atol=1e-5, rtol=1e-5)
    assert not result.cpu().masked_select(~target["valid"].expand_as(result.cpu())).any()


def test_deferred_images_skip_cpu_image_warp_zoom_and_blur(monkeypatch):
    image = np.ones((7, 24, 24), dtype=np.float32)
    mask = np.ones((24, 24), dtype=np.int64)
    grid_sample = Mock(wraps=augment.F.grid_sample)
    coordinates = Mock(wraps=ndi.map_coordinates)
    gaussian = Mock(wraps=ndi.gaussian_filter)
    zoom = Mock(side_effect=AssertionError("CPU image zoom was called"))
    monkeypatch.setattr(augment.F, "grid_sample", grid_sample)
    monkeypatch.setattr(ndi, "map_coordinates", coordinates)
    monkeypatch.setattr(ndi, "gaussian_filter", gaussian)
    monkeypatch.setattr(ndi, "zoom", zoom)
    plan = SpatialImagePlan()
    apply_augmentation(
        image, mask, config(mask.shape), rng=np.random.default_rng(7), image_plan=plan, defer_photometric=True
    )
    assert grid_sample.call_count == 2  # Labels and validity only.
    assert all(call.args[0].shape[1] == 1 for call in grid_sample.call_args_list)
    assert coordinates.call_count == 2  # Labels and validity only.
    assert gaussian.call_count == mask.ndim  # Elastic coordinate fields only.
    assert plan.blur_sigmas and plan.lowres_shape


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(1, 3), (1, 2, 3)])
def test_gaussian_replay_handles_short_and_singleton_axes(device, shape):
    image = np.arange(2 * np.prod(shape), dtype=np.float32).reshape(2, *shape)
    sigmas = (2.0,) * len(shape)
    expected = np.stack([ndi.gaussian_filter(channel, sigmas) for channel in image])
    result = _gaussian_blur(torch.from_numpy(image[None]).to(device), sigmas)
    np.testing.assert_allclose(result.cpu().numpy()[0], expected, rtol=1e-5, atol=1e-5)


def test_rejected_affine_has_no_replay_and_invalid_deferred_requests_fail():
    image = np.ones((2, 12, 12), dtype=np.float32)
    mask = np.ones((12, 12), dtype=np.int64)
    cfg = config(mask.shape, "affine")
    cfg.max_padding_ratio = 0
    cfg.scale_range = (2, 2)
    plan = SpatialImagePlan()
    apply_augmentation(image, mask, cfg, image_plan=plan, defer_photometric=True)
    assert plan.affine_grid is None
    with pytest.raises(ValueError, match="Deferred spatial images"):
        apply_augmentation(image, mask, cfg, image_plan=SpatialImagePlan())
    with pytest.raises(ValueError, match="Deferred spatial images"):
        apply_augmentation(image, mask, cfg, image_plan=SpatialImagePlan(), defer_photometric=True, training=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dimensions", ["2d", "2.5d", "3d"])
@pytest.mark.parametrize("task", ["binary_semantic", "multiclass_semantic", "instance_friendly"])
def test_cuda_training_replays_spatial_images_without_workers(tmp_path, monkeypatch, dimensions, task):
    shape = (32, 32) if dimensions == "2d" else (8, 32, 32)
    mask = np.zeros(shape, dtype=np.uint16)
    mask[..., 6:14, 6:14] = 1
    mask[..., 18:26, 18:26] = 2 if task != "binary_semantic" else 1
    image = mask.astype(np.float32) + np.random.default_rng(0).random(shape, dtype=np.float32)
    for split in ("train", "val"):
        for kind, array in (("images", image), ("masks", mask)):
            folder = tmp_path / "data" / split / kind
            folder.mkdir(parents=True)
            tifffile.imwrite(
                folder / "case.tif",
                array,
                photometric="minisblack",
                metadata={"axes": "YX" if dimensions == "2d" else "ZYX"},
            )
    replays = []
    original = spatial_augment.apply_spatial_image_plans

    def replay(images, plans):
        assert images.device.type == "cuda"
        # Padding limits can reject a warp after instance-scale normalization.
        assert all(plan.blur_sigmas is not None and plan.lowres_shape is not None for plan in plans)
        with NoCudaScalarReads():
            result = original(images, plans)
        replays.append(tuple(result.shape))
        return result

    monkeypatch.setattr(spatial_augment, "apply_spatial_image_plans", replay)
    patch = (8, 16, 16) if dimensions == "3d" else (16, 16)
    result = trainer.train(
        {
            "model_name": "spatial",
            "dataset_path": tmp_path / "data",
            "output_dir": tmp_path / "model",
            "architecture": f"resenc-tiny-{dimensions}",
            "task": task,
            "device": "cuda",
            "num_workers": 0,
            "epochs": 1,
            "steps_per_epoch": 1,
            "batch_size": 2,
            "effective_batch_size": 2,
            "patch_size": list(patch),
            "context_slices": 3,
            "preview_count": 0,
            "validation": {"mode": "light", "light_steps": 1},
            "instance_scale_normalization": {"target_object_fraction": 0.5},
            "skip_empty_patches": False,
            "augmentation": {
                "flip_probability": 1,
                "rotate90_probability": 1,
                "affine_probability": 1,
                "elastic_probability": 1,
                "blur_probability": 1,
                "lowres_probability": 1,
            },
        }
    )
    assert replays == [(2, 3 if dimensions == "2.5d" else 1, *patch)]
    assert np.isfinite(result["metrics"]["train_losses"]["total_loss"])
    log = (tmp_path / "model" / "training.log").read_text()
    assert "Image augmentation backend=cuda" in log
    assert "num_workers=0" in log
    assert str(trainer.__file__) in log


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(24, 24), (8, 24, 24)])
@pytest.mark.parametrize("task", ["binary_semantic", "multiclass_semantic", "instance_friendly"])
def test_device_targets_match_cpu_without_scalar_reads(device, shape, task):
    labels = np.zeros(shape, dtype=np.int64)
    labels[..., :12, 3:9] = 7
    labels[..., 3:17, 9:18] = 50000
    valid = np.ones(shape, dtype=bool)
    valid[..., 4:8, 3:9] = False
    expected = prepare_target(task, labels, [7, 50000], validity=valid)
    deferred = prepare_target(task, labels, [7, 50000], validity=valid, defer_dense=True)
    assert "foreground" not in deferred and "boundary" not in deferred
    batch = {key: torch.from_numpy(value[None]).to(device) for key, value in deferred.items()}
    with NoCudaScalarReads():
        actual = complete_device_targets(task, batch, [7, 50000])
    for key, value in expected.items():
        np.testing.assert_array_equal(actual[key].cpu().numpy()[0], value)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(24, 24), (8, 24, 24)])
def test_spatial_kernel_calls_are_batched_with_different_random_parameters(device, shape, monkeypatch):
    samples, references = [], []
    cfg = config(shape)
    for seed in range(4):
        image = np.random.default_rng(seed).random((3, *shape), dtype=np.float32)
        mask = np.ones(shape, dtype=np.int64)
        plan = SpatialImagePlan()
        raw, labels, valid = apply_augmentation(
            image.copy(),
            mask,
            cfg,
            rng=np.random.default_rng(seed),
            image_plan=plan,
            defer_photometric=True,
            return_validity=True,
        )
        expected, _, _ = apply_augmentation(
            image.copy(), mask, cfg, rng=np.random.default_rng(seed), defer_photometric=True, return_validity=True
        )
        references.append(expected)
        samples.append((SpatialImageSample(torch.from_numpy(raw), plan), {"valid": torch.from_numpy(valid[None])}))
    sampler = Mock(wraps=spatial_augment.F.grid_sample)
    convolution = Mock(wraps=spatial_augment.F.conv3d if len(shape) == 3 else spatial_augment.F.conv2d)
    monkeypatch.setattr(spatial_augment.F, "grid_sample", sampler)
    monkeypatch.setattr(spatial_augment.F, "conv3d" if len(shape) == 3 else "conv2d", convolution)
    batch, target = collate_spatial_samples(samples)
    with NoCudaScalarReads():
        result = batch.materialize(torch.device(device))
        result = result.masked_fill(~target["valid"].to(device), 0)
    np.testing.assert_allclose(result.cpu().numpy(), np.stack(references), atol=1e-5, rtol=1e-5)
    assert sampler.call_count == 4  # Affine, low-resolution down/up, elastic.
    assert all(call.args[0].shape[0] == 4 for call in sampler.call_args_list)
    assert convolution.call_count == len(shape)
    assert all(call.kwargs["groups"] == 12 for call in convolution.call_args_list)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape,patch", [((64, 64), (24, 24)), ((16, 48, 48), (8, 24, 24))])
def test_variable_instance_scale_crops_resize_on_device(device, shape, patch, monkeypatch):
    samples, references = [], []
    cfg = config(patch)
    cfg.instance_scale_enabled = True
    cfg.target_object_diameter_px = 8
    cfg.training_scale_jitter = (1, 1)
    for seed, diameter in enumerate((6, 9, 12, 16)):
        image = np.random.default_rng(seed).random((3, *shape), dtype=np.float32)
        mask = np.ones(shape, dtype=np.int64)
        options = {"object_diameter_px": diameter, "defer_photometric": True, "return_validity": True}
        expected, expected_mask, expected_valid = apply_augmentation(
            image.copy(), mask, cfg, rng=np.random.default_rng(seed), **options
        )
        plan = SpatialImagePlan()
        raw, labels, valid = apply_augmentation(
            image.copy(), mask, cfg, rng=np.random.default_rng(seed), image_plan=plan, **options
        )
        assert tuple(raw.shape[1:]) != patch
        np.testing.assert_array_equal(labels, expected_mask)
        np.testing.assert_array_equal(valid, expected_valid)
        references.append(expected)
        samples.append((SpatialImageSample(torch.from_numpy(raw), plan), {"valid": torch.from_numpy(valid[None])}))
    interpolator = Mock(wraps=spatial_augment.F.interpolate)
    monkeypatch.setattr(spatial_augment.F, "interpolate", interpolator)
    batch, target = collate_spatial_samples(samples)
    assert isinstance(batch.images, list)
    with NoCudaScalarReads():
        result = batch.materialize(torch.device(device)).masked_fill(~target["valid"].to(device), 0)
    np.testing.assert_allclose(result.cpu().numpy(), np.stack(references), atol=1e-5, rtol=1e-5)
    assert all(call.args[0].device.type == device for call in interpolator.call_args_list)
