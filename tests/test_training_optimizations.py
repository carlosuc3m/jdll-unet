from unittest.mock import Mock

import numpy as np
import pytest
import tifffile
import torch
from scipy import ndimage as ndi
from torch.utils._python_dispatch import TorchDispatchMode

from jdll_unet import dataset as dataset_module
from jdll_unet import targets as targets_module
from jdll_unet import trainer
from jdll_unet.augment import (
    AugmentationConfig,
    apply_augmentation,
    apply_tensor_photometric_augmentation,
    sample_patch,
)
from jdll_unet.dataset import JdllSegmentationDataset
from jdll_unet.errors import DatasetError
from jdll_unet.geometry import inspect_pair
from jdll_unet.infer import _pad_image
from jdll_unet.io import ImageMaskPair, fit_normalization, normalize_image
from jdll_unet.losses import compute_loss, masked_mean
from jdll_unet.metrics import compute_metrics
from jdll_unet.targets import has_trusted_instance_ids, normalized_instance_distance, prepare_target

DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
]


def pair_files(root, shape, name="case"):
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    image = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 7
    mask = np.ones(shape, dtype=np.uint16)
    pair = ImageMaskPair(root / "images" / f"{name}.tif", root / "masks" / f"{name}.tif", name)
    axes = "ZYX" if len(shape) == 3 else "YX"
    tifffile.imwrite(pair.image, image, photometric="minisblack", metadata={"axes": axes})
    tifffile.imwrite(pair.mask, mask, photometric="minisblack", metadata={"axes": axes})
    return pair, image, mask


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape,aux_shape", [((16, 16), (4, 4)), ((4, 16, 16), (2, 4, 4))])
@pytest.mark.parametrize("use_cpu_mask", [False, True])
def test_invalid_auxiliary_support_is_not_silently_averaged(device, shape, aux_shape, use_cpu_mask):
    valid = torch.zeros((1, 1, *shape), dtype=torch.bool)
    valid[..., 5:11, 5:11] = True
    target = {"semantic": torch.ones((1, 1, *shape), device=device), "valid": valid.to(device)}
    logits = [torch.zeros((1, 1, *size), device=device) for size in (shape, aux_shape)]
    with pytest.raises(DatasetError, match="nonempty real support.*auxiliary"):
        compute_loss("binary_semantic", logits, target, cpu_validity=valid if use_cpu_mask else None)
    valid.zero_()
    target["valid"] = valid.to(device)
    with pytest.raises(DatasetError, match="nonempty real support"):
        compute_metrics("binary_semantic", logits[0], target)


class NoCudaScalarReads(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten._local_scalar_dense.default:
            assert args[0].device.type != "cuda", "Loss read a CUDA scalar on the host"
        return func(*args, **(kwargs or {}))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    "task,channels", [("binary_semantic", 1), ("multiclass_semantic", 3), ("instance_friendly", 3)]
)
@pytest.mark.parametrize("shape", [(24, 24), (8, 24, 24)])
def test_cuda_losses_and_gradients_need_no_gpu_scalar_reads(task, channels, shape):
    mask = np.zeros(shape, dtype=np.int64)
    mask[..., 4:9, 4:9] = 1
    mask[..., 12:19, 12:19] = 2
    valid = np.ones(shape, dtype=bool)
    valid[..., :2, :] = False
    cpu_target = {
        key: torch.from_numpy(value[None]) for key, value in prepare_target(task, mask, [1, 2], validity=valid).items()
    }
    target = {key: value.cuda() for key, value in cpu_target.items()}
    logits = [
        torch.randn((1, channels, *size), device="cuda", dtype=torch.float16, requires_grad=True)
        for size in (shape, tuple(length // 2 for length in shape))
    ]
    with NoCudaScalarReads(), torch.autocast("cuda"):
        loss, _ = compute_loss(
            task, logits, target, {"focal": 0.5, "boundary_focal": 0.5}, cpu_validity=cpu_target["valid"]
        )
        loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value.grad).all() for value in logits)


@pytest.mark.parametrize("device", DEVICES)
def test_half_precision_masked_reductions_accumulate_safely(device):
    values = torch.ones((1, 1, 512, 512), dtype=torch.float16, device=device, requires_grad=True)
    valid = torch.ones_like(values, dtype=torch.bool)
    result = masked_mean(values, valid)
    assert result == 1 and result.dtype == torch.float32
    result.backward()
    assert torch.isfinite(values.grad).all()
    assert masked_mean(values, ~valid) == 0


def test_validation_augmentation_accepts_readonly_patch_inputs():
    image = np.ones((1, 16, 16), dtype=np.float32)
    image.flags.writeable = False
    result, _ = apply_augmentation(
        image, np.ones((16, 16), dtype=np.int64), AugmentationConfig(patch_size=(8, 8)), training=False
    )
    assert result.shape == (1, 8, 8)
    assert np.all(result == 1)


@pytest.mark.parametrize("shape,patch", [((8, 8), (16, 16)), ((4, 8, 8), (8, 16, 16))])
@pytest.mark.parametrize("training", [False, True])
def test_zero_padding_is_preserved_after_normalization_and_photometric_changes(shape, patch, training):
    image = np.arange(np.prod(shape), dtype=np.float32).reshape((1, *shape)) + 7
    mask = np.ones(shape, dtype=np.int64)
    cfg = AugmentationConfig(
        patch_size=patch,
        flip_probability=0,
        rotate90_probability=0,
        blur_probability=0,
        brightness_probability=1,
        brightness_range=(2, 2),
        shift_probability=1,
        shift_range=(1, 1),
        contrast_probability=0,
        gamma_probability=0,
        noise_probability=1,
        noise_std=0.1,
    )
    stats = fit_normalization(image, {"type": "minmax"})
    augmented, target, valid = apply_augmentation(
        image,
        mask,
        cfg,
        training=training,
        normalization_statistics=stats,
        return_validity=True,
        rng=np.random.default_rng(1),
    )
    assert augmented.shape[1:] == patch
    assert not augmented[:, ~valid].any()
    assert not target[~valid].any()
    assert valid.sum() == mask.size
    raw, _, support = sample_patch(image, mask, patch, np.random.default_rng(1), return_validity=True)
    assert not raw[:, ~support].any()
    padded, original_shape = _pad_image(image, patch)
    assert original_shape == shape
    np.testing.assert_array_equal(padded[(slice(None), *(slice(0, n) for n in shape))], image)
    assert np.count_nonzero(padded) == np.count_nonzero(image)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape,patch", [((8, 8), (16, 16)), ((4, 8, 8), (8, 16, 16))])
def test_tensor_photometric_matches_cpu_and_keeps_padding_zero(device, shape, patch):
    image = np.arange(np.prod(shape), dtype=np.float32).reshape((1, *shape)) / np.prod(shape)
    mask = np.ones(shape, dtype=np.int64)
    cfg = AugmentationConfig(
        patch_size=patch,
        flip_probability=0,
        rotate90_probability=0,
        blur_probability=0,
        brightness_probability=1,
        brightness_range=(1.2, 1.2),
        shift_probability=1,
        shift_range=(0.1, 0.1),
        contrast_probability=1,
        contrast_range=(0.8, 0.8),
        gamma_probability=1,
        gamma_range=(0.7, 0.7),
        noise_probability=0,
        channel_dropout_probability=0,
    )
    expected, _, valid = apply_augmentation(image, mask, cfg, rng=np.random.default_rng(1), return_validity=True)
    raw, _, _ = apply_augmentation(
        image, mask, cfg, rng=np.random.default_rng(1), defer_photometric=True, return_validity=True
    )
    actual = apply_tensor_photometric_augmentation(
        torch.from_numpy(raw[None]).to(device), torch.from_numpy(valid[None, None]).to(device), cfg
    )
    np.testing.assert_allclose(actual.cpu().numpy()[0], expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "dimensions,shape,expected_fits",
    [("2d", (24, 24), 1), ("2d", (4, 24, 24), 2), ("2.5d", (4, 24, 24), 1), ("3d", (4, 24, 24), 1)],
)
@pytest.mark.parametrize("normalization", ["percentile", "minmax", "zscore", "none"])
def test_statistics_cached_while_patch_values_remain_equivalent(
    tmp_path, monkeypatch, dimensions, shape, expected_fits, normalization
):
    source, image, _ = pair_files(tmp_path, shape)
    pair, _ = inspect_pair(source)
    patch = (4, 16, 16) if dimensions == "3d" else (16, 16)
    dataset = JdllSegmentationDataset(
        [pair],
        "binary_semantic",
        [1],
        {"type": normalization},
        AugmentationConfig(patch_size=patch),
        False,
        dimensions=dimensions,
    )
    fit = Mock(wraps=dataset_module.fit_normalization)
    monkeypatch.setattr(dataset_module, "fit_normalization", fit)
    for index in (0, 1 if len(dataset) > 1 else 0, 0):
        actual, _ = dataset[index]
        source_image = image[index][None] if dimensions == "2d" and len(shape) == 3 else image[None]
        reference = normalize_image(source_image, {"type": normalization})
        if dimensions == "2.5d":
            from jdll_unet.infer import _context_stack

            reference = _context_stack(reference, index, 3)
        np.testing.assert_allclose(actual.numpy(), reference[..., 4:20, 4:20], rtol=1e-6, atol=1e-6)
    assert fit.call_count == expected_fits


@pytest.mark.parametrize("shape", [(40, 40), (10, 40, 40)])
def test_bounding_box_distances_match_full_transforms(shape, monkeypatch):
    mask = np.zeros(shape, dtype=np.int64)
    mask[..., 3:9, 3:9] = 1
    mask[..., 25:35, 25:35] = 2
    valid = np.ones(shape, dtype=bool)
    valid[..., :5, :] = False
    spacing = (2, 0.5, 0.5) if len(shape) == 3 else (0.5, 0.5)
    expected = np.zeros(shape, dtype=np.float32)
    for label in (1, 2):
        distance = ndi.distance_transform_edt(mask == label, sampling=spacing).astype(np.float32)
        expected[mask == label] = distance[mask == label] / distance[valid].max()
    transform = Mock(wraps=ndi.distance_transform_edt)
    monkeypatch.setattr(ndi, "distance_transform_edt", transform)
    np.testing.assert_array_equal(normalized_instance_distance(mask, spacing, valid)[0], expected)
    assert transform.call_count == 2
    assert all(call.args[0].size < mask.size for call in transform.call_args_list)


def test_trusted_instance_ids_are_checked_once_and_preserve_cropped_identity(tmp_path, monkeypatch):
    from jdll_unet import annotations

    source, _, mask = pair_files(tmp_path, (12, 16))
    mask[:] = 0
    mask[1:11, 2:4] = mask[1:11, 8:12] = mask[9:11, 2:12] = 1
    mask[1:3, 14:16] = 2
    tifffile.imwrite(source.mask, mask, metadata={"axes": "YX"})
    pair, _ = inspect_pair(source)
    checker = Mock(wraps=annotations.label_components)
    monkeypatch.setattr(annotations, "label_components", checker)
    dataset = JdllSegmentationDataset(
        [pair], "instance_friendly", None, {"type": "none"}, AugmentationConfig(patch_size=(8, 8)), False
    )
    assert dataset._load_item(0)[-1]
    assert dataset._load_item(0)[-1]
    assert checker.call_count == 1
    crop = mask[:8, :14]
    fast = prepare_target("instance_friendly", crop, canonicalize_instances=False)
    assert ndi.label(crop > 0)[1] == 2
    assert np.unique(fast["instances"]).tolist() == [0, 1]
    broken = mask.copy()
    broken[9:11, 4:8] = 0
    assert not has_trusted_instance_ids(broken)
    assert np.unique(prepare_target("instance_friendly", broken)["instances"]).tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("labels", [(1, 2), (3, 25), (2**40, 2**40 + 999), (1,), (47,)])
def test_source_connectivity_does_not_depend_on_numerical_ids(labels, monkeypatch):
    mask = np.zeros((32, 32), dtype=np.int64)
    for index, label in enumerate(labels):
        mask[3:9, 3 + index * 12 : 9 + index * 12] = label
    checker = Mock(wraps=ndi.label)
    monkeypatch.setattr(ndi, "label", checker)
    assert has_trusted_instance_ids(mask)
    assert checker.call_count == len(labels)
    assert all(call.args[0].size < mask.size for call in checker.call_args_list)
    mask[23:27, 23:27] = labels[0]
    assert not has_trusted_instance_ids(mask)
    assert not has_trusted_instance_ids(np.zeros_like(mask))


def test_sparse_instance_sources_do_not_repeat_patch_connectivity(tmp_path, monkeypatch):
    from jdll_unet import annotations

    source, _, mask = pair_files(tmp_path, (32, 32))
    mask[:] = 0
    mask[5:13, 5:13] = 3
    mask[17:25, 17:25] = 25
    tifffile.imwrite(source.mask, mask, metadata={"axes": "YX"})
    pair, _ = inspect_pair(source)
    checker = Mock(wraps=annotations.label_components)
    monkeypatch.setattr(annotations, "label_components", checker)
    monkeypatch.setattr(
        targets_module, "canonical_instance_labels", Mock(side_effect=AssertionError("Repeated patch connectivity"))
    )
    dataset = JdllSegmentationDataset(
        [pair], "instance_friendly", None, {"type": "none"}, AugmentationConfig(patch_size=(32, 32)), False
    )
    for _ in range(4):
        _, target = dataset[0]
        assert torch.unique(target["instances"]).tolist() == [0, 1, 2]
    assert checker.call_count == 1


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("with_callback", [False, True])
def test_loss_transfers_follow_reporting_and_never_reuse_stale_values(tmp_path, monkeypatch, device, with_callback):
    for index in range(2):
        pair_files(tmp_path / "data", (24, 24), str(index))
    real_loss = trainer.compute_loss
    real_transfer = trainer._tensor_losses_to_float
    microsteps = []
    transfers = []
    events = []

    def loss_with_probe(*args, **kwargs):
        loss, components = real_loss(*args, **kwargs)
        if torch.is_grad_enabled():
            microsteps.append(1)
            components["report_probe"] = loss.new_tensor(float(len(microsteps)))
        return loss, components

    def transfer_with_probe(values, **kwargs):
        result = real_transfer(values, **kwargs)
        if "report_probe" in result:
            transfers.append(result["report_probe"])
        return result

    monkeypatch.setattr(trainer, "compute_loss", loss_with_probe)
    monkeypatch.setattr(trainer, "_tensor_losses_to_float", transfer_with_probe)
    result = trainer.train(
        {
            "model_name": "reporting",
            "dataset_path": tmp_path / "data",
            "output_dir": tmp_path / "model",
            "architecture": "tiny-2d",
            "device": device,
            "epochs": 2,
            "steps_per_epoch": 3,
            "patch_size": [16, 16],
            "batch_size": 1,
            "effective_batch_size": 2,
            "preview_count": 0,
            "progress_update_interval": 2,
            "log_update_interval": 3,
            "validation": {"mode": "light", "light_steps": 1},
        },
        task=events.append if with_callback else None,
    )
    assert len(microsteps) == 12
    assert result["metrics"]["train_losses"]["report_probe"] == 9.5
    if with_callback:
        progress = [
            (event["step"], event["losses"]["train/report_probe"])
            for event in events
            if event["type"] == "progress" and "train/report_probe" in event.get("losses", {})
        ]
        assert progress == [(1, 1.5), (2, 3.5), (4, 7.5), (6, 10.5)]
        assert transfers == [1.5, 3.5, 5.5, 3.5, 7.5, 10.5, 9.5]
    else:
        assert transfers == [3.5, 3.5, 9.5, 9.5]
