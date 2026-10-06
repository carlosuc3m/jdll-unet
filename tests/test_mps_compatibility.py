import copy
import json
from dataclasses import asdict
from unittest.mock import Mock

import numpy as np
import pytest
import tifffile
import torch
import torch.nn.functional as F

from jdll_unet import device_ops, trainer
from jdll_unet.config import ArchitectureConfig
from jdll_unet.errors import ConfigError
from jdll_unet.infer import infer, load_model
from jdll_unet.losses import resize_target_for_logits
from jdll_unet.model import build_unet
from jdll_unet.targets import prepare_target


def test_mps_never_constructs_an_autocast_context(monkeypatch):
    autocast = Mock(side_effect=RuntimeError("User specified an unsupported autocast device_type 'mps'"))
    monkeypatch.setattr(torch, "autocast", autocast)
    with device_ops.autocast_context(torch.device("mps")):
        pass
    autocast.assert_not_called()
    assert trainer._training_dtype(True, torch.device("mps")) == torch.float32


def test_mps_3d_requires_macos_13_2(monkeypatch):
    version_check = Mock(return_value=False)
    monkeypatch.setattr(torch.backends.mps, "is_macos_or_newer", version_check)
    with pytest.raises(ConfigError, match="macOS 13.2.*Conv3d"):
        device_ops.validate_device_dimensions(torch.device("mps"), "3d")
    version_check.assert_called_once_with(13, 2)
    version_check.return_value = True
    device_ops.validate_device_dimensions(torch.device("mps"), "3d")


@pytest.mark.parametrize("device,dimensions", [("mps", "2d"), ("mps", "2.5d"), ("cpu", "3d"), ("cuda", "3d")])
def test_macos_requirement_does_not_apply_to_other_paths(monkeypatch, device, dimensions):
    version_check = Mock(side_effect=AssertionError("Unnecessary macOS check"))
    monkeypatch.setattr(torch.backends.mps, "is_macos_or_newer", version_check)
    device_ops.validate_device_dimensions(torch.device(device), dimensions)


def test_old_macos_3d_training_fails_before_dataset_analysis(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "resolve_device", lambda _: torch.device("mps"))
    monkeypatch.setattr(torch.backends.mps, "is_macos_or_newer", lambda *_: False)
    geometry = Mock(side_effect=AssertionError("Dataset analysis should not run"))
    monkeypatch.setattr(trainer, "resolve_training_geometry", geometry)
    with pytest.raises(ConfigError, match="macOS 13.2"):
        trainer.train({"model_name": "old-macos", "dataset_path": tmp_path, "output_dir": tmp_path / "model", "device": "mps",
                       "architecture": "resenc-tiny-3d", "epochs": 1})
    geometry.assert_not_called()


@pytest.mark.parametrize("device,dtype,enabled", [("cpu", torch.float32, False),
                                                  ("cuda", torch.float32, False),
                                                  ("cuda", torch.bfloat16, True)])
def test_other_precision_contexts_are_unchanged(monkeypatch, device, dtype, enabled):
    autocast = Mock()
    monkeypatch.setattr(torch, "autocast", autocast)
    device_ops.autocast_context(torch.device(device), dtype)
    autocast.assert_called_once_with(device, dtype=dtype, enabled=enabled)


@pytest.mark.parametrize("stride", [(2, 2, 2), (1, 2, 2), (1, 1, 1)])
@pytest.mark.parametrize("bias", [True, False])
def test_transposed_convolution_preserves_outputs_parameters_and_gradients(monkeypatch, stride, bias):
    reference = torch.nn.ConvTranspose3d(4, 6, stride, stride=stride, groups=2, bias=bias).double()
    compatible = device_ops.CompatibleConvTranspose3d(4, 6, stride, stride=stride, groups=2, bias=bias).double()
    compatible.load_state_dict(reference.state_dict(), strict=True)
    monkeypatch.setattr(device_ops, "_on_mps", lambda tensor: True)
    monkeypatch.setattr(device_ops, "_warn_cpu_operation", Mock())
    images = torch.randn(2, 4, 3, 4, 5, dtype=torch.float64, requires_grad=True)
    copied = images.detach().clone().requires_grad_()
    expected = reference(images)
    output_size = [n + s - 1 for n, s in zip(expected.shape[2:], stride, strict=True)]
    expected = reference(images, output_size=output_size)
    actual = compatible(copied, output_size=output_size)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(copied.grad, images.grad)
    for a, b in zip(compatible.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad)
    assert compatible.state_dict().keys() == reference.state_dict().keys()
    device_ops._warn_cpu_operation.assert_called_once_with("ConvTranspose3d")


@pytest.mark.parametrize("indices", [False, True])
def test_max_pooling_preserves_values_indices_and_gradients(monkeypatch, indices):
    compatible = device_ops.CompatibleMaxPool3d((1, 2, 2), ceil_mode=True, return_indices=indices)
    reference = torch.nn.MaxPool3d((1, 2, 2), ceil_mode=True, return_indices=indices)
    monkeypatch.setattr(device_ops, "_on_mps", lambda tensor: True)
    monkeypatch.setattr(device_ops, "_warn_cpu_operation", Mock())
    x = torch.randn(1, 2, 3, 5, 7, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    actual, expected = compatible(x), reference(y)
    if indices:
        torch.testing.assert_close(actual[1], expected[1])
        actual, expected = actual[0], expected[0]
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(x.grad, y.grad)


@pytest.mark.parametrize("mode", ["trilinear", "nearest"])
def test_volume_resize_preserves_values_and_gradients(monkeypatch, mode):
    monkeypatch.setattr(device_ops, "_on_mps", lambda tensor: True)
    monkeypatch.setattr(device_ops, "_warn_cpu_operation", Mock())
    x = torch.randn(1, 2, 3, 5, 7, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    kwargs = {"size": (5, 7, 11), "mode": mode, "align_corners": False if mode == "trilinear" else None}
    actual = device_ops.interpolate(x, **kwargs)
    expected = F.interpolate(y, **kwargs)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(x.grad, y.grad)


@pytest.mark.parametrize("shape,size", [((1, 1, 7, 11), (3, 5)), ((1, 1, 5, 7, 11), (2, 3, 5))])
def test_nondivisible_adaptive_pooling_is_exact(monkeypatch, shape, size):
    monkeypatch.setattr(device_ops, "_on_mps", lambda tensor: True)
    monkeypatch.setattr(device_ops, "_warn_cpu_operation", Mock())
    x = torch.randn(shape)
    pool = F.adaptive_avg_pool3d if len(shape) == 5 else F.adaptive_avg_pool2d
    torch.testing.assert_close(device_ops.adaptive_avg_pool(x, size), pool(x, size))


def test_cpu_execution_never_uses_compatibility_warning(monkeypatch):
    warning = Mock(side_effect=AssertionError("CPU execution entered MPS fallback"))
    monkeypatch.setattr(device_ops, "_warn_cpu_operation", warning)
    x = torch.ones(1, 2, 4, 8, 8)
    device_ops.CompatibleConvTranspose3d(2, 2, 2, stride=2)(x)
    device_ops.CompatibleMaxPool3d(2)(x)
    device_ops.interpolate(x, size=(8, 16, 16), mode="trilinear", align_corners=False)
    device_ops.adaptive_avg_pool(x, (2, 4, 4))
    warning.assert_not_called()


def test_fallback_warns_once_per_operation():
    device_ops._warn_cpu_operation.cache_clear()
    with pytest.warns(RuntimeWarning, match="runs on CPU") as recorded:
        device_ops._warn_cpu_operation("ConvTranspose3d")
        device_ops._warn_cpu_operation("ConvTranspose3d")
    assert len(recorded) == 1
    device_ops._warn_cpu_operation.cache_clear()


@pytest.mark.parametrize("task", ["binary_semantic", "multiclass_semantic", "instance_friendly"])
def test_deep_supervision_targets_preserved_under_compatibility_routing(monkeypatch, task):
    mask = np.zeros((7, 11, 13), np.int64)
    mask[1:5, 2:6, 3:8] = 1
    mask[5:, 7:, 9:] = 2
    valid = np.ones(mask.shape, bool)
    valid[0] = False
    arrays = prepare_target(task, mask, label_values=[1, 2], validity=valid)
    target = {key: torch.from_numpy(value[None]) for key, value in arrays.items()}
    logits = torch.zeros(1, 3 if task != "binary_semantic" else 1, 3, 5, 6)
    expected = resize_target_for_logits(task, target, logits)
    monkeypatch.setattr(device_ops, "_on_mps", lambda tensor: True)
    monkeypatch.setattr(device_ops, "_warn_cpu_operation", Mock())
    actual = resize_target_for_logits(task, target, logits)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def _dataset(root, dimensions):
    shape = (8, 16, 16) if dimensions in {"2.5d", "3d"} else (16, 16)
    mask = np.zeros(shape, np.uint8)
    mask[..., 3:12, 3:12] = 1
    mask[..., 7:12, 3:12] = 2
    for split in ("train", "val"):
        for kind in ("images", "masks"):
            path = root / split / kind
            path.mkdir(parents=True)
            value = mask.astype(np.float32) if kind == "images" else mask
            tifffile.imwrite(path / "case.tif", value, photometric="minisblack",
                             metadata={"axes": "ZYX" if mask.ndim == 3 else "YX"})
    return mask


@pytest.mark.parametrize("task", ["binary_semantic", "multiclass_semantic", "instance_friendly"])
def test_training_validation_preview_and_inference_use_compatible_paths(tmp_path, monkeypatch, task):
    from jdll_unet import validation_previews, volumetric_validation
    mask = _dataset(tmp_path / "data", "3d")
    monkeypatch.setattr(device_ops, "_on_mps", lambda tensor: tensor.device.type == "cpu")
    monkeypatch.setattr(device_ops, "_warn_cpu_operation", Mock())
    calls = []
    def compatible_context(device, dtype):
        calls.append(True)
        return device_ops.autocast_context(torch.device("mps"), dtype)
    for module in (trainer, validation_previews, volumetric_validation):
        monkeypatch.setattr(module, "autocast_context", compatible_context)
    monkeypatch.setattr(torch, "autocast", Mock(side_effect=AssertionError("Unsupported autocast constructed")))
    result = trainer.train({"model_name": "compatibility", "output_dir": tmp_path / "model",
        "dataset_path": tmp_path / "data", "device": "cpu", "task": task,
        "instance_scale_normalization": {"enabled": False},
        "architecture": "resenc-tiny-3d", "deep_supervision": True, "epochs": 1, "steps_per_epoch": 1,
        "patch_size": [8, 16, 16], "batch_size": 1, "effective_batch_size": 1,
        "preview_count": 1, "validation": {"full_every": 1}})
    assert len(calls) >= 3
    assert result["full_validation_failures"] == 0
    assert result["latest_preview_path"]
    output = infer({"model_path": result["model_path"], "device": "cpu"}, mask.astype(np.float32))
    output_key = "labels" if task == "instance_friendly" else "mask"
    assert output["outputs"][output_key].shape == mask.shape


def test_checkpoint_layout_and_predictions_remain_compatible(tmp_path, monkeypatch):
    config = ArchitectureConfig(name="resenc-tiny-3d", dimensions="3d", base_channels=4, depth=2)
    model = build_unet(config).eval()
    reference = copy.deepcopy(model)
    reference.upconvs[0] = torch.nn.ConvTranspose3d(8, 4, 2, stride=2)
    reference.pools[0] = torch.nn.MaxPool3d(2)
    reference.load_state_dict(model.state_dict(), strict=True)
    torch.save({"state_dict": model.state_dict(), "architecture_config": asdict(config),
                "model_config": {"architecture_config": asdict(config), "task": "binary_semantic"}}, tmp_path / "model.pt")
    loaded, _ = load_model(tmp_path)
    monkeypatch.setattr(device_ops, "_on_mps", lambda tensor: True)
    monkeypatch.setattr(device_ops, "_warn_cpu_operation", Mock())
    x = torch.randn(1, 1, 8, 16, 16)
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), reference(x), rtol=0, atol=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires Apple MPS hardware")
@pytest.mark.parametrize("dimensions", ["2d", "2.5d", "3d"])
@pytest.mark.parametrize("task", ["binary_semantic", "multiclass_semantic", "instance_friendly"])
def test_real_mps_training_validation_previews_and_inference(tmp_path, dimensions, task):
    if dimensions == "3d" and not torch.backends.mps.is_macos_or_newer(13, 2):
        pytest.skip("Conv3d requires macOS 13.2 or newer")
    mask = _dataset(tmp_path / "data", dimensions)
    events = []
    result = trainer.train({"model_name": "mps", "output_dir": tmp_path / "model",
        "dataset_path": tmp_path / "data", "device": "mps", "task": task,
        "instance_scale_normalization": {"enabled": False},
        "architecture": f"resenc-tiny-{dimensions}", "deep_supervision": True,
        "epochs": 1, "steps_per_epoch": 1, "patch_size": list(mask.shape if dimensions == "3d" else mask.shape[-2:]),
        "batch_size": 1, "effective_batch_size": 1, "preview_count": 1, "validation": {"full_every": 1}}, task=events.append)
    assert result["config"]["training"]["resolved_device"] == "mps"
    assert result["latest_preview_path"]
    assert result["full_validation_failures"] == 0
    if dimensions == "3d":
        assert any(event.get("execution") == "hybrid_3d" for event in events)
    output = infer({"model_path": result["model_path"], "device": "mps"}, mask.astype(np.float32))
    output_key = "labels" if task == "instance_friendly" else "mask"
    assert output["outputs"][output_key].shape == mask.shape
    history = json.loads((tmp_path / "model/metrics.json").read_text())["history"]
    assert np.isfinite(history[0]["train_losses"]["total_loss"])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires Apple MPS hardware")
def test_real_mps_fallback_gradients_reach_inputs_and_optimizer_parameters(monkeypatch):
    monkeypatch.setattr(device_ops, "_warn_cpu_operation", Mock())
    reference = device_ops.CompatibleConvTranspose3d(2, 3, 2, stride=2)
    compatible = copy.deepcopy(reference).to("mps")
    x = torch.randn(1, 2, 3, 4, 5, requires_grad=True)
    y = x.detach().to("mps").requires_grad_()
    pool = device_ops.CompatibleMaxPool3d(2)
    outputs = []
    for module, inputs in ((reference, x), (compatible, y)):
        output = device_ops.interpolate(pool(module(inputs)), size=(4, 5, 6), mode="trilinear", align_corners=False)
        output.square().mean().backward()
        outputs.append(output.detach().cpu())
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-4, atol=1e-5)
    assert y.grad is not None and y.grad.device.type == "mps"
    torch.testing.assert_close(x.grad, y.grad.cpu(), rtol=1e-4, atol=1e-5)
    for expected, actual in zip(reference.parameters(), compatible.parameters(), strict=True):
        assert actual.grad is not None and actual.grad.device.type == "mps"
        torch.testing.assert_close(expected.grad, actual.grad.cpu(), rtol=1e-4, atol=1e-5)
    before = compatible.weight.detach().clone()
    torch.optim.SGD(compatible.parameters(), lr=0.1).step()
    assert not torch.equal(before, compatible.weight)
