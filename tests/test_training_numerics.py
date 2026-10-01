from contextlib import nullcontext

import numpy as np
import pytest
import tifffile
import torch

from jdll_unet import trainer
from jdll_unet.errors import TrainingError
from jdll_unet.losses import compute_loss


@pytest.mark.parametrize("enabled,native,expected", [
    (False, True, torch.float32), (True, True, torch.bfloat16), (True, False, torch.float32),
])
def test_cuda_precision_uses_native_bf16_or_fp32(monkeypatch, enabled, native, expected):
    devices = []

    def device_context(device):
        devices.append(device)
        return nullcontext()

    def supported(*, including_emulation):
        assert including_emulation is False
        return native

    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", supported)
    device = torch.device("cuda:1")
    assert trainer._training_dtype(enabled, device) == expected
    assert devices == ([device] if enabled else [])


@pytest.mark.parametrize("major,expected", [(7, torch.float32), (8, torch.bfloat16)])
def test_older_torch_does_not_select_emulated_bf16(monkeypatch, major, expected):
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (major, 0))
    assert trainer._training_dtype(True, torch.device("cuda")) == expected


def test_cpu_precision_does_not_query_cuda(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("CPU precision queried CUDA")

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", unexpected)
    assert trainer._training_dtype(True, torch.device("cpu")) == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_large_finite_3d_activations_do_not_overflow_training_precision():
    model = torch.nn.Sequential(
        torch.nn.Conv3d(1, 2, 3, padding=1, bias=False),
        torch.nn.GroupNorm(1, 2),
        torch.nn.Conv3d(2, 3, 1),
    ).cuda()
    with torch.no_grad():
        model[0].weight.fill_(1)
    images = torch.linspace(3000, 3100, 8**3, device="cuda").reshape(1, 1, 8, 8, 8)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        assert torch.isinf(model[0](images)).any()
        assert torch.isnan(model(images)).any()

    dtype = trainer._training_dtype(True, torch.device("cuda"))
    shape = (1, 1, 8, 8, 8)
    target = {key: torch.zeros(shape, device="cuda") for key in ("foreground", "boundary", "distance")}
    cpu_validity = torch.ones(shape, dtype=torch.bool)
    target["valid"] = cpu_validity.cuda()
    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        logits = model(images)
        loss, components = compute_loss("instance_friendly", logits, target, cpu_validity=cpu_validity)
    assert torch.isfinite(logits).all()
    assert loss.dtype == torch.float32
    assert all(torch.isfinite(value) for value in components.values())
    loss.backward()
    trainer._check_optimizer_update(model, [loss.detach()], epoch=1, step=1)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("kind", ["loss", "gradient"])
def test_update_guard_rejects_nonfinite_loss_or_gradients(bad, kind):
    model = torch.nn.Linear(2, 1)
    model(torch.ones(1, 2)).sum().backward()
    losses = [torch.tensor(1.), torch.tensor(2.)]
    if kind == "loss":
        losses[0] = torch.tensor(bad)
    else:
        model.weight.grad[0, 0] = bad
    with pytest.raises(TrainingError, match="epoch 27, optimizer step 6750"):
        trainer._check_optimizer_update(model, losses, epoch=27, step=6750)


def test_update_guard_does_not_clip_or_change_finite_gradients():
    model = torch.nn.Linear(2, 1)
    model(torch.ones(1, 2)).sum().backward()
    before = [p.grad.clone() for p in model.parameters()]
    trainer._check_optimizer_update(model, [torch.tensor(1.)], epoch=1, step=1)
    for parameter, original in zip(model.parameters(), before, strict=True):
        torch.testing.assert_close(parameter.grad, original)


def test_reporting_rejects_nonfinite_values_before_json():
    with pytest.raises(TrainingError, match="validation.*distance_loss"):
        trainer._tensor_losses_to_float({"distance_loss": torch.tensor(float("nan"))}, context="validation")


@pytest.mark.parametrize("failure", ["training_loss", "gradient", "validation_loss"])
def test_nonfinite_training_reports_cause_and_preserves_checkpoints(tmp_path, monkeypatch, failure):
    images, masks = tmp_path / "data/images", tmp_path / "data/masks"
    images.mkdir(parents=True)
    masks.mkdir()
    mask = np.zeros((32, 32), np.uint8)
    mask[8:24, 8:24] = 1
    for index in range(3):
        tifffile.imwrite(images / f"{index}.tif", mask.astype(np.float32))
        tifffile.imwrite(masks / f"{index}.tif", mask)
    original_loss = trainer.compute_loss
    original_build = trainer.build_unet
    models = []
    steps = []
    callbacks = []

    def build(*args, **kwargs):
        model = original_build(*args, **kwargs)
        models.append(model)
        if failure == "gradient":
            next(model.parameters()).register_hook(lambda grad: grad * float("nan"))
        return model

    def loss(*args, **kwargs):
        value, parts = original_loss(*args, **kwargs)
        if (failure == "training_loss" and models[0].training) or (
            failure == "validation_loss" and not models[0].training
        ):
            value = value * float("nan")
        return value, parts

    original_step = torch.optim.AdamW.step

    def step(*args, **kwargs):
        steps.append(True)
        return original_step(*args, **kwargs)

    monkeypatch.setattr(trainer, "build_unet", build)
    monkeypatch.setattr(trainer, "compute_loss", loss)
    monkeypatch.setattr(torch.optim.AdamW, "step", step)
    output = tmp_path / "model"
    with pytest.raises(TrainingError, match="Non-finite"):
        trainer.train({"model_name": "nonfinite", "dataset_path": images.parent, "output_dir": output,
                       "architecture": "tiny-2d", "device": "cpu", "epochs": 1, "steps_per_epoch": 1,
                       "patch_size": [32, 32], "batch_size": 1, "effective_batch_size": 2,
                       "task": "binary_semantic", "preview_count": 0}, task=callbacks.append)
    assert len(steps) == (1 if failure == "validation_loss" else 0)
    assert callbacks[-1]["type"] == "error"
    assert callbacks[-1]["error_class"] == "TrainingError"
    assert not (output / "metrics.json").exists()
    assert not (output / "weights_best.pt").exists()
    if failure == "validation_loss":
        checkpoint = torch.load(output / "weights_last.pt", map_location="cpu", weights_only=False)
        assert checkpoint["metrics"]["validation_pending"]
        assert all(torch.isfinite(value).all() for value in checkpoint["state_dict"].values())
    else:
        assert not (output / "weights_last.pt").exists()
