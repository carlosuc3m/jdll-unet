import json

import pytest
import torch

from benchmarks.activation_monitor import ActivationMonitor, measure_activations, tensor_statistics
from jdll_unet.config import ArchitectureConfig
from jdll_unet.model import build_unet


class ValidationPatches:
    seed = 12

    def __len__(self):
        return 4

    def __getitem__(self, index):
        return torch.ones(1, 16, 16) * index, {"instances": torch.full((1, 16, 16), index % 2)}

    def provenance(self, index):
        return {"source_image": f"case_{index}"}


def test_diagnostics_restore_state_rng_and_hooks():
    model = build_unet(ArchitectureConfig(base_channels=4, depth=2, normalization="batch", dropout=0.2)).train()
    image = torch.randn(1, 1, 16, 16)
    modes = [module.training for module in model.modules()]
    state = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    report = measure_activations(model, image, torch.float32)
    assert report["nonfinite_layers"] == []
    assert "encoders.0.projection_norm" in report["layers"]
    assert "upconvs.0" in report["layers"]
    assert [module.training for module in model.modules()] == modes
    assert all(not module._forward_hooks for module in model.modules())
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)


def test_monitor_schedule_and_fixed_probe_artifacts(tmp_path):
    model = build_unet(ArchitectureConfig(base_channels=4, depth=2))
    monitor = ActivationMonitor(tmp_path, every=15, total_epochs=100)
    assert [epoch for epoch in range(101) if monitor.due(epoch)] == [0, 15, 30, 45, 60, 75, 90, 100]
    path = monitor.record(0, model, ValidationPatches(), torch.float32)
    assert not monitor.due(0)
    report = json.loads(path.read_text())
    assert len(report["probes"]) == 4
    assert sum(probe["foreground"] for probe in report["probes"]) == 2
    saved = (tmp_path / "probes.pt").read_bytes()
    path = monitor.record(15, model, None, torch.float32)
    assert (tmp_path / "probes.pt").read_bytes() == saved
    later = json.loads(path.read_text())
    assert later["probes"] == report["probes"]
    assert monitor.record(16, model, None, torch.float32) is None


def test_nonfinite_diagnostics_are_explicit_and_json_safe():
    stats = tensor_statistics(torch.tensor([0., float("nan"), float("inf")]))
    assert stats["nan_count"] == 1
    assert stats["inf_count"] == 1
    assert stats["mean"] is None
    json.dumps(stats, allow_nan=False)


def test_diagnostics_remove_hooks_on_failure():
    model = build_unet(ArchitectureConfig(base_channels=4, depth=2)).train()
    with pytest.raises(RuntimeError):
        measure_activations(model, torch.ones(1, 3, 16, 16), torch.float32)
    assert model.training
    assert all(not module._forward_hooks for module in model.modules())
