import json
from dataclasses import asdict

import pytest
import torch

from jdll_unet.config import ArchitectureConfig, architecture_defaults
from jdll_unet.errors import ConfigError
from jdll_unet.finetune import resolve_source_model
from jdll_unet.infer import load_model
from jdll_unet.model import ResidualEncoderBlock, build_unet


@pytest.mark.parametrize("dimensions,shape", [("2d", (16, 16)), ("2.5d", (16, 16)), ("3d", (8, 16, 16))])
def test_learned_projection_is_normalized_and_identity_stays_identity(dimensions, shape):
    block = ResidualEncoderBlock(8, 16, dimensions=dimensions)
    assert isinstance(block.projection_norm, torch.nn.GroupNorm)
    x = torch.randn(2, 8, *shape)
    before = block.projection_norm(block.projection(x))
    with torch.no_grad():
        block.projection.weight.mul_(100)
    after = block.projection_norm(block.projection(x))
    torch.testing.assert_close(before, after, rtol=1e-3, atol=1e-3)
    block(x).square().mean().backward()
    assert torch.isfinite(block.projection.weight.grad).all()
    assert block.projection_norm.weight.grad is not None

    identity = ResidualEncoderBlock(8, 8, dimensions=dimensions)
    assert isinstance(identity.projection, torch.nn.Identity)
    assert isinstance(identity.projection_norm, torch.nn.Identity)
    assert identity.projection_norm(identity.projection(x)) is x


@pytest.mark.parametrize("normalization,expected", [
    ("group", torch.nn.GroupNorm), ("batch", torch.nn.BatchNorm2d),
    ("instance", torch.nn.InstanceNorm2d), ("none", torch.nn.Identity),
])
def test_projection_uses_configured_normalization(normalization, expected):
    block = ResidualEncoderBlock(2, 8, normalization=normalization)
    assert isinstance(block.projection_norm, expected)


@pytest.mark.parametrize("dimensions", ["2d", "2.5d", "3d"])
def test_resenc_presets_normalize_projections_without_changing_decoder(dimensions):
    architecture = architecture_defaults(f"resenc-tiny-{dimensions}")
    assert architecture.normalize_projection
    model = build_unet(architecture)
    assert all(isinstance(stage.projection_norm, torch.nn.GroupNorm) for stage in model.encoders)
    assert all(isinstance(layer, (torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)) for layer in model.upconvs)


@pytest.mark.parametrize("dimensions,shape", [("2d", (16, 16)), ("2.5d", (16, 16)), ("3d", (8, 16, 16))])
@pytest.mark.parametrize("legacy", [True, False])
def test_saved_architecture_preserves_old_and_new_checkpoint_predictions(tmp_path, dimensions, shape, legacy):
    architecture = ArchitectureConfig(name=f"resenc-tiny-{dimensions}", dimensions=dimensions,
                                      base_channels=4, depth=2, normalize_projection=not legacy)
    model = build_unet(architecture).eval()
    payload = asdict(architecture)
    if legacy:
        del payload["normalize_projection"]
    config = {"format": "jdll-unet", "format_version": 1, "architecture_config": payload,
              "task": "binary_semantic", "label_values": [1], "training": {"learning_rate": 0.001}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    torch.save({"state_dict": model.state_dict(), "architecture_config": payload,
                "model_config": config, "task": "binary_semantic"}, tmp_path / "model.pt")
    loaded, _ = load_model(tmp_path)
    assert loaded.config.normalize_projection is not legacy
    image = torch.randn(1, 1, *shape)
    with torch.no_grad():
        torch.testing.assert_close(loaded(image), model(image), rtol=0, atol=0)
    source = resolve_source_model(tmp_path)
    assert source.architecture.normalize_projection is not legacy
    assert set(source.state_dict) == set(model.state_dict())


def test_projection_policy_requires_boolean():
    with pytest.raises(ConfigError, match="normalize_projection"):
        ArchitectureConfig.from_dict({"normalize_projection": "true"})
