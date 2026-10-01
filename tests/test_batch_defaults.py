import json

import numpy as np
import pytest
import tifffile
import torch

from jdll_unet.config import (
    TrainingConfig,
    architecture_defaults,
    default_batch_size,
    default_effective_batch_size,
    parse_training_config,
)
from jdll_unet.errors import ConfigError
from jdll_unet.image_reading import image_reading_session
from jdll_unet.planning import plan_patch_and_microbatch
from jdll_unet.trainer import train
from jdll_unet.training_geometry import resolve_training_geometry


@pytest.fixture
def batch_request(tmp_path):
    root = tmp_path / "data"
    for folder in ("images", "masks"):
        (root / folder).mkdir(parents=True)
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[32:96, 32:96] = 1
    for index in range(2):
        tifffile.imwrite(root / "images" / f"{index}.tif", mask * 200)
        tifffile.imwrite(root / "masks" / f"{index}.tif", mask)
    return {
        "model_name": "batch-defaults",
        "dataset_path": root,
        "output_dir": tmp_path / "model",
        "architecture": "resenc-tiny-2d",
        "task": "binary_semantic",
        "patch_size": [128, 128],
    }


@pytest.mark.parametrize("architecture", ["tiny-2d", "resenc-tiny-2d", "residual-tiny-2d"])
def test_tiny_2d_batch_defaults(architecture):
    for device in ("cuda", "cuda:0"):
        assert default_batch_size(architecture, torch.device(device)) == 32
        assert default_effective_batch_size(architecture, torch.device(device)) == 32
    for device, expected in (("cpu", 16), ("mps", 4)):
        assert default_batch_size(architecture, torch.device(device)) == expected
        assert default_effective_batch_size(architecture, torch.device(device)) == expected


@pytest.mark.parametrize(
    "architecture,cpu_batch,cuda_batch",
    [
        ("resenc-tiny-2.5d", 4, 4),
        ("resenc-tiny-3d", 1, 2),
        ("resenc-medium-2d", 2, 4),
        ("resenc-big-2d", 1, 2),
        ("resenc-large-2d", 1, 1),
    ],
)
def test_other_presets_keep_batch_defaults(architecture, cpu_batch, cuda_batch):
    for device, batch in (("cpu", cpu_batch), ("cuda", cuda_batch)):
        assert default_batch_size(architecture, torch.device(device)) == batch
        assert default_effective_batch_size(architecture, torch.device(device)) == 4


@pytest.mark.parametrize("as_dataclass", [False, True])
@pytest.mark.parametrize(
    "device,overrides,expected_micro,expected_effective",
    [
        ("cuda", {}, 32, 32),
        ("cpu", {}, 16, 16),
        ("cuda", {"batch_size": 8}, 8, 32),
        ("cuda", {"effective_batch_size": 4}, 4, 4),
        ("cuda", {"batch_size": 8, "effective_batch_size": 8}, 8, 8),
        ("cpu", {"batch_size": 8}, 8, 16),
        ("cpu", {"effective_batch_size": 4}, 4, 4),
        ("cpu", {"batch_size": 8, "effective_batch_size": 8}, 8, 8),
    ],
)
def test_runtime_defaults_preserve_explicit_requests(
    batch_request, as_dataclass, device, overrides, expected_micro, expected_effective
):
    request = {**batch_request, "device": device, **overrides}
    cfg = parse_training_config(TrainingConfig(**request) if as_dataclass else request)
    assert ("effective_batch_size" in cfg._provided_fields) == ("effective_batch_size" in overrides)
    with image_reading_session():
        geometry = resolve_training_geometry(
            cfg,
            architecture_defaults(cfg.architecture),
            inherited=False,
            device=torch.device(device),
            available_memory=4 * 1024**3,
            emit=lambda *args, **kwargs: True,
            check_cancel=lambda: None,
        )
    assert cfg.effective_batch_size == expected_effective
    assert geometry.memory.resolved_microbatch == expected_micro


def test_runtime_cpu_fallback_does_not_select_cuda_batch(batch_request):
    cfg = parse_training_config({**batch_request, "device": "cuda"})
    with image_reading_session():
        geometry = resolve_training_geometry(
            cfg,
            architecture_defaults(cfg.architecture),
            inherited=False,
            device=torch.device("cpu"),
            available_memory=4 * 1024**3,
            emit=lambda *args, **kwargs: True,
            check_cancel=lambda: None,
        )
    assert cfg.effective_batch_size == geometry.memory.resolved_microbatch == 16


@pytest.mark.parametrize("device,effective", [("cpu", 16), ("cuda", 32)])
def test_explicit_patch_still_reduces_microbatch_for_memory(batch_request, device, effective):
    cfg = parse_training_config({**batch_request, "device": device})
    with image_reading_session():
        geometry = resolve_training_geometry(
            cfg,
            architecture_defaults(cfg.architecture),
            inherited=False,
            device=torch.device(device),
            available_memory=128 * 1024**2,
            emit=lambda *args, **kwargs: True,
            check_cancel=lambda: None,
        )
    assert cfg.effective_batch_size == effective
    assert 1 <= geometry.memory.resolved_microbatch < effective
    assert effective % geometry.memory.resolved_microbatch == 0
    assert geometry.memory.resolved_patch == (128, 128)
    assert geometry.memory.reductions == ("user_patch_override", "microbatch_reduced_for_memory")


def test_unaffordable_explicit_patch_fails_without_silent_resize():
    with pytest.raises(ConfigError, match="Requested patch.*microbatch one"):
        plan_patch_and_microbatch(
            (128, 128),
            (128, 128),
            (16, 32, 64, 128),
            (1, 2, 2, 2),
            4,
            32,
            effective_batch_size=32,
            available_memory_bytes=1024,
            allow_patch_reduction=False,
        )


@pytest.mark.parametrize(
    "device,expected",
    [
        ("cpu", 16),
        pytest.param("cuda", 32, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")),
    ],
)
def test_default_batch_saved_and_reusable(batch_request, device, expected):
    result = train(
        {
            **batch_request,
            "device": device,
            "patch_size": [16, 16],
            "epochs": 1,
            "steps_per_epoch": 1,
            "preview_count": 0,
            "validation": {"mode": "light", "light_steps": 1},
        }
    )
    exported = json.loads(json.dumps(result["config"]["training"]))
    assert exported["batch_size"] == exported["microbatch_size"] == exported["effective_batch_size"] == expected
    assert exported["accumulation_steps"] == 1
    parsed = parse_training_config(exported)
    assert parsed.effective_batch_size == expected and "effective_batch_size" in parsed._provided_fields
