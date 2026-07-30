import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch

from jdll_unet.config import LRSchedulerConfig, architecture_defaults, parse_training_config
from jdll_unet.errors import ModelLoadError
from jdll_unet.finetune import (
    SourceModel,
    initialize_finetune_model,
    resolve_finetune_learning_rates,
    resolve_source_model,
)
from jdll_unet.model import build_unet
from jdll_unet.schedulers import LearningRateScheduler
from jdll_unet.trainer import train


def _source(
    architecture,
    *,
    task: str = "binary_semantic",
    labels: tuple[int, ...] | None = (1,),
    learning_rate: float | None = 0.002,
) -> SourceModel:
    model = build_unet(architecture)
    return SourceModel(
        Path("/source"),
        Path("/source/model.pt"),
        architecture,
        {},
        dict(model.state_dict()),
        task,
        labels,
        learning_rate,
    )


def test_scratch_auto_and_explicit_learning_rates_are_preserved(tmp_path: Path):
    automatic = parse_training_config(
        {
            "model_name": "auto",
            "output_dir": tmp_path / "auto",
            "dataset_path": tmp_path / "dataset",
        }
    )
    explicit = parse_training_config(
        {
            "model_name": "explicit",
            "output_dir": tmp_path / "explicit",
            "dataset_path": tmp_path / "dataset",
            "learning_rate": 3.7e-4,
        }
    )
    assert automatic.learning_rate == "auto"
    assert explicit.learning_rate == 3.7e-4


@pytest.mark.parametrize("dimensions", ["2d", "3d"])
@pytest.mark.parametrize("source_channels,target_channels", [(1, 3), (3, 1), (2, 3)])
def test_input_channel_adaptation_rules(dimensions: str, source_channels: int, target_channels: int):
    architecture = architecture_defaults(
        f"resenc-tiny-{dimensions}",
        input_channels=source_channels,
        output_channels=1,
    )
    source = _source(architecture)
    key = "encoders.0.body.0.weight"
    kernel = source.state_dict[key]
    for channel in range(source_channels):
        kernel[:, channel].fill_(channel + 1)
    target_architecture = replace(architecture, input_channels=target_channels)
    model = build_unet(target_architecture)

    report, adapted = initialize_finetune_model(
        model,
        source,
        target_task="binary_semantic",
        target_label_values=[1],
        backbone_learning_rate=2e-4,
        adapted_learning_rate=2e-3,
    )

    loaded = model.state_dict()[key]
    if source_channels == 1:
        assert torch.allclose(loaded, torch.full_like(loaded, 1 / target_channels))
    elif target_channels == 1:
        assert torch.allclose(loaded, torch.full_like(loaded, sum(range(1, source_channels + 1))))
    else:
        assert torch.all(loaded[:, 0] == 1)
        assert torch.all(loaded[:, 1] == 2)
        assert torch.all(loaded[:, 2] == 0)
    assert key in adapted
    assert key in report.adapted_tensors


def test_centered_25d_context_adaptation():
    architecture = architecture_defaults("resenc-tiny-2.5d", input_channels=3, output_channels=1)
    architecture.context_slices = 3
    source = _source(architecture)
    key = "encoders.0.body.0.weight"
    for channel, value in enumerate((10, 20, 30)):
        source.state_dict[key][:, channel].fill_(value)
    target_architecture = replace(architecture, input_channels=5, context_slices=5)
    model = build_unet(target_architecture)

    initialize_finetune_model(
        model,
        source,
        target_task="binary_semantic",
        target_label_values=[1],
        backbone_learning_rate=2e-4,
        adapted_learning_rate=2e-3,
    )

    loaded = model.state_dict()[key]
    assert [float(loaded[0, channel, 0, 0]) for channel in range(5)] == [0, 10, 20, 30, 0]


def test_25d_context_adaptation_preserves_image_channel_groups():
    architecture = architecture_defaults("resenc-tiny-2.5d", input_channels=6, output_channels=1)
    architecture.context_slices = 3
    source = _source(architecture)
    key = "encoders.0.body.0.weight"
    for channel in range(6):
        source.state_dict[key][:, channel].fill_(channel + 1)
    target_architecture = replace(architecture, input_channels=15, context_slices=5)
    model = build_unet(target_architecture)

    initialize_finetune_model(
        model,
        source,
        target_task="binary_semantic",
        target_label_values=[1],
        backbone_learning_rate=2e-4,
        adapted_learning_rate=2e-3,
    )

    loaded = model.state_dict()[key]
    assert [float(loaded[0, channel, 0, 0]) for channel in range(15)] == [
        0,
        1,
        2,
        3,
        0,
        0,
        4,
        5,
        6,
        0,
        0,
        0,
        0,
        0,
        0,
    ]


def test_exact_finetune_load_copies_every_tensor_and_uses_one_rate():
    architecture = architecture_defaults("resenc-tiny-2d", input_channels=1, output_channels=1)
    source = _source(architecture)
    model = build_unet(architecture)

    report, adapted = initialize_finetune_model(
        model,
        source,
        target_task="binary_semantic",
        target_label_values=[1],
        backbone_learning_rate=2e-4,
        adapted_learning_rate=2e-3,
    )

    assert not adapted
    assert report.adapted_layers_learning_rate is None
    assert not report.adapted_tensors
    assert not report.reinitialized_tensors
    assert all(torch.equal(value, source.state_dict[name]) for name, value in model.state_dict().items())


def test_class_mapping_adapts_primary_and_deep_supervision_heads():
    architecture = architecture_defaults(
        "resenc-medium-2d",
        input_channels=1,
        output_channels=3,
        deep_supervision=True,
    )
    source = _source(architecture, task="multiclass_semantic", labels=(1, 2))
    for name, tensor in source.state_dict.items():
        if name.startswith("out.") or name.startswith("deep_supervision_heads."):
            for channel in range(tensor.shape[0]):
                tensor[channel].fill_(10 + channel)
    model = build_unet(architecture)
    initial = {name: value.clone() for name, value in model.state_dict().items()}

    report, adapted = initialize_finetune_model(
        model,
        source,
        target_task="multiclass_semantic",
        target_label_values=[2, 3],
        backbone_learning_rate=2e-4,
        adapted_learning_rate=2e-3,
    )

    for name, tensor in model.state_dict().items():
        if not (name.startswith("out.") or name.startswith("deep_supervision_heads.")):
            continue
        assert torch.all(tensor[0] == 10)
        assert torch.all(tensor[1] == 12)
        assert torch.equal(tensor[2], initial[name][2])
        assert name in report.adapted_tensors
        if name.endswith(("weight", "bias")):
            assert name in adapted


def test_semantic_change_reinitializes_all_output_heads():
    architecture = architecture_defaults(
        "resenc-medium-2d",
        input_channels=1,
        output_channels=1,
        deep_supervision=True,
    )
    source = _source(architecture)
    target_architecture = replace(architecture, output_channels=3)
    model = build_unet(target_architecture)
    initial = {name: value.clone() for name, value in model.state_dict().items()}

    report, _adapted = initialize_finetune_model(
        model,
        source,
        target_task="instance_friendly",
        target_label_values=list(range(1, 8)),
        backbone_learning_rate=2e-4,
        adapted_learning_rate=2e-3,
    )

    heads = [
        name
        for name in initial
        if name.startswith("out.") or name.startswith("deep_supervision_heads.")
    ]
    assert heads
    assert all(torch.equal(model.state_dict()[name], initial[name]) for name in heads)
    assert set(heads) == set(report.reinitialized_tensors)


def _write_source_model(
    folder: Path,
    *,
    input_channels: int = 1,
    output_channels: int = 1,
    learning_rate: float | None = 0.002,
) -> Path:
    folder.mkdir(parents=True)
    architecture = architecture_defaults(
        "resenc-tiny-2d",
        input_channels=input_channels,
        output_channels=output_channels,
    )
    model = build_unet(architecture)
    training = {"learning_rate": learning_rate} if learning_rate is not None else {}
    config = {
        "format": "jdll-unet",
        "format_version": 1,
        "task": "binary_semantic",
        "architecture": architecture.name,
        "architecture_config": asdict(architecture),
        "input_channels": input_channels,
        "label_values": [1],
        "training": training,
    }
    (folder / "config.json").write_text(json.dumps(config))
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture_config": asdict(architecture),
            "model_config": config,
            "task": "binary_semantic",
        },
        folder / "model.pt",
    )
    return folder


def test_source_resolution_rejects_missing_metadata_and_backbone_mismatch(tmp_path: Path):
    missing = tmp_path / "missing"
    missing.mkdir()
    torch.save({"state_dict": {}}, missing / "model.pt")
    with pytest.raises(ModelLoadError, match="no valid state_dict|no recoverable"):
        resolve_source_model(missing)

    inconsistent = _write_source_model(tmp_path / "inconsistent")
    state = torch.load(inconsistent / "model.pt", map_location="cpu", weights_only=False)
    state["state_dict"]["encoders.1.body.0.weight"] = torch.zeros((1,))
    torch.save(state, inconsistent / "model.pt")
    with pytest.raises(ModelLoadError, match="disagree with architecture metadata"):
        resolve_source_model(inconsistent)

    disagreement = _write_source_model(tmp_path / "disagreement")
    config = json.loads((disagreement / "config.json").read_text())
    config["architecture_config"]["activation"] = "silu"
    (disagreement / "config.json").write_text(json.dumps(config))
    with pytest.raises(ModelLoadError, match="architecture_config disagree"):
        resolve_source_model(disagreement)

    malformed = _write_source_model(tmp_path / "malformed")
    malformed_state = torch.load(malformed / "model.pt", map_location="cpu", weights_only=False)
    malformed_config = json.loads((malformed / "config.json").read_text())
    malformed_config.pop("architecture_config")
    malformed_state.pop("architecture_config")
    malformed_state["model_config"].pop("architecture_config")
    (malformed / "config.json").write_text(json.dumps(malformed_config))
    torch.save(malformed_state, malformed / "model.pt")
    with pytest.raises(ModelLoadError, match="Missing or invalid architecture_config"):
        resolve_source_model(malformed)


def test_scheduler_preserves_parameter_group_ratio():
    first = torch.nn.Parameter(torch.tensor(1.0))
    second = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam(
        [{"params": [first], "lr": 1e-4}, {"params": [second], "lr": 1e-3}]
    )
    scheduler = LearningRateScheduler(
        optimizer,
        LRSchedulerConfig(type="poly", min_lr=1e-5),
        total_steps=10,
        total_epochs=2,
    )
    scheduler.step_epoch(0)
    assert optimizer.param_groups[1]["lr"] / optimizer.param_groups[0]["lr"] == pytest.approx(10)


def _rgb_dataset(root: Path) -> None:
    images = root / "images"
    masks = root / "masks"
    images.mkdir(parents=True)
    masks.mkdir()
    yy, xx = np.mgrid[:24, :24]
    for index in range(3):
        mask = (((yy - 12) ** 2 + (xx - 12) ** 2) < (5 + index) ** 2).astype(np.uint8)
        image = np.stack([mask * 100, mask * 80, mask * 60], axis=-1).astype(np.uint8)
        tifffile.imwrite(images / f"sample_{index}.tif", image)
        tifffile.imwrite(masks / f"sample_{index}.tif", mask)


def test_finetune_training_recovers_architecture_groups_rates_metadata_and_callback(tmp_path: Path):
    source = _write_source_model(tmp_path / "source")
    dataset = tmp_path / "dataset"
    _rgb_dataset(dataset)
    output = tmp_path / "output"
    events = []

    result = train(
        {
            "model_name": "fine-tuned",
            "output_dir": output,
            "dataset_path": dataset,
            "starting_point": "fine_tune",
            "base_model": source,
            "learning_rate": "auto",
            "device": "cpu",
            "epochs": 1,
            "steps_per_epoch": 1,
            "patch_size": [24, 24],
            "batch_size": 1,
            "validation": {"mode": "light", "light_steps": 1},
            "preview_count": 0,
        },
        task=events.append,
    )

    config = json.loads((output / "config.json").read_text())
    training = config["training"]
    initialization = training["fine_tuning_initialization"]
    assert config["architecture"] == "resenc-tiny-2d"
    assert config["architecture_config"]["input_channels"] == 3
    assert training["source_learning_rate"] == 0.002
    assert training["backbone_learning_rate"] == pytest.approx(0.0002)
    assert training["adapted_layers_learning_rate"] == 0.002
    assert initialization["adapted_tensors"]
    plan = next(event for event in events if event["type"] == "training_plan")
    assert plan["starting_point"] == "fine_tune"
    assert plan["source_input_channels"] == 1
    assert plan["target_input_channels"] == 3
    assert plan["backbone_learning_rate"] == pytest.approx(0.0002)
    checkpoint = torch.load(result["model_path"], map_location="cpu", weights_only=False)
    assert checkpoint["scheduler_state_dict"]["base_lrs"] == pytest.approx([0.0002, 0.002])


def test_finetune_fallback_rates_and_numeric_lr_rejection(tmp_path: Path):
    source = _write_source_model(tmp_path / "source-no-lr", learning_rate=None)
    resolved = resolve_source_model(source)
    assert resolved.learning_rate is None
    assert resolve_finetune_learning_rates(None) == (1e-4, 1e-3)
    assert resolve_finetune_learning_rates(0.002) == pytest.approx((0.0002, 0.002))
    with pytest.raises(ValueError, match="Numeric learning_rate"):
        train(
            {
                "model_name": "invalid",
                "output_dir": tmp_path / "invalid",
                "dataset_path": tmp_path / "missing",
                "starting_point": "fine_tune",
                "base_model": source,
                "learning_rate": 1e-4,
            }
        )
