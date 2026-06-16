from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from jdll_unet.appose_api import infer as appose_infer
from jdll_unet.callbacks import CallbackDispatcher
from jdll_unet.config import architecture_defaults, parse_training_config, write_json
from jdll_unet.errors import ConfigError, InferenceError
from jdll_unet.infer import clear_model_cache, infer
from jdll_unet.losses import compute_loss, primary_logits
from jdll_unet.model import build_unet
from jdll_unet.postprocess import postprocess_binary


def _base_config(tmp_path: Path) -> dict:
    return {
        "model_name": "model",
        "output_dir": str(tmp_path / "out"),
        "dataset_path": str(tmp_path / "data"),
    }


def test_config_coerces_booleans_and_rejects_bad_ranges(tmp_path: Path):
    config = parse_training_config({**_base_config(tmp_path), "save_every_epoch": "false"})
    assert config.save_every_epoch is False

    with pytest.raises(ConfigError, match="foreground_probability"):
        parse_training_config({**_base_config(tmp_path), "foreground_probability": 2.0})

    with pytest.raises(ConfigError, match="Unknown NormalizationConfig"):
        parse_training_config({**_base_config(tmp_path), "normalization": {"unknown": 1}})

    with pytest.raises(ConfigError, match="model_name"):
        parse_training_config({**_base_config(tmp_path), "model_name": "bad/name"})


def test_write_json_is_readable_after_atomic_write(tmp_path: Path):
    path = tmp_path / "nested" / "config.json"
    write_json(path, {"b": 2, "a": [1, 2]})

    assert path.read_text().endswith("\n")
    assert ".config.json.tmp" not in {item.name for item in path.parent.iterdir()}


def _write_minimal_model(folder: Path, input_channels: int = 1) -> Path:
    folder.mkdir()
    arch = architecture_defaults("tiny-2d", input_channels=input_channels, output_channels=1)
    model = build_unet(arch)
    model_config = {
        "format": "jdll-unet",
        "format_version": 1,
        "model_name": "model",
        "task": "binary_semantic",
        "architecture": "tiny-2d",
        "architecture_config": asdict(arch),
        "input_axes": "yx",
        "output_axes": "yx",
        "input_channels": input_channels,
        "num_classes": 1,
        "normalization": {"type": "none"},
        "postprocessing": {"threshold": 0.5, "min_object_size": 0},
        "training": {"patch_size": [16, 16]},
    }
    write_json(folder / "config.json", model_config)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "architecture_config": asdict(arch),
            "model_config": model_config,
        },
        folder / "model.pt",
    )
    return folder / "model.pt"


def test_inference_validates_channels_and_tile_options(tmp_path: Path):
    clear_model_cache()
    model_path = _write_minimal_model(tmp_path / "model", input_channels=1)

    with pytest.raises(InferenceError, match="expects 1 input channel"):
        infer({"model_path": str(model_path), "tile_size": [16, 16]}, {"image": np.zeros((16, 16, 3), dtype=np.float32)})

    with pytest.raises(InferenceError, match="tile_overlap"):
        infer({"model_path": str(model_path), "tile_size": [16, 16], "tile_overlap": 1.0}, {"image": np.zeros((16, 16), dtype=np.float32)})


def test_appose_api_emits_error_update_on_failure():
    updates = []

    with pytest.raises(InferenceError):
        appose_infer({}, {"image": np.zeros((8, 8), dtype=np.float32)}, task=updates.append)

    assert updates[-1]["type"] == "error"
    assert updates[-1]["error_class"] == "InferenceError"


class _ApposeLikeTask:
    def __init__(self) -> None:
        self.updates = []

    def update(self, *, message, current, maximum, info):
        self.updates.append({"message": message, "current": current, "maximum": maximum, "info": info})


def test_callback_dispatcher_supports_appose_and_callables():
    task = _ApposeLikeTask()
    callable_events = []
    dispatcher = CallbackDispatcher([task, callable_events.append])

    assert dispatcher.emit("progress", message="hello", current=1, maximum=2, step=1)

    assert task.updates[0]["message"] == "hello"
    assert task.updates[0]["current"] == 1
    assert task.updates[0]["maximum"] == 2
    assert task.updates[0]["info"]["type"] == "progress"
    assert callable_events[0]["type"] == "progress"
    assert callable_events[0]["message"] == "hello"


def test_callback_dispatcher_can_request_cancellation():
    dispatcher = CallbackDispatcher(lambda _event: False)

    assert dispatcher.emit("progress", step=1) is False
    assert dispatcher.cancel_requested()


def test_postprocess_rejects_invalid_options():
    with pytest.raises(InferenceError, match="threshold"):
        postprocess_binary(np.zeros((8, 8), dtype=np.float32), threshold=2.0)


def test_resenc_architecture_and_deep_supervision_outputs():
    arch = architecture_defaults("resenc-tiny-2d", input_channels=1, output_channels=1, deep_supervision=True)
    model = build_unet(arch)
    outputs = model(torch.zeros((2, 1, 32, 32)))

    assert arch.block_type == "residual"
    assert isinstance(outputs, list)
    assert primary_logits(outputs).shape == (2, 1, 32, 32)
    assert outputs[1].shape[-2:] == (16, 16)

    target = torch.zeros((2, 1, 32, 32))
    loss, components = compute_loss("binary_semantic", outputs, target)
    assert loss.isfinite()
    assert "deep_supervision_loss" in components
