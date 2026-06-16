import json
from pathlib import Path

import numpy as np
import tifffile
import torch

from jdll_unet.appose_api import infer, train


def _synthetic_dataset(root: Path, count: int = 4) -> None:
    images = root / "images"
    masks = root / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    yy, xx = np.mgrid[:32, :32]
    for index in range(count):
        cy = 14 + (index % 2) * 3
        cx = 14 + (index // 2) * 3
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2 <= 6**2).astype(np.uint8)
        image = (mask.astype(np.float32) * 180 + np.random.default_rng(index).normal(20, 3, mask.shape)).astype(np.float32)
        tifffile.imwrite(images / f"sample_{index}.tif", image)
        tifffile.imwrite(masks / f"sample_{index}.tif", mask)


def test_tiny_training_and_inference_smoke(tmp_path: Path):
    dataset = tmp_path / "dataset"
    _synthetic_dataset(dataset)
    output_dir = tmp_path / "model"

    result = train(
        {
            "model_name": "smoke",
            "output_dir": str(output_dir),
            "dataset_path": str(dataset),
            "starting_point": "scratch",
            "architecture": "tiny-2d",
            "device": "cpu",
            "epochs": 1,
            "seed": 123,
            "patch_size": [32, 32],
            "batch_size": 2,
            "learning_rate": 0.001,
            "auto_focal": True,
            "auto_focal_foreground_threshold": 0.2,
            "auto_focal_weight": 0.25,
            "preview_count": 1,
            "augmentation": {
                "flip_probability": 0.0,
                "rotate90_probability": 0.0,
                "brightness_probability": 0.0,
                "contrast_probability": 0.0,
                "gamma_probability": 0.0,
                "noise_probability": 0.0,
            },
        }
    )

    assert result["task"] == "binary_semantic"
    assert result["lr_scheduler"]["type"] == "poly"
    assert result["loss_weights"]["focal"] == 0.25
    assert result["target_sparsity"]["foreground_focal_enabled"] is True
    assert result["metrics"]["learning_rate"] < 0.001
    assert "focal_loss" in result["metrics"]["train_losses"]
    assert result["config"]["architecture_config"]["normalization"] == "group"
    for name in ("config.json", "weights_best.pt", "weights_last.pt", "model.pt", "training.log", "metrics.json"):
        assert (output_dir / name).exists()
    config = json.loads((output_dir / "config.json").read_text())
    assert config["training"]["lr_scheduler"]["type"] == "poly"
    assert config["training"]["model_normalization"] == "group"
    assert config["training"]["effective_loss_weights"]["focal"] == 0.25
    assert config["training"]["target_sparsity"]["foreground_focal_enabled"] is True
    checkpoint = torch.load(output_dir / "weights_last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["scheduler_state_dict"]["type"] == "poly"

    inference = infer(
        {"model_path": str(output_dir / "model.pt"), "device": "cpu", "tile_size": [32, 32]},
        {"image_path": str(dataset / "images" / "sample_0.tif")},
    )

    assert inference["metadata"]["task"] == "binary_semantic"
    assert inference["outputs"]["foreground_probability"].shape == (32, 32)
    assert inference["outputs"]["mask"].shape == (32, 32)


def test_training_emits_callback_events_and_png_previews(tmp_path: Path):
    dataset = tmp_path / "dataset"
    _synthetic_dataset(dataset)
    output_dir = tmp_path / "callback_model"
    events = []

    result = train(
        {
            "model_name": "callback-smoke",
            "output_dir": str(output_dir),
            "dataset_path": str(dataset),
            "architecture": "tiny-2d",
            "device": "cpu",
            "epochs": 1,
            "seed": 111,
            "patch_size": [32, 32],
            "batch_size": 2,
            "preview_count": 1,
            "progress_update_interval": 1,
            "augmentation": {
                "flip_probability": 0.0,
                "rotate90_probability": 0.0,
                "brightness_probability": 0.0,
                "contrast_probability": 0.0,
                "gamma_probability": 0.0,
                "noise_probability": 0.0,
            },
        },
        task=events.append,
    )

    event_types = [event["type"] for event in events]
    assert "progress" in event_types
    assert "preview" in event_types
    assert event_types[-1] == "complete"
    preview_event = next(event for event in events if event["type"] == "preview")
    assert Path(preview_event["preview_path"]).is_absolute()
    assert Path(preview_event["latest_preview_path"]).exists()
    assert result["latest_preview_path"] == preview_event["latest_preview_path"]
    assert (output_dir / "previews" / "preview_000_image.png").exists()
    assert (output_dir / "previews" / "preview_000_target.png").exists()
    assert (output_dir / "previews" / "preview_000_prediction.png").exists()
    assert (output_dir / "previews" / "preview_000_overlay.png").exists()


def test_training_callback_can_cancel(tmp_path: Path):
    dataset = tmp_path / "dataset"
    _synthetic_dataset(dataset)
    output_dir = tmp_path / "cancelled_model"

    def cancel_on_progress(event):
        return False if event["type"] == "progress" else None

    result = train(
        {
            "model_name": "cancel-smoke",
            "output_dir": str(output_dir),
            "dataset_path": str(dataset),
            "architecture": "tiny-2d",
            "device": "cpu",
            "epochs": 2,
            "seed": 222,
            "patch_size": [32, 32],
            "batch_size": 2,
            "preview_count": 0,
            "progress_update_interval": 1,
        },
        task=cancel_on_progress,
    )

    assert result["cancelled"] is True
    assert (output_dir / "weights_last.pt").exists()


def test_resenc_deep_supervision_training_smoke(tmp_path: Path):
    dataset = tmp_path / "dataset"
    _synthetic_dataset(dataset)
    output_dir = tmp_path / "resenc_model"

    result = train(
        {
            "model_name": "resenc-smoke",
            "output_dir": str(output_dir),
            "dataset_path": str(dataset),
            "architecture": "resenc-tiny-2d",
            "deep_supervision": True,
            "device": "cpu",
            "epochs": 1,
            "seed": 321,
            "patch_size": [32, 32],
            "batch_size": 2,
            "preview_count": 0,
            "augmentation": {
                "flip_probability": 0.0,
                "rotate90_probability": 0.0,
                "brightness_probability": 0.0,
                "contrast_probability": 0.0,
                "gamma_probability": 0.0,
                "noise_probability": 0.0,
            },
        }
    )

    assert result["config"]["architecture_config"]["block_type"] == "residual"
    assert result["config"]["architecture_config"]["deep_supervision"] is True
    assert "deep_supervision_loss" in result["metrics"]["train_losses"]

    inference = infer(
        {"model_path": str(output_dir / "model.pt"), "device": "cpu", "tile_size": [32, 32]},
        {"image_path": str(dataset / "images" / "sample_0.tif")},
    )
    assert inference["outputs"]["foreground_probability"].shape == (32, 32)
