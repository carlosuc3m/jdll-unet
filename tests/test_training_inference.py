import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch

from jdll_unet.appose_api import infer, train
from jdll_unet.errors import ConfigError, InferenceError


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


def _synthetic_volume_dataset(root: Path, count: int = 3) -> None:
    images = root / "images"
    masks = root / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    zz, yy, xx = np.mgrid[:8, :16, :16]
    for index in range(count):
        cz = 3 + (index % 2)
        cy = 7 + (index % 2)
        cx = 7 + (index // 2)
        mask = (((zz - cz) ** 2) / 5 + (yy - cy) ** 2 + (xx - cx) ** 2 <= 4**2).astype(np.uint8)
        image = (mask.astype(np.float32) * 160 + np.random.default_rng(index).normal(15, 2, mask.shape)).astype(np.float32)
        tifffile.imwrite(images / f"volume_{index}.tif", image)
        tifffile.imwrite(masks / f"volume_{index}.tif", mask)


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


def test_2d_instance_scale_normalization_training_and_inference(tmp_path: Path):
    dataset = tmp_path / "instance_dataset"
    _synthetic_dataset(dataset)
    output_dir = tmp_path / "instance_model"

    result = train(
        {
            "model_name": "instance-scale",
            "output_dir": output_dir,
            "dataset_path": dataset,
            "task": "instance_friendly",
            "architecture": "tiny-2d",
            "device": "cpu",
            "epochs": 1,
            "seed": 9,
            "patch_size": [32, 32],
            "batch_size": 2,
            "preview_count": 0,
            "instance_scale_normalization": {
                "target_object_fraction": 0.25,
                "training_scale_jitter": [0.5, 2.0],
            },
        }
    )

    statistics_path = Path(result["dataset_statistics_path"])
    statistics = json.loads(statistics_path.read_text())
    assert statistics["instance_scale_statistics"]["training"]["images_measured"] > 0
    assert result["config"]["training"]["instance_scale_normalization"]["target_object_fraction"] == 0.25

    inference = infer(
        {"model_path": result["model_path"], "device": "cpu", "object_size": 12.0},
        {"image_path": dataset / "images/sample_0.tif"},
    )
    assert inference["outputs"]["foreground_probability"].shape == (32, 32)
    assert inference["outputs"]["boundary_probability"].shape == (32, 32)
    assert inference["outputs"]["labels"].shape == (32, 32)
    assert np.isclose(inference["metadata"]["instance_scale_factor"], 2 / 3)

    with pytest.raises(InferenceError, match="object_size"):
        infer({"model_path": result["model_path"], "device": "cpu"}, {"image_path": dataset / "images/sample_0.tif"})


def test_instance_scale_normalization_rejects_3d(tmp_path: Path):
    dataset = tmp_path / "instance_volume_dataset"
    _synthetic_volume_dataset(dataset, count=2)

    with pytest.raises(ConfigError, match="supports 2D and 2.5D models only"):
        train(
            {
                "model_name": "unsupported-instance-scale-3d",
                "output_dir": tmp_path / "unsupported_model",
                "dataset_path": dataset,
                "task": "instance_friendly",
                "architecture": "tiny-3d",
                "epochs": 1,
                "patch_size": [8, 16, 16],
            }
        )


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


def test_tiny_3d_training_and_inference_smoke(tmp_path: Path):
    dataset = tmp_path / "volume_dataset"
    _synthetic_volume_dataset(dataset)
    output_dir = tmp_path / "volume_model"

    result = train(
        {
            "model_name": "volume-smoke",
            "output_dir": str(output_dir),
            "dataset_path": str(dataset),
            "architecture": "tiny-3d",
            "device": "cpu",
            "epochs": 1,
            "seed": 456,
            "patch_size": [8, 16, 16],
            "batch_size": 1,
            "learning_rate": 0.001,
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
    assert result["config"]["architecture_config"]["dimensions"] == "3d"
    assert result["config"]["training"]["patch_size"] == [8, 16, 16]
    assert result["config"]["input_axes"] == "zyx"
    assert result["config"]["output_axes"] == "zyx"
    assert Path(result["latest_preview_path"]).exists()
    preview = json.loads(Path(result["latest_preview_path"]).read_text())
    assert preview["items"][0]["z_index"] == 4

    checkpoint = torch.load(output_dir / "weights_last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["architecture_config"]["dimensions"] == "3d"

    inference = infer(
        {"model_path": str(output_dir / "model.pt"), "device": "cpu", "tile_size": [8, 16, 16]},
        {"image_path": str(dataset / "images" / "volume_0.tif")},
    )

    assert inference["metadata"]["input_shape"] == [1, 8, 16, 16]
    assert inference["outputs"]["foreground_probability"].shape == (8, 16, 16)
    assert inference["outputs"]["mask"].shape == (8, 16, 16)


def test_25d_instance_training_and_full_volume_inference(tmp_path: Path):
    dataset = tmp_path / "instance_25d_dataset"
    images = dataset / "images"
    masks = dataset / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    for volume_index in range(3):
        mask = np.zeros((5, 32, 32), dtype=np.uint16)
        mask[1:4, 6:12, 6:12] = 1
        mask[1:4, 19:25, 19:25] = 2
        image = mask.astype(np.float32) * 100 + np.random.default_rng(volume_index).normal(10, 1, mask.shape)
        tifffile.imwrite(images / f"volume_{volume_index}.tif", image.astype(np.float32))
        tifffile.imwrite(masks / f"volume_{volume_index}.tif", mask)

    output_dir = tmp_path / "instance_25d_model"
    result = train(
        {
            "model_name": "instance-25d",
            "output_dir": output_dir,
            "dataset_path": dataset,
            "task": "instance_friendly",
            "architecture": "resenc-tiny-2.5d",
            "context_slices": 3,
            "device": "cpu",
            "epochs": 1,
            "patch_size": [24, 24],
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

    assert result["config"]["architecture_config"]["dimensions"] == "2.5d"
    assert result["config"]["architecture_config"]["input_channels"] == 3
    assert result["config"]["architecture_config"]["context_slices"] == 3
    inference = infer(
        {"model_path": result["model_path"], "device": "cpu", "object_size": 7.0, "tile_size": [24, 24]},
        {"image_path": images / "volume_0.tif"},
    )
    assert inference["outputs"]["foreground_probability"].shape == (5, 32, 32)
    assert inference["outputs"]["boundary_probability"].shape == (5, 32, 32)
    assert inference["outputs"]["labels"].shape == (5, 32, 32)
