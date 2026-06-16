from pathlib import Path

import numpy as np
import tifffile

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
    for name in ("config.json", "weights_best.pt", "weights_last.pt", "model.pt", "training.log", "metrics.json"):
        assert (output_dir / name).exists()

    inference = infer(
        {"model_path": str(output_dir / "model.pt"), "device": "cpu", "tile_size": [32, 32]},
        {"image_path": str(dataset / "images" / "sample_0.tif")},
    )

    assert inference["metadata"]["task"] == "binary_semantic"
    assert inference["outputs"]["foreground_probability"].shape == (32, 32)
    assert inference["outputs"]["mask"].shape == (32, 32)
