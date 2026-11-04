from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from jdll_unet import infer as run_inference
from jdll_unet.config import PostprocessingConfig, architecture_defaults, model_folder_config, parse_training_config
from jdll_unet.errors import ConfigError, InferenceError
from jdll_unet.postprocess import postprocess_binary, postprocess_instance, postprocess_multiclass


@pytest.mark.parametrize("shape", [(16, 16), (4, 16, 16)])
def test_binary_outputs_remain_binary_for_disconnected_objects(shape):
    probability = np.zeros(shape, dtype=np.float32)
    probability[..., 1:5, 1:5] = 0.9
    probability[..., 10:14, 10:14] = 0.9
    outputs = postprocess_binary(probability)
    assert set(outputs) == {"mask"}
    assert outputs["mask"].dtype == np.uint8
    assert set(np.unique(outputs["mask"])) == {0, 1}
    np.testing.assert_array_equal(outputs["mask"], probability >= 0.5)


@pytest.mark.parametrize("task, channels", [("binary_semantic", 1), ("multiclass_semantic", 3), ("instance_friendly", 3)])
def test_unknown_postprocessing_fields_are_ignored_and_never_exported(tmp_path, task, channels):
    config = parse_training_config({
        "model_name": "test", "output_dir": tmp_path, "dataset_path": tmp_path, "task": task,
        "postprocessing": {"connected_components": True, "arbitrary_unused_field": {"anything": 42}, "threshold": 0.6},
    })
    expected = asdict(PostprocessingConfig(threshold=0.6))
    assert asdict(config.postprocessing) == expected
    arch = architecture_defaults("tiny-2d", input_channels=1, output_channels=channels)
    exported = model_folder_config(config, task, arch)
    assert exported["postprocessing"] == expected
    assert exported["training"]["postprocessing"] == expected
    assert "connected_components" not in expected


def test_known_postprocessing_settings_are_still_validated(tmp_path):
    base = {"model_name": "test", "output_dir": tmp_path, "dataset_path": tmp_path}
    with pytest.raises(ConfigError, match="threshold"):
        parse_training_config({**base, "postprocessing": {"threshold": 2, "unused": True}})
    with pytest.raises(ConfigError, match="method"):
        parse_training_config({**base, "postprocessing": {"method": "unknown"}})


@pytest.mark.parametrize("shape", [(16, 16), (4, 16, 16)])
def test_semantic_predictions_keep_class_ids(shape):
    classes = np.zeros(shape, dtype=np.uint8)
    classes[..., 1:5, 1:5] = 1
    classes[..., 10:14, 10:14] = 2
    probabilities = np.stack([classes == value for value in range(3)]).astype(np.float32)
    outputs = postprocess_multiclass(probabilities)
    assert set(outputs) == {"mask"}
    np.testing.assert_array_equal(outputs["mask"], classes)


@pytest.mark.parametrize("shape", [(16, 16), (4, 16, 16)])
def test_instance_reconstruction_keeps_explicit_methods_without_a_silent_fallback(shape):
    foreground = np.zeros(shape, dtype=np.float32)
    foreground[..., 1:5, 1:5] = 1
    foreground[..., 10:14, 10:14] = 1
    boundary = np.zeros_like(foreground)
    for method in ("distance_boundary_watershed", "connected_components"):
        distance = foreground if method == "distance_boundary_watershed" else None
        outputs = postprocess_instance(foreground, boundary, distance, method=method)
        assert set(np.unique(outputs["labels"])) == {0, 1, 2}
    with pytest.raises(InferenceError, match="requires a distance prediction map"):
        postprocess_instance(foreground, boundary)
    with pytest.raises(InferenceError, match="Unsupported instance reconstruction method"):
        postprocess_instance(foreground, boundary, foreground, method="unknown")


@pytest.mark.parametrize("task", ["binary_semantic", "multiclass_semantic", "instance_friendly"])
def test_real_inference_keeps_task_outputs_and_ignores_unknown_fields(monkeypatch, task):
    import importlib

    module = importlib.import_module("jdll_unet.infer")

    class Model(torch.nn.Module):
        config = SimpleNamespace(output_channels=1 if task == "binary_semantic" else 3)

        def forward(self, image):
            if task == "multiclass_semantic":
                return torch.cat([-10 * (image - value).square() for value in range(3)], dim=1)
            foreground = 20 * (image - 0.5)
            return foreground if task == "binary_semantic" else torch.cat(
                [foreground, torch.full_like(image, -20), foreground], dim=1)

    config = {
        "task": task, "input_channels": 1, "architecture_config": {"dimensions": "2d"},
        "normalization": {"type": "none"}, "training": {"patch_size": [16, 16]},
        "postprocessing": {"connected_components": True, "arbitrary_unused_field": 42},
    }
    monkeypatch.setattr(module, "load_model", lambda *args: (Model(), config))
    image = np.zeros((16, 16), dtype=np.float32)
    image[1:5, 1:5] = 1
    image[10:14, 10:14] = 2
    result = run_inference({"model_path": "test", "device": "cpu", "tile_overlap": 0,
                            "postprocessing": {"another_unused_field": "ignored"}}, {"image": image})
    outputs = result["outputs"]
    if task == "instance_friendly":
        assert set(np.unique(outputs["labels"])) == {0, 1, 2}
    else:
        assert "labels" not in outputs
        expected = image.astype(np.uint8) if task == "multiclass_semantic" else (image > 0).astype(np.uint8)
        np.testing.assert_array_equal(outputs["mask"], expected)
