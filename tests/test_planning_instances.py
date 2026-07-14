import json
from pathlib import Path

import numpy as np
import tifffile

from jdll_unet.io import ImageMaskPair
from jdll_unet.planning import build_dataset_plan, derive_stage_geometry, resolve_context_stride
from jdll_unet.postprocess import postprocess_instance
from jdll_unet.semantic_scale import compare_semantic_region_fraction, semantic_scale_diagnostics
from jdll_unet.targets import boundary_target, normalized_instance_distance


def _pair(root: Path, name: str, spacing=None) -> ImageMaskPair:
    image = root / f"{name}.tif"
    mask = root / f"{name}_mask.tif"
    tifffile.imwrite(image, np.zeros((4, 8, 8), dtype=np.float32), photometric="minisblack")
    tifffile.imwrite(mask, np.zeros((4, 8, 8), dtype=np.uint16), photometric="minisblack")
    if spacing is not None:
        image.with_suffix(".json").write_text(json.dumps({"spacing": spacing}))
    return ImageMaskPair(image, mask, name)


def test_spacing_imputation_and_context_reliability(tmp_path: Path):
    pairs = [_pair(tmp_path, "a", [2, 1, 1]), _pair(tmp_path, "b", [4, 1, 1]), _pair(tmp_path, "c")]
    plan = build_dataset_plan(pairs, "3d")
    assert plan.known_fraction == 2 / 3
    assert plan.cases[2].spacing == (3.0, 1.0, 1.0)
    assert plan.cases[2].source == "imputed_per_axis_median"
    assert plan.context_spacing == 3.0


def test_context_policies_and_anisotropic_geometry():
    assert resolve_context_stride("adjacent", fixed_stride=7, target_spacing=2, z_spacing=1) == 1
    assert resolve_context_stride("fixed_stride", fixed_stride=3, target_spacing=None, z_spacing=1) == 3
    assert resolve_context_stride("nearest_physical", fixed_stride=1, target_spacing=2, z_spacing=0.5) == 4
    assert resolve_context_stride("nearest_physical", fixed_stride=1, target_spacing=2, z_spacing=3) == 1
    kernels, strides = derive_stage_geometry((16, 96, 96), (5, 1, 1), 4)
    assert kernels[0] == (1, 3, 3)
    assert strides[0] == (1, 2, 2)
    shape = np.asarray((16, 96, 96))
    for stride in strides:
        shape //= stride
    assert np.all(shape >= 4)


def test_physical_distance_boundary_and_watershed():
    labels = np.zeros((16, 16), dtype=np.int32)
    labels[3:10, 2:8] = 1
    labels[3:10, 8:14] = 2
    boundary = boundary_target(labels)[0]
    assert boundary[5, 7] == 1 and boundary[5, 8] == 1
    assert boundary[2, 5] == 1
    distance = normalized_instance_distance(labels)[0]
    assert np.isclose(distance[labels == 1].max(), 1)
    foreground = (labels > 0).astype(np.float32)
    result = postprocess_instance(foreground, boundary, distance, min_object_size=1)
    assert set(np.unique(result["labels"])) == {0, 1, 2}


def test_semantic_scale_diagnostics_2d_per_class_and_border_fallback():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:4, 2:4] = 1
    mask[0:2, 7:10] = 2
    result = semantic_scale_diagnostics([mask], dimensions="2d", patch_size=(10, 10), label_values=[1, 2])
    assert result["per_class"]["1"]["median"] == 0.04
    assert result["per_class"]["2"]["median"] == 0.06
    assert result["per_class"]["2"]["used_border_fallback"] is True
    assert result["pooled_foreground"]["median"] == 0.04


def test_semantic_scale_diagnostics_25d_and_3d():
    volume = np.zeros((3, 6, 6), dtype=np.uint8)
    volume[:, 2:4, 2:4] = 1
    result_25d = semantic_scale_diagnostics([volume], dimensions="2.5d", patch_size=(6, 6), label_values=[1])
    result_3d = semantic_scale_diagnostics([volume], dimensions="3d", patch_size=(3, 6, 6), label_values=[1])
    assert np.isclose(result_25d["pooled_foreground"]["median"], 4 / 36)
    assert np.isclose(result_3d["pooled_foreground"]["median"], 12 / 108)
    comparison = compare_semantic_region_fraction(0.5, result_3d)
    assert comparison["status"] == "above_training_p90"
    assert comparison["warning"] is True
