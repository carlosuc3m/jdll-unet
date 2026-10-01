"""Inspect a saved training window without changing weights or source data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.prepared_run import SavedRun
from jdll_unet.augment import AugmentationConfig, apply_tensor_photometric_augmentation
from jdll_unet.config import ArchitectureConfig, parse_training_config
from jdll_unet.dataset import JdllSegmentationDataset
from jdll_unet.image_reading import image_reading_session
from jdll_unet.losses import compute_loss
from jdll_unet.model import build_unet
from jdll_unet.spatial_augment import collate_spatial_samples
from jdll_unet.targets import complete_device_targets
from jdll_unet.trainer import _check_optimizer_update, _training_dtype


class Nonfinite(RuntimeError):
    pass


def check(name, value):
    if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
        raise Nonfinite(f"{name}: {value.dtype}, shape={tuple(value.shape)}")


def run(args):
    torch.set_num_threads(4)
    torch.manual_seed(42)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    training = config["training"]
    model = build_unet(ArchitectureConfig.from_dict(checkpoint["architecture_config"])).cuda().train()
    model.load_state_dict(checkpoint["state_dict"])
    dtype = _training_dtype(True, torch.device("cuda")) if args.precision == "auto" else getattr(torch, args.precision)
    print(json.dumps({"precision": str(dtype), "checkpoint_epoch": checkpoint["epoch"]}), flush=True)
    del checkpoint
    handles = []
    for name, module in model.named_modules():
        if not list(module.children()):
            def hook(layer, inputs, output, name=name):
                check(f"{name} ({type(layer).__name__}) input", inputs[0])
                check(f"{name} ({type(layer).__name__}) output", output)
            handles.append(module.register_forward_hook(hook))
    replay = SavedRun(args.reuse_run)
    replay.prepare()
    with image_reading_session():
        geometry = replay.resolve(parse_training_config(training), None, emit=lambda *a, **k: None)
        sizes = replay.plan["training_instance_sizes"]
        data = JdllSegmentationDataset(
            geometry.train, config["task"], config["label_values"], config["normalization"],
            AugmentationConfig(**training["augmentation"]), True, dimensions="3d", seed=training["seed"],
            instance_sizes=sizes, fallback_instance_size=float(np.median(list(sizes.values()))),
            case_spacings={row.case: row.spacing for row in geometry.spacing.cases},
            target_spacing=geometry.spacing.target_spacing,
            sample_count=training["steps_per_epoch"] * training["effective_batch_size"],
            empty_patch_fraction=training["empty_patch_fraction"], defer_spatial=True, defer_photometric=True,
        )
        data._normalization_statistics.update({
            (index, None): replay.records[pair.stem]["normalization"] for index, pair in enumerate(data.pairs)
        })
        data.set_epoch(args.epoch)
        for index in range(args.start, args.stop):
            source = data.pairs[data.items[int(data.epoch_indices[index])][0]].stem
            images, cpu_target = collate_spatial_samples([data[index]])
            for name, value in cpu_target.items():
                check(f"CPU target {name}", value)
            images = images.materialize(torch.device("cuda"))
            check("spatial images", images)
            target = complete_device_targets(config["task"], {k: v.cuda() for k, v in cpu_target.items()}, config["label_values"])
            for variant in range(args.variants):
                augmented = apply_tensor_photometric_augmentation(images, target["valid"], data.augmentation)
                check("photometric images", augmented)
                try:
                    with torch.set_grad_enabled(args.backward), torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                        logits = model(augmented)
                        loss, parts = compute_loss(config["task"], logits, target,
                            training["effective_loss_weights"], cpu_validity=cpu_target["valid"])
                    check("loss", loss)
                    for key, value in parts.items():
                        check(f"loss component {key}", value)
                    if args.backward:
                        loss.backward()
                        _check_optimizer_update(model, [loss.detach()], epoch=args.epoch, step=index + 1)
                        model.zero_grad(set_to_none=True)
                except Nonfinite as exc:
                    print(json.dumps({"index": index, "source": source, "variant": variant,
                        "first_nonfinite": str(exc), "input_range": [images.min().item(), images.max().item()],
                        "augmented_range": [augmented.min().item(), augmented.max().item()]}), flush=True)
                    if args.output:
                        torch.save({"images": augmented.cpu(), "target": cpu_target,
                                    "index": index, "source": source, "first_nonfinite": str(exc)}, args.output)
                    return
            print(json.dumps({"index": index, "source": source, "finite_variants": args.variants,
                              "foreground_voxels": int(target["foreground"].sum())}), flush=True)
    for handle in handles:
        handle.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reuse-run", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=27)
    parser.add_argument("--start", type=int, default=980)
    parser.add_argument("--stop", type=int, default=1000)
    parser.add_argument("--variants", type=int, default=8)
    parser.add_argument("--precision", choices=("auto", "float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--output", type=Path)
    run(parser.parse_args())
