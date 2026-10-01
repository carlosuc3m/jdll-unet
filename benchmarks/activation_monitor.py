"""Periodic fixed-input activation diagnostics, outside the optimization loop."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from jdll_unet.config import write_json
from jdll_unet.model import ConvBlock, ResidualEncoderBlock


def tensor_statistics(tensor):
    value = tensor.detach().double()
    values = torch.stack((value.min(), value.max(), value.mean(), value.square().mean().sqrt(),
                          value.std(unbiased=False), value.isnan().sum(), value.isinf().sum())).cpu().tolist()
    names = ("minimum", "maximum", "mean", "rms", "std", "nan_count", "inf_count")
    result = {key: (number if math.isfinite(number) else None) for key, number in zip(names, values, strict=True)}
    result.update(shape=list(tensor.shape), dtype=str(tensor.dtype).removeprefix("torch."))
    return result


def measure_activations(model, images, dtype):
    rows = {}
    handles = []
    modes = [(module, module.training) for module in model.modules()]
    selected = (torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d,
                torch.nn.GroupNorm, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
                torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d, ConvBlock, ResidualEncoderBlock)

    def hook(name):
        def record(module, inputs, output):
            rows[name] = {"type": type(module).__name__, "input": tensor_statistics(inputs[0]),
                          "output": tensor_statistics(output)}
        return record

    devices = [images.device.index if images.device.index is not None else torch.cuda.current_device()] if images.is_cuda else []
    try:
        for name, module in model.named_modules():
            if isinstance(module, selected):
                handles.append(module.register_forward_hook(hook(name)))
        model.eval()
        with torch.random.fork_rng(devices=devices), torch.inference_mode(), torch.autocast(
            images.device.type, dtype=dtype, enabled=dtype != torch.float32
        ):
            outputs = model(images)
            heads = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]
            head_stats = [tensor_statistics(value) for value in heads]
    finally:
        for handle in handles:
            handle.remove()
        for module, training in modes:
            module.training = training
    return {"layers": rows, "heads": head_stats,
            "nonfinite_layers": [name for name, row in rows.items()
                                 if row["output"]["nan_count"] or row["output"]["inf_count"]]}


class ActivationMonitor:
    def __init__(self, output, every, total_epochs, reference=None):
        self.output = Path(output)
        self.every = every
        self.total_epochs = total_epochs
        self.reference = Path(reference) if reference is not None else None
        self.probes = None
        self.recorded = set()

    def due(self, epoch):
        return self.every > 0 and epoch not in self.recorded and (
            epoch == 0 or epoch % self.every == 0 or epoch == self.total_epochs
        )

    def prepare_probes(self, dataset):
        probes = []
        counts = {True: 0, False: 0}
        for index in range(len(dataset)):
            image, target = dataset[index]
            labels = target.get("instances", target.get("semantic")) if isinstance(target, dict) else target
            if labels is None:
                labels = target["foreground"]
            foreground = bool(torch.any(labels > 0))
            if counts[foreground] >= 2:
                continue
            counts[foreground] += 1
            probes.append({"name": f"validation_{index:04d}", "image": image[None].detach().cpu().clone(),
                           "foreground": foreground, "sample_index": index, "seed": dataset.seed,
                           "source": dataset.provenance(index)})
            if min(counts.values()) == 2:
                break
        if self.reference is not None:
            reference = torch.load(self.reference, map_location="cpu", weights_only=True)
            probes.append({"name": "previous_overflow_crop", "image": reference["images"].float(),
                           "source": reference.get("source"), "sample_index": reference.get("index"),
                           "origin": str(self.reference)})
        self.output.mkdir(parents=True, exist_ok=True)
        torch.save(probes, self.output / "probes.pt")
        self.probes = probes

    def record(self, epoch, model, dataset, dtype):
        if not self.due(epoch):
            return None
        if self.probes is None:
            self.prepare_probes(dataset)
        device = next(model.parameters()).device
        records = []
        for probe in self.probes:
            image = probe["image"].to(device)
            modes = {}
            for precision in dict.fromkeys((dtype, torch.float32)):
                modes[str(precision).removeprefix("torch.")] = measure_activations(model, image, precision)
            records.append({**{key: value for key, value in probe.items() if key != "image"}, "precisions": modes})
        report = {"epoch": epoch, "every": self.every, "evaluation_mode": True,
                  "training_precision": str(dtype).removeprefix("torch."), "probes": records}
        path = self.output / f"epoch_{epoch:04d}.json"
        write_json(path, report)
        self.recorded.add(epoch)
        return path
