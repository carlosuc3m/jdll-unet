"""Reuse a verified 3D instance run and persist the runtime assets it did not save."""

from __future__ import annotations

import hashlib
import json
import mmap
import shutil
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import ndimage as ndi

from jdll_unet import trainer
from jdll_unet.annotations import AnnotationPreparation, AnnotationRecord, _storage_dtype
from jdll_unet.config import ArchitectureConfig, write_json
from jdll_unet.dataset import inspect_dataset
from jdll_unet.geometry import DomainReader, load_domain_image, load_domain_mask
from jdll_unet.image_reading import current_read_session, image_reading_session
from jdll_unet.io import fit_normalization
from jdll_unet.label_statistics import LabelRegion, MaskAnalysis, analyze_mask
from jdll_unet.planning import CaseSpacing, DatasetPlan, RuntimeMemoryPlan
from jdll_unet.scale import InstanceSizeEstimate
from jdll_unet.training_geometry import TrainingGeometry

from .profile_sample_preprocessing import pair_from_saved


def save_analysis(path, analysis):
    metadata = asdict(analysis)
    indices = metadata.pop("foreground_indices")
    with path.open("wb") as stream:
        np.savez_compressed(stream, metadata=json.dumps(metadata), foreground_indices=indices)


def load_analysis(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        indices = data["foreground_indices"].copy()
    def regions(items):
        return {int(k): LabelRegion(v["count"], tuple(tuple(b) for b in v["bounds"])) for k, v in items.items()}
    return MaskAnalysis(tuple(meta["shape"]), regions(meta["objects"]),
                        tuple(regions(plane) for plane in meta["planes"]), indices)


def repair_copy(mask, analysis, path, expected_extra):
    """Rebuild deleted repairs using object bounding boxes, preserving raster-order IDs."""
    counts, extras = {}, []
    for label, region in analysis.objects.items():
        spatial = tuple(slice(a, b) for a, b in region.bounds)
        components, count = ndi.label(mask[spatial] == label)
        counts[label] = count
        if count > 1:
            for component, bounds in enumerate(ndi.find_objects(components), start=1):
                if component == 1:
                    continue
                crop = components[bounds]
                first = np.unravel_index(int(np.flatnonzero(crop == component)[0]), crop.shape)
                absolute = tuple(int(i + b.start + region.bounds[axis][0]) for axis, (i, b) in enumerate(zip(first, bounds, strict=True)))
                extras.append((int(np.ravel_multi_index(absolute, mask.shape)), label, component))
        del components
    if len(extras) != expected_extra:
        raise ValueError("Reconstructed connectivity differs from the saved run; do not reuse its plan")
    maximum = max(analysis.objects, default=0)
    replacements = {}
    parents = {}
    for offset, (_, label, component) in enumerate(sorted(extras), start=1):
        replacements.setdefault(label, {})[component] = maximum + offset
        parents[maximum + offset] = label
    dtype = _storage_dtype(maximum + len(extras))
    if shutil.disk_usage(path.parent).free < mask.size * dtype.itemsize + 4 * 1024**3:
        raise OSError("Insufficient disk space for prepared masks and the 4 GiB reserve")
    temporary = path.with_suffix(".tmp.npy")
    output = np.lib.format.open_memmap(temporary, mode="w+", dtype=dtype, shape=mask.shape)
    for start in range(0, mask.size, 1024**2):
        output.reshape(-1)[start:start + 1024**2] = mask.flat[start:start + 1024**2]
    for label, mapping in replacements.items():
        spatial = tuple(slice(a, b) for a, b in analysis.objects[label].bounds)
        components, _ = ndi.label(mask[spatial] == label)
        view = output[spatial]
        for component, value in mapping.items():
            view[components == component] = value
        del components, view
    output.flush()
    del output
    temporary.replace(path)
    return counts, parents


class SavedRun:
    def __init__(self, run, *, normalize_projection=None):
        self.run = Path(run)
        self.normalize_projection = normalize_projection
        self.model_dir = self.run / "model"
        self.config = json.loads((self.model_dir / "config.json").read_text())
        self.plan = json.loads((self.model_dir / "dataset_plan.json").read_text())
        self.training = self.config["training"]
        if self.plan["dimensions"] != "3d" or self.config["task"] != "instance_friendly":
            raise ValueError("Saved-run replay currently supports verified 3D instance datasets only")
        self.train = [pair_from_saved(row) for row in self.plan["training_domains"]]
        self.val = [pair_from_saved(row) for row in self.plan["validation_domains"]]
        domains = {(row["source"], row["split"]): tuple(tuple(b) for b in row["region"])
                   for row in self.plan["case_sampling"]}
        self.train = [replace(p, region=domains[(p.stem, "train")]) for p in self.train]
        self.val = [replace(p, region=domains[(p.stem, "val")]) for p in self.val]
        self.cache = self.run / "prepared_cache"
        self.cache.mkdir(exist_ok=True)
        self.records = {}
        self.cutoff = (self.model_dir / "dataset_plan.json").stat().st_mtime_ns

    def paths(self, pair):
        key = hashlib.sha256(str(pair.mask.resolve()).encode()).hexdigest()[:20]
        return self.cache / key

    def fingerprint(self, pair):
        files = {}
        for path in (pair.image, pair.mask):
            stat = path.stat()
            if stat.st_mtime_ns > self.cutoff:
                raise ValueError(f"Source changed after the saved checks: {path}")
            files[str(path.resolve())] = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
        return {"files": files, "normalization": self.config["normalization"], "version": 1}

    def prepare(self):
        annotations = {row["path"]: row for row in self.plan["annotation_preparation"]["sources"]}
        with image_reading_session() as session:
            reader = DomainReader(max_bytes=0, session=session)
            for index, pair in enumerate(self.train + self.val, start=1):
                base = self.paths(pair)
                fingerprint = self.fingerprint(pair)
                metadata_path = base.with_suffix(".json")
                if metadata_path.exists():
                    metadata = json.loads(metadata_path.read_text())
                    if metadata["fingerprint"] == fingerprint and base.with_suffix(".npz").exists() and (
                        not metadata["repaired"] or base.with_suffix(".npy").exists()
                    ):
                        self.records[pair.stem] = metadata
                        continue
                saved = annotations[str(pair.mask.resolve())]
                mask = load_domain_mask(pair, "3d", reader, raw=True, original=True)
                analysis = analyze_mask(mask, pair.mask)
                repaired = saved["status"] == "repaired"
                if repaired:
                    counts, parents = repair_copy(mask, analysis, base.with_suffix(".npy"), saved["extra_components"])
                    del mask
                    prepared = np.load(base.with_suffix(".npy"), mmap_mode="r", allow_pickle=False)
                    analysis = analyze_mask(prepared, pair.mask)
                    if hasattr(prepared._mmap, "madvise"):
                        prepared._mmap.madvise(mmap.MADV_DONTNEED)
                    del prepared
                else:
                    counts, parents = dict.fromkeys(analysis.labels, 1), {}
                    del mask
                save_analysis(base.with_suffix(".npz"), analysis)
                del analysis
                image = load_domain_image(pair, "3d", reader, raw=True)
                normalization = fit_normalization(image, self.config["normalization"])
                del image
                metadata = {"fingerprint": fingerprint, "repaired": repaired, "components": counts,
                            "parents": parents, "normalization": normalization}
                write_json(metadata_path, metadata)
                self.records[pair.stem] = metadata
                print(json.dumps({"asset_prepared": pair.stem, "completed": index,
                                  "total": len(self.train) + len(self.val), "repaired_copy": repaired}), flush=True)

    def resolve(self, cfg, source_arch, *, emit, **kwargs):
        if cfg.architecture != self.training["architecture"] or str(cfg.dataset_path) != self.training["dataset_path"]:
            raise ValueError("Saved plan must match the requested dataset and architecture")
        session = current_read_session()
        session.domain_reader = DomainReader(max_bytes=128 * 1024**2, session=session)
        preparation = AnnotationPreparation(cfg.annotation_preparation, emit=emit)
        preparation.instance_mode = True
        session.annotations = preparation
        for pair in self.train + self.val:
            metadata = self.records[pair.stem]
            if self.fingerprint(pair) != metadata["fingerprint"]:
                raise ValueError(f"Source changed since asset preparation: {pair.image}")
            base = self.paths(pair)
            counts = {int(k): v for k, v in metadata["components"].items()}
            labels = np.load(base.with_suffix(".npy"), mmap_mode="r", allow_pickle=False) if metadata["repaired"] else None
            record = AnnotationRecord(str(pair.mask.resolve()), counts, labels=labels,
                disk_path=base.with_suffix(".npy") if labels is not None else None,
                storage="disk" if labels is not None else None,
                domain_counts={(): counts, pair.region: counts},
                repair_parents={int(k): v for k, v in metadata["parents"].items()}, owns_disk_file=False)
            preparation.records[preparation._key(pair, "3d")] = record
            source = preparation.source_analysis(pair, "3d")
            cache = source.prepared if labels is not None else source.original
            cache[()] = cache[pair.region] = load_analysis(base.with_suffix(".npz"))
        spacing_values = dict(self.plan["spacing"])
        spacing_values["cases"] = tuple(CaseSpacing(row["case"], tuple(row["spacing"]), row["source"],
            tuple(row["original_spacing"]) if row["original_spacing"] else None) for row in spacing_values["cases"])
        for key in ("default_spacing", "target_spacing"):
            if spacing_values[key] is not None:
                spacing_values[key] = tuple(spacing_values[key])
        spacing = DatasetPlan(**spacing_values)
        plan = json.loads((self.run / "resolved_training_plan.json").read_text())
        memory_values = plan["memory_plan"]
        for key in ("preferred_patch", "resolved_patch", "reductions"):
            memory_values[key] = tuple(memory_values[key])
        arch_values = dict(self.config["architecture_config"])
        for key in ("channels", "encoder_blocks", "kernels", "strides"):
            arch_values[key] = tuple(tuple(v) if isinstance(v, list) else v for v in arch_values[key])
        arch = ArchitectureConfig.from_dict(arch_values)
        if self.normalize_projection is not None:
            if cfg.starting_point != "scratch":
                raise ValueError("Changing projection normalization requires training from scratch")
            arch = replace(arch, normalize_projection=self.normalize_projection)
        if json.loads(json.dumps(asdict(cfg.instance_scale_normalization))) != self.training["instance_scale_normalization"]:
            raise ValueError("Saved object-size measurements require the same instance-scale policy")
        if cfg.instance_scale_normalization.object_size_measure != "equivalent_sphere_diameter":
            raise ValueError("Saved-run replay currently requires equivalent sphere measurements")
        estimates = {}
        for pairs, key in ((self.train, "training_instance_sizes"), (self.val, "validation_instance_sizes")):
            for pair in pairs:
                analysis = preparation.statistics(pair, "3d", session.domain_reader)
                available = sum(region.count >= cfg.instance_scale_normalization.min_instance_area
                                for region in analysis.objects.values())
                size = self.plan[key].get(pair.stem)
                estimates[(pair.stem, pair.region)] = (None if size is None else InstanceSizeEstimate(
                    size, min(cfg.instance_scale_normalization.max_instances_per_image, available), available), 0)
        emit("cached_dataset_plan", message="Reusing verified source geometry, spacing, and prepared annotation assets.")
        return TrainingGeometry(self.train, self.val, inspect_dataset(self.train, "3d"), self.config["task"],
            spacing, RuntimeMemoryPlan(**memory_values), arch, self.training["context"]["stride_policy"],
            self.training["context"]["spacing"], self.plan["sources"], self.plan["network_shape"],
            estimates, self.plan["case_sampling"], preparation.diagnostics())

    @contextmanager
    def activate(self):
        original = trainer.make_dataset
        def make_dataset(*args, **kwargs):
            data = original(*args, **kwargs)
            data._normalization_statistics.update({(index, None): self.records[p.stem]["normalization"]
                                                   for index, p in enumerate(data.pairs)})
            return data
        with patch.object(trainer, "resolve_training_geometry", self.resolve), patch.object(trainer, "make_dataset", make_dataset):
            yield
