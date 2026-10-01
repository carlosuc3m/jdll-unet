"""Run-scoped source annotation analysis and bounded prepared-mask storage."""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
from typing import IO, Any

import numpy as np
from skimage.measure import label as label_components

from .config import AnnotationPreparationConfig
from .errors import TaskDetectionError
from .io import ImageMaskPair
from .label_statistics import MaskAnalysis, analyze_mask

_STORAGE_CHUNK_VOXELS = 1024**2


def _storage_dtype(maximum: int) -> np.dtype:
    for scalar_type in (np.uint8, np.uint16, np.uint32, np.uint64):
        if 0 <= maximum <= np.iinfo(scalar_type).max:
            return np.dtype(scalar_type)
    raise ValueError("Prepared instance IDs must fit in an unsigned 64-bit integer")


def _write_compact_labels(stream: IO[bytes], labels: np.ndarray, dtype: np.dtype) -> None:
    np.lib.format.write_array_header_1_0(
        stream,
        {"descr": np.lib.format.dtype_to_descr(dtype), "fortran_order": False, "shape": labels.shape},
    )
    # Convert bounded C-order chunks, including for strided arrays, rather than
    # materializing a second full volume just to write the cache.
    for start in range(0, labels.size, _STORAGE_CHUNK_VOXELS):
        labels.flat[start : start + _STORAGE_CHUNK_VOXELS].astype(dtype, copy=False).tofile(stream)


def component_labels_and_sources(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label equal-valued regions once and map components to original IDs."""
    components, count = label_components(mask, background=0, connectivity=1, return_num=True)
    source_ids = np.zeros(count + 1, dtype=np.int64)
    flat = components.reshape(-1)
    # Every voxel of a component has the same source ID. Bounded scatter avoids
    # scanning a full volume (or overlapping bounding boxes) for each component.
    for start in range(0, flat.size, 1024**2):
        stop = start + 1024**2
        source_ids[flat[start:stop]] = mask.flat[start:stop]
    return components, source_ids


@dataclass
class AnnotationRecord:
    path: str
    components_per_label: dict[int, int]
    labels: np.ndarray | None = None
    disk_path: Path | None = None
    storage: str | None = None
    skipped_reason: str | None = None
    domain_counts: dict[tuple, dict[int, int]] = field(default_factory=dict)
    repair_parents: dict[int, int] = field(default_factory=dict)
    owns_disk_file: bool = True

    @cached_property
    def extra_components(self) -> int:
        return sum(count - 1 for count in self.components_per_label.values())

    @cached_property
    def affected_ids(self) -> int:
        return sum(count > 1 for count in self.components_per_label.values())


@dataclass
class SourceAnalysis:
    path: str
    original: dict[tuple, MaskAnalysis] = field(default_factory=dict)
    prepared: dict[tuple, MaskAnalysis] = field(default_factory=dict)
    geometry: dict[str, Any] = field(default_factory=dict)


class AnnotationPreparation:
    """Analyze original full sources once; expose repairs only for an instance task.

    The byte budget bounds retained corrected masks, not the transient working
    memory required by connected-component labeling of one source.
    """

    def __init__(
        self,
        config: AnnotationPreparationConfig | None = None,
        *,
        emit: Callable[..., Any] | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.config = replace(config) if config is not None else AnnotationPreparationConfig()
        self.emit = emit
        self.cache_dir = Path(self.config.cache_dir) if self.config.cache_dir else cache_dir
        self.records: dict[tuple, AnnotationRecord] = {}
        self.sources: dict[tuple, SourceAnalysis] = {}
        self._label_sets: dict[tuple, tuple[int, ...]] = {}
        self.ram_bytes = 0
        self.instance_mode = False
        self._directory: tempfile.TemporaryDirectory | None = None
        self._warned: set[tuple] = set()

    def __getstate__(self) -> dict[str, Any]:
        # Spawned consumers reuse the parent's files, but never own their cleanup.
        return {
            **vars(self),
            "emit": None,
            "_directory": None,
            "records": {
                key: replace(record, labels=None, owns_disk_file=False) if record.storage == "disk" else record
                for key, record in self.records.items()
            },
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        for record in self.records.values():
            if record.disk_path is not None:
                record.labels = np.load(record.disk_path, mmap_mode="r", allow_pickle=False)

    def _emit(self, event: str, **payload: Any) -> None:
        if self.emit is not None:
            self.emit(event, **payload)
        elif event == "warning":
            logging.getLogger(__name__).warning(payload["message"])

    def _key(self, pair: ImageMaskPair, dimensions: str | None) -> tuple:
        stat = pair.mask.stat()
        return (
            stat.st_dev,
            stat.st_ino,
            pair.mask_axes,
            dimensions if pair.mask_axes is None else None,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
            stat.st_size,
            self.config.repair_disconnected_instances,
            self.config.ram_cache_mb,
            self.config.cache_dir,
            self.config.disk_reserve_mb,
        )

    def _release(self, record: AnnotationRecord) -> None:
        if record.storage == "ram" and record.labels is not None:
            self.ram_bytes -= record.labels.nbytes
        record.labels = None
        if record.disk_path is not None and record.owns_disk_file:
            record.disk_path.unlink(missing_ok=True)
        record.disk_path = None
        record.storage = None

    def remember_labels(self, pair: ImageMaskPair, dimensions: str | None, labels: tuple[int, ...]) -> None:
        self._label_sets[(self._key(pair, dimensions), pair.region)] = labels

    def source_analysis(self, pair: ImageMaskPair, dimensions: str | None) -> SourceAnalysis:
        key = self._key(pair, dimensions)
        if key not in self.sources:
            path = str(pair.mask.resolve())
            for old in [
                old for old, source in self.sources.items()
                if old[:4] == key[:4] or (source.path == path and old[2:4] == key[2:4])
            ]:
                del self.sources[old]
                for domain in [domain for domain in self._label_sets if domain[0] == old]:
                    del self._label_sets[domain]
            self.sources[key] = SourceAnalysis(path)
        return self.sources[key]

    def statistics(
        self, pair: ImageMaskPair, dimensions: str | None, reader: Any = None, *,
        original: bool = False, mask: np.ndarray | None = None
    ) -> MaskAnalysis:
        from .geometry import load_domain_mask

        source = self.source_analysis(pair, dimensions)
        record = self.analyze(pair, dimensions, reader) if self.instance_mode and not original else None
        repaired = record is not None and record.labels is not None
        cache = source.prepared if repaired else source.original
        if pair.region not in cache:
            if mask is None:
                mask = load_domain_mask(pair, dimensions, reader, raw=True, original=not repaired)
            cache[pair.region] = analyze_mask(mask, pair.mask)
            if not repaired:
                self.remember_labels(pair, dimensions, cache[pair.region].labels)
        return cache[pair.region]

    def cached_detection_components(self, pair: ImageMaskPair, dimensions: str | None) -> dict[int, int] | None:
        record = self.records.get(self._key(pair, dimensions))
        return record.domain_counts.get(pair.region) if record is not None else None

    def detection_labels(self, pair: ImageMaskPair, dimensions: str | None) -> tuple[int, ...]:
        components = self.cached_detection_components(pair, dimensions)
        if components is not None:
            return tuple(sorted(components))
        key = (self._key(pair, dimensions), pair.region)
        if key not in self._label_sets:
            self._label_sets[key] = self.statistics(pair, dimensions, original=True).labels
        return self._label_sets[key]

    def _retain(self, labels: np.ndarray, record: AnnotationRecord, *, maximum: int) -> None:
        dtype = _storage_dtype(maximum)
        storage_bytes = labels.size * dtype.itemsize
        if self.ram_bytes + storage_bytes <= self.config.ram_cache_mb * 1024**2:
            try:
                compact = labels.astype(dtype, copy=False)
            except MemoryError:
                pass  # Disk writes need only a bounded conversion buffer.
            else:
                compact.flags.writeable = False
                record.labels, record.storage = compact, "ram"
                self.ram_bytes += storage_bytes
                return
        path = None
        try:
            if self._directory is None:
                if self.cache_dir is not None:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                self._directory = tempfile.TemporaryDirectory(prefix="jdll-annotations-", dir=self.cache_dir)
            directory = Path(self._directory.name)
            required = storage_bytes + 4096 + int(self.config.disk_reserve_mb * 1024**2)
            if shutil.disk_usage(directory).free < required:
                record.skipped_reason = "insufficient_cache_space"
                return
            # A regular write reports ENOSPC; writing to an unreserved mmap may SIGBUS.
            with tempfile.NamedTemporaryFile(dir=directory, suffix=".npy", delete=False) as stream:
                path = Path(stream.name)
                _write_compact_labels(stream.file, labels, dtype)
            record.labels = np.load(path, mmap_mode="r", allow_pickle=False)
            record.disk_path, record.storage = path, "disk"
        except (OSError, MemoryError) as exc:
            if path is not None:
                path.unlink(missing_ok=True)
            record.skipped_reason = f"cache_unavailable: {exc}"

    def analyze(self, pair: ImageMaskPair, dimensions: str | None = "2d", reader: Any = None) -> AnnotationRecord:
        from .geometry import load_domain_mask

        key = self._key(pair, dimensions)
        if key in self.records:
            return self.records[key]
        resolved_path = str(pair.mask.resolve())
        for stale in [
            old
            for old, record in self.records.items()
            if old[:4] == key[:4] or (record.path == resolved_path and old[2:4] == key[2:4])
        ]:
            self._release(self.records.pop(stale))
        mask = load_domain_mask(replace(pair, region=()), dimensions, reader, raw=True, original=True)
        self.statistics(replace(pair, region=()), dimensions, reader, original=True, mask=mask)
        components, source_ids = component_labels_and_sources(mask)
        counts = dict(Counter(int(value) for value in source_ids[1:]))
        record = AnnotationRecord(resolved_path, counts, domain_counts={(): counts})
        if pair.region:
            region = tuple(slice(start, end) for start, end in pair.region)
            present = np.unique(components[region])
            record.domain_counts[pair.region] = dict(Counter(int(source_ids[int(c)]) for c in present if c))
        if record.extra_components:
            maximum = max(counts)
            if not self.config.repair_disconnected_instances:
                record.skipped_reason = "repair_disabled"
            elif maximum + record.extra_components > np.iinfo(np.int64).max:
                record.skipped_reason = "instance_id_overflow"
            else:
                lookup = np.asarray(source_ids, dtype=np.int64)
                seen = set()
                for component, value in enumerate(source_ids[1:], start=1):
                    source_id = int(value)
                    if source_id in seen:
                        maximum += 1
                        lookup[component] = maximum
                        record.repair_parents[maximum] = source_id
                    seen.add(source_id)
                # Reuse the connectivity workspace, remapping in bounded chunks.
                flat = components.reshape(-1)
                for start in range(0, flat.size, 1024**2):
                    chunk = flat[start : start + 1024**2]
                    chunk[:] = lookup[chunk]
                self._retain(components, record, maximum=maximum)
                if record.labels is not None:
                    self.source_analysis(pair, dimensions).prepared[()] = analyze_mask(record.labels, pair.mask)
        self.records[key] = record
        return record

    def detection_components(self, pair: ImageMaskPair, dimensions: str | None) -> dict[int, int]:
        from .geometry import load_domain_mask

        record = self.analyze(pair, dimensions)
        if pair.region not in record.domain_counts:
            if record.labels is not None:
                region = tuple(slice(start, end) for start, end in pair.region)
                present = np.unique(record.labels[region])
                counts = Counter(record.repair_parents.get(int(label), int(label)) for label in present if label)
            elif not record.extra_components:
                present = np.unique(load_domain_mask(pair, dimensions, original=True, raw=True))
                counts = Counter(int(label) for label in present if label)
            else:
                raise TaskDetectionError(
                    "Cannot infer a new region's task from discarded connectivity data. Specify the task explicitly."
                )
            record.domain_counts[pair.region] = dict(counts)
        return record.domain_counts[pair.region]

    def prepared_mask(self, pair: ImageMaskPair, dimensions: str | None, reader: Any) -> np.ndarray | None:
        if not self.instance_mode:
            return None
        record = self.analyze(pair, dimensions, reader)
        if record.extra_components and record.labels is None:
            key = self._key(pair, dimensions)
            if key not in self._warned:
                self._warned.add(key)
                self._emit(
                    "warning",
                    reason="annotation_repair_skipped",
                    path=str(pair.mask),
                    message=f"Instance repair skipped for {pair.mask.name}: {record.skipped_reason}. "
                    "Using original IDs without per-patch repair; disconnected regions retain a shared identity.",
                )
        return record.labels

    def prepare(
        self, pairs: list[ImageMaskPair], dimensions: str, check_cancel: Callable[[], None], reader: Any = None
    ) -> None:
        self.instance_mode = True
        for pair in pairs:
            check_cancel()
            self.prepared_mask(pair, dimensions, reader)

    def diagnostics(self) -> dict[str, Any]:
        records = list(self.records.values())
        affected_sources = sum(record.affected_ids > 0 for record in records)
        affected_ids = sum(record.affected_ids for record in records)
        total_ids = sum(len(record.components_per_label) for record in records)
        return {
            "connectivity": "face",
            "instance_task": self.instance_mode,
            "sources_analyzed": len(records),
            "affected_sources": affected_sources,
            "affected_source_fraction": affected_sources / len(records) if records else 0.0,
            "original_ids": total_ids,
            "affected_ids": affected_ids,
            "affected_id_fraction": affected_ids / total_ids if total_ids else 0.0,
            "repaired_components": sum(
                record.extra_components for record in records if self.instance_mode and record.labels is not None
            ),
            "sources": [
                {
                    "path": record.path,
                    "original_ids": len(record.components_per_label),
                    "affected_ids": record.affected_ids,
                    "extra_components": record.extra_components,
                    "status": "semantic_unchanged"
                    if not self.instance_mode
                    else "unchanged"
                    if not record.extra_components
                    else "repaired"
                    if record.labels is not None
                    else "repair_skipped",
                    "storage": record.storage if self.instance_mode else None,
                    "storage_dtype": str(record.labels.dtype)
                    if self.instance_mode and record.labels is not None else None,
                    "storage_bytes": record.labels.nbytes
                    if self.instance_mode and record.labels is not None else 0,
                    "skipped_reason": record.skipped_reason if self.instance_mode else None,
                }
                for record in records
            ],
        }

    def report(self) -> None:
        if not self.instance_mode:
            for record in self.records.values():
                self._release(record)
            return
        summary = self.diagnostics()
        if self.emit is not None:
            widespread = (
                summary["affected_ids"] > 0
                and max(summary["affected_source_fraction"], summary["affected_id_fraction"])
                >= self.config.warning_fraction
            )
            self.emit(
                "warning" if widespread else "log",
                reason="annotation_preparation",
                message=f"Instance annotation preparation: {summary['affected_sources']}/{summary['sources_analyzed']} "
                f"sources and {summary['affected_ids']}/{summary['original_ids']} IDs are disconnected; "
                f"{summary['repaired_components']} extra components relabelled."
                + (
                    " Widespread disconnection: verify that labels represent objects, not classes or intentionally "
                    "fragmented objects. The selected task has not been changed."
                    if widespread
                    else ""
                ),
                **summary,
            )

    def close(self) -> None:
        for record in self.records.values():
            self._release(record)
        self.records.clear()
        self.sources.clear()
        self._label_sets.clear()
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None
