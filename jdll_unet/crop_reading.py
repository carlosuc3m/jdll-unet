"""Lazy domain arrays: read, normalize, and resample only requested spatial support."""

from __future__ import annotations

import mmap
from dataclasses import dataclass

import numpy as np
from numpy.typing import DTypeLike
from scipy import ndimage as ndi

from .geometry import DomainReader, load_domain_image, load_domain_mask
from .io import ImageMaskPair, normalize_image


@dataclass
class CropArray:
    pair: ImageMaskPair
    reader: DomainReader
    mask: bool = False
    original_mask: bool = False
    statistics: dict | None = None
    center_z: int | None = None
    context_slices: int = 1
    context_stride: int = 1
    resampled_shape: tuple[int, ...] | None = None

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        if self.center_z is not None:
            return self.pair.domain_shape[-2:]
        return self.resampled_shape or self.pair.domain_shape

    @property
    def shape(self) -> tuple[int, ...]:
        return self.spatial_shape if self.mask else (self.pair.image_channels * self.context_slices, *self.spatial_shape)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def _read_native(self, spatial: tuple[slice, ...]) -> np.ndarray:
        pair = self.pair
        preparation = self.reader.session.annotations
        if self.mask and not self.original_mask and preparation is not None:
            prepared = preparation.prepared_mask(pair, "3d" if len(pair.domain_shape) == 3 else "2d", self.reader)
            if prepared is not None:
                region = pair.region or tuple((0, size) for size in pair.spatial_shape)
                absolute = tuple(slice(s.start + start, s.stop + start) for s, (start, _) in zip(spatial, region, strict=True))
                if isinstance(prepared, np.memmap):
                    result = np.array(prepared[absolute], copy=True)
                    # Keep file-backed pages reclaimable instead of growing RSS across cases.
                    mapping = getattr(prepared, "_mmap", None)
                    if mapping is not None and hasattr(mapping, "madvise"):
                        mapping.madvise(mmap.MADV_DONTNEED)
                    return result
                return prepared[absolute]
        axes = pair.mask_axes if self.mask else pair.image_axes
        if axes is None:
            array = (
                load_domain_mask(pair, "3d" if len(pair.domain_shape) == 3 else "2d", self.reader,
                                 raw=True, original=self.original_mask)
                if self.mask else load_domain_image(pair, "3d" if len(pair.domain_shape) == 3 else "2d", self.reader, raw=True)
            )
            return array[spatial if self.mask else (slice(None), *spatial)]
        spatial_axes = "ZYX" if len(pair.spatial_shape) == 3 else "YX"
        region = pair.region or tuple((0, size) for size in pair.spatial_shape)
        selection_by_axis = {
            axis: slice(s.start + start, s.stop + start)
            for axis, s, (start, _) in zip(spatial_axes, spatial, region, strict=True)
        }
        selection = tuple(selection_by_axis.get(axis, slice(None) if axis == "C" and not self.mask else slice(0, 1)) for axis in axes)
        array = self.reader.read(pair.mask if self.mask else pair.image, role="mask" if self.mask else "image", selection=selection)
        kept = spatial_axes if self.mask else "C" + spatial_axes
        for index in reversed(range(len(axes))):
            if axes[index] not in kept:
                array = np.take(array, 0, axis=index)
                axes = axes[:index] + axes[index + 1:]
        if not self.mask and "C" not in axes:
            array, axes = array[None], "C" + axes
        return array.transpose(tuple(axes.index(axis) for axis in kept))

    def __getitem__(self, selection: tuple[slice, ...]) -> np.ndarray:
        if not isinstance(selection, tuple):
            selection = (selection,)
        if selection == (Ellipsis,):
            selection = (slice(None),) * self.ndim
        spatial = selection if self.mask else selection[1:]
        if len(spatial) != len(self.spatial_shape) or any(not isinstance(s, slice) or s.step not in (None, 1) for s in spatial):
            raise IndexError("Crop arrays require unit-stride spatial slices")
        spatial = tuple(slice(*s.indices(n)[:2]) for s, n in zip(spatial, self.spatial_shape, strict=True))
        if self.center_z is not None:
            if self.mask:
                return self._read_native((slice(self.center_z, self.center_z + 1), *spatial))[0]
            radius = self.context_slices // 2
            planes = []
            for z in range(self.center_z - radius * self.context_stride,
                           self.center_z + radius * self.context_stride + 1, self.context_stride):
                if 0 <= z < self.pair.domain_shape[0]:
                    raw = self._read_native((slice(z, z + 1), *spatial))[:, 0]
                    planes.append(normalize_image(raw, statistics=self.statistics))
                else:
                    planes.append(np.zeros((self.pair.image_channels, *(s.stop - s.start for s in spatial)), np.float32))
            result = np.stack(planes, axis=1).reshape(self.shape[0], *(s.stop - s.start for s in spatial))
        elif self.resampled_shape is None or self.resampled_shape == self.pair.domain_shape:
            raw = self._read_native(spatial)
            result = raw if self.mask else normalize_image(raw, statistics=self.statistics)
        else:
            # Match scipy.zoom(grid_mode=False) globally, not a new local crop grid.
            ratios = np.array([(n - 1) / (m - 1) if m > 1 else 0.0
                               for n, m in zip(self.pair.domain_shape, self.resampled_shape, strict=True)])
            coordinates = [np.arange(s.start, s.stop, dtype=np.float64) * ratio
                           for s, ratio in zip(spatial, ratios, strict=True)]
            native = tuple(slice(max(0, int(np.floor(c[0]))), min(n, int(np.ceil(c[-1])) + 1))
                           for c, n in zip(coordinates, self.pair.domain_shape, strict=True))
            raw = self._read_native(native)
            if self.mask:
                indices = [np.floor(c + 0.5).astype(np.intp) - s.start for c, s in zip(coordinates, native, strict=True)]
                result = raw[np.ix_(*indices)]
            else:
                raw = normalize_image(raw, statistics=self.statistics)
                offset = [c[0] - s.start for c, s in zip(coordinates, native, strict=True)]
                result = np.stack([ndi.affine_transform(channel, np.diag(ratios), offset=offset,
                                   output_shape=tuple(len(c) for c in coordinates), order=1, mode="nearest", prefilter=False)
                                   for channel in raw])
        return result if self.mask else result[selection[0]]

    def __array__(self, dtype: DTypeLike | None = None, copy: bool | None = None) -> np.ndarray:
        result = self[(slice(None),) * self.ndim]
        return np.asarray(result, dtype=dtype).copy() if copy else np.asarray(result, dtype=dtype)


@dataclass
class ResizedCropArray:
    """Crop-first global half-pixel image/nearest-label resize, without full arrays."""

    source: np.ndarray | CropArray | ResizedCropArray
    spatial_shape: tuple[int, ...]
    mask: bool = False

    @property
    def shape(self) -> tuple[int, ...]:
        return self.spatial_shape if self.mask else (self.source.shape[0], *self.spatial_shape)

    def __getitem__(self, selection: tuple[slice, ...]) -> np.ndarray:
        spatial = selection if self.mask else selection[1:]
        native_shape = self.source.shape if self.mask else self.source.shape[1:]
        if tuple(native_shape) == self.spatial_shape:
            return self.source[selection]
        coordinates = []
        ratios = []
        for s, n, m in zip(spatial, native_shape, self.spatial_shape, strict=True):
            start, stop, step = s.indices(m)
            if step != 1 or start >= stop:
                raise IndexError("Resized crop arrays require nonempty unit-stride slices")
            ratio = n / m
            coords = np.arange(start, stop, dtype=np.float64) * ratio
            if not self.mask:
                coords += (ratio - 1) / 2
            coordinates.append(coords)
            ratios.append(ratio)
        native = tuple(slice(max(0, int(np.floor(c[0]))), min(n, int(np.ceil(c[-1])) + 1))
                       for c, n in zip(coordinates, native_shape, strict=True))
        raw = self.source[native if self.mask else (slice(None), *native)]
        if self.mask:
            indices = [np.clip(np.floor(c).astype(np.intp), 0, n - 1) - s.start
                       for c, n, s in zip(coordinates, native_shape, native, strict=True)]
            return raw[np.ix_(*indices)]
        offset = [c[0] - s.start for c, s in zip(coordinates, native, strict=True)]
        result = np.stack([ndi.affine_transform(channel, np.diag(ratios), offset=offset,
                          output_shape=tuple(len(c) for c in coordinates), order=1, mode="nearest", prefilter=False)
                          for channel in raw])
        return result[selection[0]]
