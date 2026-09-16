"""Session-scoped, header-validated routing for the supported image formats."""

from __future__ import annotations

import struct
import zlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import tifffile
from PIL import Image

from .errors import DataFormatError

T = TypeVar("T")
FORMATS = {".tif": "TIFF", ".tiff": "TIFF", ".png": "PNG", ".bmp": "BMP", ".jpg": "JPEG", ".jpeg": "JPEG"}
DECODER_ERRORS = (
    OSError,
    ValueError,
    SyntaxError,
    NotImplementedError,
    struct.error,
    zlib.error,
    tifffile.TiffFileError,
    Image.DecompressionBombError,
)


def _signature_format(path: Path) -> str | None:
    with path.open("rb") as stream:
        header = stream.read(8)
    if header[:4] in (b"II\x2a\x00", b"MM\x00\x2a", b"II\x2b\x00", b"MM\x00\x2b"):
        return "TIFF"
    if header == b"\x89PNG\r\n\x1a\n":
        return "PNG"
    if header[:3] == b"\xff\xd8\xff":
        return "JPEG"
    return "BMP" if header[:2] == b"BM" else None


def _open_format(path: Path, file_format: str) -> Any:
    return tifffile.TiffFile(path) if file_format == "TIFF" else Image.open(path, formats=[file_format])


def decode_pixels(reader: Any, file_format: str, role: str, *, memmap: bool = False) -> np.ndarray:
    if file_format == "TIFF":
        series = reader.series[0]
        mapped = memmap and series.dataoffset is not None and series[0].is_memmappable
        return series.asarray(out="memmap" if mapped else None)
    if reader.mode == "P" and role == "image":
        if isinstance(reader.info.get("transparency"), bytes):
            return np.asarray(reader.convert("RGBA").convert("RGB"))
        return np.asarray(reader.convert("RGB"))
    return np.asarray(reader)


class ImageReadSession:
    def __init__(self, emit: Callable[..., Any] | None = None) -> None:
        self.emit = emit
        self.preferences: dict[tuple[str, str, str], str] = {}
        self.notices: set[tuple[tuple[str, str, str], str]] = set()
        self.domain_reader: Any = None

    def __getstate__(self) -> dict[str, Any]:
        # DataLoader workers need routing state, not UI callbacks or another pixel cache.
        return {**vars(self), "emit": None, "domain_reader": None}

    def read(
        self, path: Path | str, operation: Callable[[Any, str], T], *, role: str = "image", pixels: bool = True
    ) -> T:
        path = Path(path)
        if role not in {"image", "mask"}:
            raise ValueError(f"Unknown image reading role: {role}")
        nominal = FORMATS.get(path.suffix.lower())
        if nominal is None or (role == "mask" and nominal == "JPEG"):
            raise DataFormatError(f"Unsupported {role} extension: {path}; JPEG masks are not supported")
        if not path.is_file():
            raise DataFormatError(f"Image file does not exist or is not a file: {path}")
        key = (str(path.absolute().parent), path.suffix.lower(), role)
        expected = self.preferences.get(key, nominal)
        actual = expected
        try:
            reader = _open_format(path, expected)
        except DECODER_ERRORS as error:
            try:
                detected = _signature_format(path)
            except OSError as probe_error:
                raise DataFormatError(f"Cannot inspect {path} after {expected} reader failed: {error}") from probe_error
            if detected is None or detected == expected:
                detail = "invalid/unsupported signature" if detected is None else f"confirmed {detected}"
                raise DataFormatError(f"Cannot read {path} as {expected} ({detail}): {error}") from error
            actual = detected
            try:
                reader = _open_format(path, actual)
            except DECODER_ERRORS as corrected_error:
                raise DataFormatError(
                    f"Cannot read {path}: attempted {expected}, detected {actual}: {corrected_error}"
                ) from corrected_error
        try:
            with reader:
                if role == "mask" and actual == "JPEG":
                    raise DataFormatError(f"JPEG masks are not supported because compression changes labels: {path}")
                if actual == "TIFF" and len(reader.series) != 1:
                    raise DataFormatError(
                        f"Unsupported multiple TIFF series: {path}; export one series with explicit axes"
                    )
                if actual != "TIFF" and getattr(reader, "n_frames", 1) != 1:
                    raise DataFormatError(
                        f"Unsupported multiple raster frames: {path}; export a TIFF stack with explicit axes"
                    )
                result = operation(reader, actual)
        except DataFormatError as error:
            raise DataFormatError(f"Cannot read {path} as {actual}: {error}") from error
        except DECODER_ERRORS as error:
            raise DataFormatError(f"Cannot decode {path}: attempted {expected}, actual {actual}: {error}") from error
        # Metadata inspection alone cannot prove that a corrected pixel read works.
        if pixels and actual != expected:
            self.preferences[key] = actual
            notice = (key, actual)
            if notice not in self.notices:
                self.notices.add(notice)
                if self.emit is not None:
                    self.emit(
                        "warning",
                        reason="image_reader_correction",
                        path=str(path),
                        role=role,
                        extension=key[1],
                        actual_format=actual,
                        message=f"Detected {actual} data in {key[1]} files under {path.parent}; using the {actual} reader.",
                    )
        return result

    def pixels(self, path: Path | str, *, role: str = "image", memmap: bool = False) -> np.ndarray:
        return self.read(path, lambda reader, fmt: decode_pixels(reader, fmt, role, memmap=memmap), role=role)


_active_session: ContextVar[ImageReadSession | None] = ContextVar("image_read_session", default=None)


def current_read_session() -> ImageReadSession:
    return _active_session.get() or ImageReadSession()


@contextmanager
def image_reading_session(emit: Callable[..., Any] | None = None) -> Iterator[ImageReadSession]:
    session = ImageReadSession(emit)
    token = _active_session.set(session)
    try:
        yield session
    finally:
        session.domain_reader = None
        _active_session.reset(token)
