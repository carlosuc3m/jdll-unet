"""Bounded single-producer loading without DataLoader worker processes."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from contextvars import copy_context
from queue import Empty, SimpleQueue
from threading import get_ident
from typing import Any

from .image_reading import ImageReadSession


class BatchPrefetcher:
    """Prepare at most one batch ahead; the owner must close before its I/O session."""

    def __init__(self, loader: Iterable[Any], *, read_session: ImageReadSession | None = None) -> None:
        self.loader = loader
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jdll-batch")
        self.pending: Future[Any] | None = None
        self.closed = False
        self.active = False
        self.owner_thread = get_ident()
        self.read_session = read_session
        self.emit = read_session.emit if read_session is not None else None
        self.events: SimpleQueue[tuple[tuple[Any, ...], dict[str, Any]]] = SimpleQueue()
        if read_session is not None and self.emit is not None:
            read_session.emit = self._emit

    def _emit(self, *args: Any, **kwargs: Any) -> Any:
        if get_ident() == self.owner_thread:
            return self.emit(*args, **kwargs) if self.emit is not None else None
        self.events.put((args, kwargs))
        return True

    def _flush_events(self) -> None:
        while True:
            try:
                args, kwargs = self.events.get_nowait()
            except Empty:
                return
            if self.emit is not None:
                self.emit(*args, **kwargs)

    def __iter__(self) -> Iterator[Any]:
        if self.closed or self.active:
            raise RuntimeError("Prefetch loader is closed or already being consumed")
        self.active = True
        # Iterator creation stays on the main thread, preserving DataLoader RNG
        # initialization. The producer inherits the current image-reading session.
        try:
            iterator = iter(self.loader)
            context = copy_context()
            sentinel = object()

            def produce() -> Any:
                return context.run(next, iterator, sentinel)

            self.pending = self.executor.submit(produce)
            while True:
                batch = self.pending.result()
                self._flush_events()
                if batch is sentinel:
                    return
                self.pending = self.executor.submit(produce)
                yield batch
        finally:
            if self.pending is not None:
                self.pending.cancel()
                # Drain an in-flight read before set_epoch or session cleanup.
                if not self.pending.cancelled():
                    with suppress(Exception):
                        self.pending.result()
            self.pending = None
            self.active = False

    def close(self, *, flush_events: bool = True) -> None:
        if self.closed:
            return
        self.closed = True
        if self.pending is not None:
            self.pending.cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)
        try:
            if flush_events:
                self._flush_events()
        finally:
            if self.read_session is not None and self.read_session.emit == self._emit:
                self.read_session.emit = self.emit
