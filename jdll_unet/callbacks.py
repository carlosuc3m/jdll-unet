"""Generic event callbacks with an Appose-compatible adapter."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CallbackEvent:
    type: str
    message: str | None = None
    current: int | None = None
    maximum: int | None = None
    info: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        payload = dict(self.info)
        payload["type"] = self.type
        if self.message is not None:
            payload.setdefault("message", self.message)
        if self.current is not None:
            payload.setdefault("current", self.current)
        if self.maximum is not None:
            payload.setdefault("maximum", self.maximum)
        return payload

    def appose_kwargs(self) -> dict[str, Any]:
        info = self.payload()
        return {
            "message": self.message or "",
            "current": self.current,
            "maximum": self.maximum,
            "info": info,
        }


def _as_callbacks(callback: Any) -> list[Any]:
    if callback is None:
        return []
    if isinstance(callback, Iterable) and not isinstance(callback, (str, bytes, dict)):
        return list(callback)
    return [callback]


class CallbackDispatcher:
    """Dispatch flat payloads to generic callbacks and Appose task objects."""

    def __init__(self, callback: Any = None) -> None:
        self.callbacks = _as_callbacks(callback)
        self._cancel_requested = False

    def emit(
        self,
        event_type: str,
        *,
        message: str | None = None,
        current: int | None = None,
        maximum: int | None = None,
        **info: Any,
    ) -> bool:
        event = CallbackEvent(event_type, message=message, current=current, maximum=maximum, info=info)
        for callback in self.callbacks:
            result = self._dispatch_one(callback, event)
            if result is False:
                self._cancel_requested = True
        if self.cancel_requested():
            self._cancel_requested = True
        return not self._cancel_requested

    def cancel_requested(self) -> bool:
        if self._cancel_requested:
            return True
        return any(_callback_cancel_requested(callback) for callback in self.callbacks)

    def _dispatch_one(self, callback: Any, event: CallbackEvent) -> Any:
        update = getattr(callback, "update", None)
        if callable(update):
            try:
                return update(**event.appose_kwargs())
            except TypeError:
                return update(event.payload())

        emit = getattr(callback, "emit", None)
        if callable(emit):
            return emit(event.payload())

        if callable(callback):
            return callback(event.payload())

        raise TypeError("callback must be callable, expose .emit(payload), or expose .update(...)")


def _callback_cancel_requested(callback: Any) -> bool:
    is_cancelled = getattr(callback, "is_cancelled", None)
    if callable(is_cancelled):
        return bool(is_cancelled())
    cancelled = getattr(callback, "cancelled", False)
    return bool(cancelled() if callable(cancelled) else cancelled)
