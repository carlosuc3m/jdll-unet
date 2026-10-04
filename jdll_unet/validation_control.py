"""Transport-independent, run-scoped one-shot full-validation requests."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from typing import Any
from uuid import uuid4


class FullValidationController:
    """Thread-safe incoming control. Re-delivering a token never repeats a request."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: list[str] = []
        self._seen: set[str] = set()

    def request_full_validation(self, request_id: str | None = None) -> str:
        token = request_id if request_id is not None else uuid4().hex
        if not isinstance(token, str) or not token.strip():
            raise ValueError("Full-validation request IDs must be nonempty strings")
        with self._lock:
            if token not in self._seen:
                self._seen.add(token)
                self._pending.append(token)
        return token

    def poll(self) -> tuple[str, ...]:
        with self._lock:
            pending = tuple(self._pending)
            self._pending.clear()
        return pending


ValidationControl = FullValidationController | Callable[[], str | Iterable[str] | None]


class FullValidationSchedule:
    """Snapshot requests at the boundary; requests arriving later stay pending."""

    def __init__(self, interval: int, control: ValidationControl | None, emit: Callable[..., Any],
                 *, next_epoch: int | None = None) -> None:
        self.interval = interval
        self.next_epoch = (next_epoch if next_epoch is not None else interval) if interval else None
        self.control = control
        self.emit = emit
        self.run_id = uuid4().hex
        self.seen: set[str] = set()
        self.pending: list[str] = []
        self.active: dict[str, Any] | None = None

    def event(self, status: str, **payload: Any) -> None:
        self.emit("full_validation", status=status, run_id=self.run_id, **payload)

    def poll(self) -> None:
        result = self.control.poll() if isinstance(self.control, FullValidationController) else self.control() if self.control else None
        if result is not None and not isinstance(result, (str, Iterable)):
            raise ValueError("Validation control must return request IDs, not callback/cancellation booleans")
        tokens = (result,) if isinstance(result, str) else tuple(result or ())
        for token in tokens:
            if not isinstance(token, str) or not token.strip():
                raise ValueError("Validation control must return request IDs, not callback/cancellation booleans")
            if token not in self.seen:
                self.seen.add(token)
                self.pending.append(token)
                self.event("pending", request_id=token, message="Full validation requested for the next available epoch end.")

    def boundary(self, epoch: int) -> dict[str, Any] | None:
        self.poll()
        periodic = self.next_epoch is not None and epoch >= self.next_epoch
        if not self.pending and not periodic:
            return None
        requests, self.pending = self.pending, []
        self.active = {"epoch": epoch, "request_ids": requests, "pass_id": uuid4().hex,
                       "reason": "requested_and_periodic" if requests and periodic else "requested" if requests else "periodic"}
        self.event("accepted", **self.active, message=f"Full validation scheduled after epoch {epoch}.")
        return dict(self.active)

    def started(self) -> None:
        assert self.active is not None
        epoch = self.active["epoch"]
        self.next_epoch = epoch + self.interval if self.interval else None
        self.event("started", **self.active, next_epoch=self.next_epoch,
                   message=f"Starting full validation after epoch {epoch}.")

    def finish(self, status: str, **payload: Any) -> None:
        assert self.active is not None
        active = self.active
        self.active = None
        self.event(status, **active, next_epoch=self.next_epoch, **payload)

    def state_dict(self) -> dict[str, Any]:
        # Manual tokens belong to a live run and must never be replayed on resume.
        return {"interval": self.interval, "next_epoch": self.next_epoch}

    def close(self, *, cancelled: bool = False) -> None:
        if self.active is not None:
            self.finish("cancelled" if cancelled else "failed", message="Full validation did not complete.")
        self.poll()
        self.event("closed", unserved_request_ids=list(self.pending), cancelled=cancelled,
                   message="Full-validation control closed; no further epoch boundary is available.")
        self.pending.clear()
