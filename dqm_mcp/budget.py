"""
Guardrail on how many plots the server pulls from the DQM GUI.

Two limits, both configurable through Settings:

* ``max_images_per_call``  — a single tool call may not touch more panels than
  this. The largest shift-layout group has 11 panels, so the default of 12
  admits every plot while refusing anything unbounded.
* ``max_fetches_per_hour`` — network requests to the GUI in a rolling window.
  Only real network fetches count; a PNG already on disk or a hit in the
  response cache is free.

The budget records an attempt when it is reserved, before the request is
made, so failed requests still count — they hit the GUI just the same.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timezone


class BudgetError(RuntimeError):
    """Raised when a call would exceed the fetch guardrail."""


class FetchBudget:
    def __init__(
        self,
        max_per_window: int,
        max_per_call: int,
        window_seconds: int = 3600,
        clock=time.time,
    ):
        if max_per_window < 0 or max_per_call < 1:
            raise ValueError("max_per_window must be >= 0 and max_per_call >= 1")
        self.max_per_window = max_per_window
        self.max_per_call = max_per_call
        self.window_seconds = window_seconds
        self._clock = clock
        self._stamps: deque[float] = deque()
        self._lock = threading.Lock()

    # -- internals -----------------------------------------------------------

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._stamps and self._stamps[0] <= cutoff:
            self._stamps.popleft()

    def _resets_at(self, now: float) -> float | None:
        return (self._stamps[0] + self.window_seconds) if self._stamps else None

    # -- public --------------------------------------------------------------

    def check_call(self, n_images: int) -> None:
        """Refuse a call that would handle more panels than the per-call cap."""
        if n_images > self.max_per_call:
            raise BudgetError(
                f"This plot has {n_images} panels, above the per-call limit of "
                f"{self.max_per_call} (DQM_MCP_MAX_IMAGES_PER_CALL)."
            )

    def reserve(self, n_network: int) -> None:
        """
        Claim *n_network* fetches from the rolling window, or raise.

        Atomic: either all *n_network* are recorded or none are, so a
        multi-panel plot is never half-fetched because the budget ran out
        midway.
        """
        if n_network <= 0:
            return
        with self._lock:
            now = self._clock()
            self._prune(now)
            used = len(self._stamps)
            if used + n_network > self.max_per_window:
                remaining = max(self.max_per_window - used, 0)
                resets = self._resets_at(now)
                when = (
                    datetime.fromtimestamp(resets, tz=timezone.utc).strftime("%H:%M:%S UTC")
                    if resets else "now"
                )
                raise BudgetError(
                    f"Fetch guardrail: this call needs {n_network} network fetch(es) "
                    f"but only {remaining} of {self.max_per_window} remain in the last "
                    f"{self.window_seconds // 60} min (DQM_MCP_MAX_FETCHES_PER_HOUR). "
                    f"Oldest fetch expires at {when}. Cached plots are still served."
                )
            for _ in range(n_network):
                self._stamps.append(now)

    def status(self) -> dict:
        with self._lock:
            now = self._clock()
            self._prune(now)
            used = len(self._stamps)
            resets = self._resets_at(now)
            return {
                "used_last_window": used,
                "remaining": max(self.max_per_window - used, 0),
                "max_per_window": self.max_per_window,
                "window_seconds": self.window_seconds,
                "oldest_expires_in_s": round(resets - now) if resets else None,
                "max_images_per_call": self.max_per_call,
            }


_budget: FetchBudget | None = None


def get_budget() -> FetchBudget:
    global _budget
    if _budget is None:
        from .config import get_settings
        s = get_settings()
        _budget = FetchBudget(
            max_per_window=s.max_fetches_per_hour,
            max_per_call=s.max_images_per_call,
        )
    return _budget


def reset_budget() -> None:
    global _budget
    _budget = None
