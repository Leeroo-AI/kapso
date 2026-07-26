"""Coordinator-issued Linux clocks unavailable at the adapter boundary."""

from __future__ import annotations

import time


class _SystemRunActionClock:
    """Read the clocks used to authorize release and containment transitions."""

    def boottime_nanoseconds(self) -> int:
        return time.clock_gettime_ns(time.CLOCK_BOOTTIME)

    def realtime_nanoseconds(self) -> int:
        return time.clock_gettime_ns(time.CLOCK_REALTIME)
