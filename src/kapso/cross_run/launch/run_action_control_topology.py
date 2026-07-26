"""Closed semantic topology of one run-action control directory."""

from __future__ import annotations

from enum import Enum


class RunActionControlDirectoryTopology(str, Enum):
    """Every valid complete set of durable control entries."""

    EMPTY = "empty"
    RELEASED = "released"
    TIMED_OUT = "timed_out"

    @property
    def entries(self) -> tuple[str, ...]:
        return {
            RunActionControlDirectoryTopology.EMPTY: (),
            RunActionControlDirectoryTopology.RELEASED: ("release",),
            RunActionControlDirectoryTopology.TIMED_OUT: ("release", "timeout"),
        }[self]


__all__ = ["RunActionControlDirectoryTopology"]
