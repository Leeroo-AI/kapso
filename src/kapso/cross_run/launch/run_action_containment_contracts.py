"""Transient exact evidence for one timeout-containment signal attempt."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from kapso.cross_run.launch.run_action_barrier_contracts import (
    RunActionBarrierRunningContainerObservation,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionTerminalObservation,
)


class RunActionTimeoutContainmentSignal(str, Enum):
    """The only two signals admitted by timeout containment."""

    TERMINATE = "SIGTERM"
    KILL = "SIGKILL"


class RunActionTimeoutContainmentState(str, Enum):
    """Stable exact state observed after the signal command returned."""

    RUNNING = "running"
    TERMINAL = "terminal"


@dataclass(frozen=True)
class RunActionTimeoutContainmentResult:
    """One command result joined to a stable post-signal occurrence."""

    signal: RunActionTimeoutContainmentSignal | None
    selected_at_boottime_nanoseconds: int | None
    signal_dispatch_confirmed: bool
    state: RunActionTimeoutContainmentState
    running_observation: RunActionBarrierRunningContainerObservation | None
    terminal_observation: RunActionTerminalObservation | None

    def __post_init__(self) -> None:
        running_present = (
            type(self.running_observation)
            is RunActionBarrierRunningContainerObservation
        )
        terminal_present = (
            type(self.terminal_observation) is RunActionTerminalObservation
        )
        signal_present = type(self.signal) is RunActionTimeoutContainmentSignal
        selection_present = (
            type(self.selected_at_boottime_nanoseconds) is int
            and self.selected_at_boottime_nanoseconds > 0
        )
        if (
            (signal_present, selection_present) not in {(False, False), (True, True)}
            or type(self.signal_dispatch_confirmed) is not bool
            or type(self.state) is not RunActionTimeoutContainmentState
            or (running_present, terminal_present)
            != {
                RunActionTimeoutContainmentState.RUNNING: (True, False),
                RunActionTimeoutContainmentState.TERMINAL: (False, True),
            }[self.state]
            or (self.running_observation is not None and not running_present)
            or (self.terminal_observation is not None and not terminal_present)
            or (
                not signal_present
                and (
                    self.signal is not None
                    or self.selected_at_boottime_nanoseconds is not None
                    or self.signal_dispatch_confirmed
                    or self.state is not RunActionTimeoutContainmentState.TERMINAL
                )
            )
            or (
                signal_present
                and not self.signal_dispatch_confirmed
                and self.state is not RunActionTimeoutContainmentState.TERMINAL
            )
        ):
            raise ValueError(
                "timeout containment result lacks one exact post-signal occurrence"
            )


__all__ = [
    "RunActionTimeoutContainmentResult",
    "RunActionTimeoutContainmentSignal",
    "RunActionTimeoutContainmentState",
]
