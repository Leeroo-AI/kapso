"""Bounded local Git command boundary shared by cross-run capture paths."""

from __future__ import annotations

from pathlib import Path

from kapso.cross_run.github.command import (
    CommandOutputKind,
    CommandRequest,
    CommandResult,
    CommandRunner,
    SubprocessCommandRunner,
)


class BoundedGitCommand:
    """Run local Git with the same bounded subprocess policy as GitHub I/O."""

    def __init__(
        self,
        *,
        timeout_seconds: int,
        maximum_output_bytes: int,
        runner: CommandRunner | None = None,
    ) -> None:
        if type(timeout_seconds) is not int or timeout_seconds <= 0:
            raise ValueError("Git command timeout must be a positive integer")
        if type(maximum_output_bytes) is not int or maximum_output_bytes <= 0:
            raise ValueError("Git command output limit must be a positive integer")
        self.timeout_seconds = timeout_seconds
        self.maximum_output_bytes = maximum_output_bytes
        self.runner = runner if runner is not None else SubprocessCommandRunner()

    def run(
        self,
        workspace: Path,
        arguments: tuple[str, ...],
        *,
        output_kind: CommandOutputKind,
    ) -> CommandResult:
        """Run one fixed-argv Git operation and retain redacted failures."""
        return self.runner.run(
            CommandRequest(
                argv=("git", *arguments),
                cwd=workspace,
                timeout_seconds=self.timeout_seconds,
                output_kind=output_kind,
                maximum_output_bytes=self.maximum_output_bytes,
                capture_failure=True,
            )
        )
