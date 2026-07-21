from pathlib import Path

import pytest

from kapso.cross_run.git_command import BoundedGitCommand
from kapso.cross_run.github.command import CommandOutputKind, GitHubCommandError


def test_local_git_stdout_is_bounded(tmp_path: Path):
    command = BoundedGitCommand(timeout_seconds=5, maximum_output_bytes=8)

    with pytest.raises(GitHubCommandError, match="stdout exceeds configured limit"):
        command.run(
            tmp_path,
            ("-c", "alias.emit=!printf 0123456789", "emit"),
            output_kind=CommandOutputKind.BINARY,
        )


def test_local_git_timeout_kills_the_process(tmp_path: Path):
    command = BoundedGitCommand(timeout_seconds=1, maximum_output_bytes=1024)

    with pytest.raises(GitHubCommandError, match="command timed out"):
        command.run(
            tmp_path,
            ("-c", "alias.pause=!sleep 5", "pause"),
            output_kind=CommandOutputKind.BINARY,
        )


def test_local_git_failure_diagnostics_are_redacted(tmp_path: Path):
    secret = "gho_super_secret_diagnostic"
    (tmp_path / "diagnostic.txt").write_text(secret, encoding="utf-8")
    command = BoundedGitCommand(timeout_seconds=5, maximum_output_bytes=1024)

    result = command.run(
        tmp_path,
        (
            "-c",
            "alias.fail=!sh -c 'cat diagnostic.txt >&2; exit 7'",
            "fail",
        ),
        output_kind=CommandOutputKind.BINARY,
    )

    diagnostic = result.stderr.decode("utf-8")
    assert result.returncode == 7
    assert secret not in diagnostic
    assert "[REDACTED]" in diagnostic
