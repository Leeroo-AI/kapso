import io
import shutil
import sys
import time
from pathlib import Path

import pytest

import kapso.cross_run.process as process_module
from kapso.cross_run.process import (
    BoundedProcessError,
    BoundedProcessOutcome,
    BoundedProcessRequest,
    BoundedProcessRunner,
    bounded_process_stream_observations_are_canonical,
    canonicalize_bounded_process_stream_observations,
)


def _run_python(
    tmp_path,
    source,
    *,
    timeout_seconds=5,
    stdout_limit=1024,
    stderr_limit=1024,
):
    return BoundedProcessRunner().run(
        BoundedProcessRequest(
            argv=(sys.executable, "-c", source),
            trusted_root=tmp_path.resolve(),
            cwd=tmp_path,
            timeout_seconds=timeout_seconds,
            cleanup_timeout_seconds=1,
            stdout_byte_limit=stdout_limit,
            stderr_byte_limit=stderr_limit,
            environment={},
        )
    )


def test_process_runs_argv_directly_and_preserves_separate_streams(tmp_path):
    metacharacters = "$(touch should-not-exist); `false`"
    result = BoundedProcessRunner().run(
        BoundedProcessRequest(
            argv=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1]); print('error', file=sys.stderr)",
                metacharacters,
            ),
            trusted_root=tmp_path.resolve(),
            cwd=tmp_path,
            timeout_seconds=5,
            cleanup_timeout_seconds=1,
            stdout_byte_limit=1024,
            stderr_byte_limit=1024,
            environment={"INHERITED_SECRET": "absent"},
        )
    )

    assert result.outcome is BoundedProcessOutcome.COMPLETED
    assert result.returncode == 0
    assert result.stdout == f"{metacharacters}\n".encode()
    assert result.stderr == b"error\n"
    assert not (tmp_path / "should-not-exist").exists()


def test_process_receives_only_the_explicit_environment(tmp_path):
    environment_executable = shutil.which("env")
    assert environment_executable is not None
    result = BoundedProcessRunner().run(
        BoundedProcessRequest(
            argv=(environment_executable,),
            trusted_root=tmp_path.resolve(),
            cwd=tmp_path,
            timeout_seconds=5,
            cleanup_timeout_seconds=1,
            stdout_byte_limit=1024,
            stderr_byte_limit=1024,
            environment={},
        )
    )

    assert result.stdout == b""


@pytest.mark.parametrize(
    ("source", "stdout_limit", "stderr_limit", "outcome"),
    (
        (
            "import sys; sys.stdout.write('x' * 9)",
            8,
            1024,
            BoundedProcessOutcome.STDOUT_LIMIT_EXCEEDED,
        ),
        (
            "import sys; sys.stderr.write('x' * 9)",
            1024,
            8,
            BoundedProcessOutcome.STDERR_LIMIT_EXCEEDED,
        ),
    ),
)
def test_process_enforces_each_stream_limit_independently(
    tmp_path,
    source,
    stdout_limit,
    stderr_limit,
    outcome,
):
    result = _run_python(
        tmp_path,
        source,
        stdout_limit=stdout_limit,
        stderr_limit=stderr_limit,
    )

    assert result.outcome is outcome
    assert len(result.stdout) <= stdout_limit
    assert len(result.stderr) <= stderr_limit
    canonical_stdout, canonical_stderr = (
        canonicalize_bounded_process_stream_observations(
            outcome=result.outcome,
            stdout_bytes_observed=result.stdout_bytes_observed,
            stderr_bytes_observed=result.stderr_bytes_observed,
            stdout_byte_limit=stdout_limit,
            stderr_byte_limit=stderr_limit,
        )
    )
    assert bounded_process_stream_observations_are_canonical(
        outcome=result.outcome,
        stdout_bytes_observed=canonical_stdout,
        stderr_bytes_observed=canonical_stderr,
        stdout_byte_limit=stdout_limit,
        stderr_byte_limit=stderr_limit,
    )
    if outcome is BoundedProcessOutcome.STDOUT_LIMIT_EXCEEDED:
        assert result.stdout_bytes_observed > stdout_limit
        assert canonical_stdout == stdout_limit + 1
    else:
        assert result.stderr_bytes_observed > stderr_limit
        assert canonical_stderr == stderr_limit + 1


def test_process_timeout_is_a_typed_terminal_outcome(tmp_path):
    result = BoundedProcessRunner().run(
        BoundedProcessRequest(
            argv=(sys.executable, "-c", "import time; time.sleep(60)"),
            trusted_root=tmp_path.resolve(),
            cwd=tmp_path,
            timeout_seconds=1,
            cleanup_timeout_seconds=1,
            stdout_byte_limit=1024,
            stderr_byte_limit=1024,
            environment={},
        )
    )

    assert result.outcome is BoundedProcessOutcome.TIMED_OUT
    assert result.duration_seconds < 5.0
    assert result.returncode != 0


def test_nonzero_exit_is_a_completed_observation(tmp_path):
    result = _run_python(tmp_path, "raise SystemExit(7)")

    assert result.outcome is BoundedProcessOutcome.COMPLETED
    assert result.returncode == 7


def _process_is_dead(process_id):
    deadline = time.monotonic() + 1.0
    process_state_path = Path(f"/proc/{process_id}/stat")
    while process_state_path.exists() and time.monotonic() < deadline:
        if process_state_path.read_text().split()[2] == "Z":
            return True
        time.sleep(0.01)
    return not process_state_path.exists()


def test_stream_overflow_kills_the_complete_process_group(tmp_path):
    result = _run_python(
        tmp_path,
        "import subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        "print(child.pid, flush=True); print('x' * 1024, flush=True); time.sleep(60)",
        stdout_limit=32,
    )
    child_process_id = int(result.stdout.splitlines()[0])

    assert result.outcome is BoundedProcessOutcome.STDOUT_LIMIT_EXCEEDED
    assert _process_is_dead(child_process_id)


def test_timeout_kills_descendant_after_the_group_leader_exits(tmp_path):
    result = _run_python(
        tmp_path,
        "import subprocess, sys; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        "print(child.pid, flush=True)",
        timeout_seconds=1,
    )
    child_process_id = int(result.stdout.strip())

    assert result.outcome is BoundedProcessOutcome.TIMED_OUT
    assert _process_is_dead(child_process_id)


def test_normal_completion_kills_a_detached_group_descendant(tmp_path):
    result = _run_python(
        tmp_path,
        "import subprocess, sys; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], "
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, "
        "stderr=subprocess.DEVNULL); print(child.pid, flush=True)",
    )
    child_process_id = int(result.stdout.strip())

    assert result.outcome is BoundedProcessOutcome.COMPLETED
    assert _process_is_dead(child_process_id)


def test_fast_natural_exit_does_not_mask_a_stream_limit(tmp_path, monkeypatch):
    class _DelayedBuffer(io.BytesIO):
        def write(self, payload):
            time.sleep(0.2)
            return super().write(payload)

    monkeypatch.setattr(process_module.io, "BytesIO", _DelayedBuffer)
    result = _run_python(
        tmp_path,
        "import os; os.write(1, b'xx'); os._exit(0)",
        stdout_limit=1,
    )

    assert result.outcome is BoundedProcessOutcome.STDOUT_LIMIT_EXCEEDED


def test_sink_failure_unconditionally_kills_the_owned_process(tmp_path, monkeypatch):
    marker_path = tmp_path / "process-id"

    class _FailingBuffer(io.BytesIO):
        def write(self, _payload):
            raise OSError("injected sink failure")

    monkeypatch.setattr(process_module.io, "BytesIO", _FailingBuffer)
    with pytest.raises(OSError, match="injected sink failure"):
        _run_python(
            tmp_path,
            "import os, pathlib, sys, time; "
            "pathlib.Path('process-id').write_text(str(os.getpid())); "
            "print('trigger', flush=True); time.sleep(60)",
        )

    assert _process_is_dead(int(marker_path.read_text()))


def test_process_can_stream_bounded_output_to_absent_files(tmp_path):
    stdout_path = (tmp_path / "stdout.bin").resolve()
    stderr_path = (tmp_path / "stderr.bin").resolve()
    result = BoundedProcessRunner().run(
        BoundedProcessRequest(
            argv=(
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write(b'output'); "
                "sys.stderr.buffer.write(b'diagnostic')",
            ),
            trusted_root=tmp_path.resolve(),
            cwd=tmp_path,
            timeout_seconds=5,
            cleanup_timeout_seconds=1,
            stdout_byte_limit=1024,
            stderr_byte_limit=1024,
            environment={},
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
    )

    assert result.outcome is BoundedProcessOutcome.COMPLETED
    assert result.stdout == b""
    assert result.stderr == b""
    assert stdout_path.read_bytes() == b"output"
    assert stderr_path.read_bytes() == b"diagnostic"


@pytest.mark.parametrize(
    "request_factory",
    (
        lambda root: BoundedProcessRequest(
            argv=(),
            trusted_root=root.resolve(),
            cwd=root,
            timeout_seconds=1,
            cleanup_timeout_seconds=1,
            stdout_byte_limit=1,
            stderr_byte_limit=1,
            environment={},
        ),
        lambda root: BoundedProcessRequest(
            argv=(sys.executable,),
            trusted_root=root.resolve(),
            cwd=root,
            timeout_seconds=True,
            cleanup_timeout_seconds=1,
            stdout_byte_limit=1,
            stderr_byte_limit=1,
            environment={},
        ),
        lambda root: BoundedProcessRequest(
            argv=(sys.executable,),
            trusted_root=root.resolve(),
            cwd=root,
            timeout_seconds=1,
            cleanup_timeout_seconds=1,
            stdout_byte_limit=1,
            stderr_byte_limit=1,
            environment={},
            stdout_path=Path("relative"),
        ),
        lambda root: BoundedProcessRequest(
            argv=[sys.executable],
            trusted_root=root.resolve(),
            cwd=root,
            timeout_seconds=1,
            cleanup_timeout_seconds=1,
            stdout_byte_limit=1,
            stderr_byte_limit=1,
            environment={},
        ),
    ),
)
def test_process_request_rejects_invalid_authority(tmp_path, request_factory):
    with pytest.raises(BoundedProcessError):
        request_factory(tmp_path)
