"""What the kernel says about a process (core/processes.py).

The real-process tests run on whichever kernel the suite runs on, so CI covers
/proc and a Mac covers libproc. The /proc parsing also runs against a planted
tree, so a Mac exercises it too.
"""

import os
import subprocess
import sys
import time

import pytest

from kapso.core import processes
from kapso.core.processes import process_commands, process_start


def _sleeper(*extra_args):
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", *extra_args])


def _plant_process(root, pid, *, comm="python3", state="S", start="4242", cmdline=b"python3\0x.py\0"):
    directory = root / str(pid)
    directory.mkdir(parents=True)
    # proc(5): pid (comm) state, then fields 4 to 21, then starttime as 22.
    filler = " ".join(["0"] * 18)
    (directory / "stat").write_bytes(f"{pid} ({comm}) {state} {filler} {start} 0 0\n".encode())
    (directory / "cmdline").write_bytes(cmdline)


def test_a_running_process_keeps_one_start_stamp_until_it_is_gone():
    child = _sleeper()
    started = process_start(child.pid)
    assert started and process_start(child.pid) == started
    assert process_start(child.pid) != process_start(os.getpid())
    child.kill()
    child.wait()
    assert process_start(child.pid) == ""


def test_a_killed_process_nobody_has_reaped_counts_as_gone():
    # A zombie still holds its pid, so asking only "does the pid exist" calls a
    # killed campaign alive for as long as its parent has not collected it.
    child = _sleeper()
    child.kill()
    deadline = time.monotonic() + 5
    while process_start(child.pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert process_start(child.pid) == ""
    child.wait()


def test_commands_list_the_arguments_and_not_the_environment():
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)", "needle-argument.py"],
        env={**os.environ, "NEEDLE_ONLY_IN_ENV": "needle-environ.py"},
    )
    deadline = time.monotonic() + 5
    command = ""
    while "needle-argument.py" not in command and time.monotonic() < deadline:
        command = process_commands().get(child.pid, "")
    child.kill()
    child.wait()
    assert "needle-argument.py" in command
    assert "needle-environ.py" not in command


def test_proc_stat_is_read_past_a_command_name_with_spaces_and_parens(tmp_path, monkeypatch):
    monkeypatch.setattr(processes, "PROC_ROOT", str(tmp_path))
    _plant_process(tmp_path, 10, comm="py (thon) S 99", start="777")
    assert process_start(10) == "777"


@pytest.mark.parametrize("state", ["Z", "X"])
def test_proc_state_of_a_finished_process_is_gone(tmp_path, monkeypatch, state):
    monkeypatch.setattr(processes, "PROC_ROOT", str(tmp_path))
    _plant_process(tmp_path, 10, state=state)
    assert process_start(10) == ""


def test_proc_without_the_pid_is_gone(tmp_path, monkeypatch):
    monkeypatch.setattr(processes, "PROC_ROOT", str(tmp_path))
    assert process_start(10) == ""


def test_proc_commands_join_the_arguments(tmp_path, monkeypatch):
    monkeypatch.setattr(processes, "PROC_ROOT", str(tmp_path))
    _plant_process(tmp_path, 10, cmdline=b"python3\0kapso_eval.py\0--full\0")
    (tmp_path / "self").mkdir()
    assert process_commands() == {10: "python3 kapso_eval.py --full "}


def test_a_kernel_with_no_way_to_ask_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(processes, "PROC_ROOT", str(tmp_path / "missing"))
    monkeypatch.setattr(processes.sys, "platform", "freebsd14")
    assert process_start(os.getpid()) is None
    assert process_commands() == {}


@pytest.mark.skipif(
    sys.platform != "darwin" or os.geteuid() == 0, reason="needs a process macOS refuses to describe"
)
def test_macos_refusing_to_describe_a_process_is_not_a_death():
    # launchd is running; the kernel answers EPERM to anyone but root.
    assert process_start(1) is None
