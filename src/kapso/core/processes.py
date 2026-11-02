"""What the kernel can say about a process, on Linux and on macOS.

Linux answers through /proc. macOS has no /proc, so it is asked through
libproc and sysctl rather than `ps`: `ps` is setuid on macOS, and a sandboxed
caller (a coding agent running kapso, for one) is refused permission to exec it.
"""

import ctypes
import errno
import os
import struct
import sys
from typing import Dict, Optional

# Where the kernel lists live processes.
PROC_ROOT = "/proc"

# Field positions of /proc/<pid>/stat after the ")" that ends the command name
# (proc(5) numbers state 3 and starttime 22).
_STAT_STATE_INDEX = 0
_STAT_START_TIME_INDEX = 19
_STAT_NO_LONGER_RUNNING = ("Z", "X")

# macOS <sys/proc_info.h>: the struct proc_bsdinfo that PROC_PIDTBSDINFO fills,
# and where its pbi_start_tvsec and pbi_start_tvusec sit.
_PROC_PIDTBSDINFO = 3
_PROC_BSDINFO_SIZE = 136
_PROC_BSDINFO_START_OFFSET = 120

# macOS <sys/sysctl.h>: the process's own arguments.
_CTL_KERN = 1
_KERN_PROCARGS2 = 49
_KERN_ARGMAX = 8


def process_start(pid: int) -> Optional[str]:
    """The kernel's start stamp for a running pid, which tells this process
    from a later one handed the same pid. Empty when no such process is
    running (a zombie counts as gone), None where this machine cannot say."""
    if os.path.isdir(PROC_ROOT):
        return _linux_process_start(pid)
    if sys.platform == "darwin":
        return _darwin_process_start(pid)
    return None


def process_commands() -> Dict[int, str]:
    """Every running process's command line, by pid. Empty where this machine
    cannot list them."""
    if os.path.isdir(PROC_ROOT):
        return _linux_process_commands()
    if sys.platform == "darwin":
        return _darwin_process_commands()
    return {}


def _linux_process_start(pid: int) -> str:
    stat_path = os.path.join(PROC_ROOT, str(pid), "stat")
    if not os.path.exists(stat_path):
        return ""
    with open(stat_path, "rb") as handle:
        fields = handle.read().rpartition(b")")[2].decode("ascii").split()
    if fields[_STAT_STATE_INDEX] in _STAT_NO_LONGER_RUNNING:
        return ""
    return fields[_STAT_START_TIME_INDEX]


def _linux_process_commands() -> Dict[int, str]:
    commands = {}
    for entry in os.listdir(PROC_ROOT):
        if not entry.isdigit():
            continue
        cmdline_path = os.path.join(PROC_ROOT, entry, "cmdline")
        if not os.path.exists(cmdline_path):
            continue
        with open(cmdline_path, "rb") as handle:
            commands[int(entry)] = handle.read().replace(b"\0", b" ").decode(
                "utf-8", "replace"
            )
    return commands


def _darwin_process_start(pid: int) -> Optional[str]:
    libc = ctypes.CDLL(None, use_errno=True)
    buffer = ctypes.create_string_buffer(_PROC_BSDINFO_SIZE)
    written = libc.proc_pidinfo(
        ctypes.c_int(pid),
        ctypes.c_int(_PROC_PIDTBSDINFO),
        ctypes.c_uint64(0),
        buffer,
        ctypes.c_int(_PROC_BSDINFO_SIZE),
    )
    if written == _PROC_BSDINFO_SIZE:
        seconds, microseconds = struct.unpack_from(
            "=QQ", buffer.raw, _PROC_BSDINFO_START_OFFSET
        )
        return f"{seconds}.{microseconds:06d}"
    # ESRCH covers a zombie too. Anything else, EPERM above all, is a process
    # this caller may not look at, which says nothing about whether it runs.
    return "" if ctypes.get_errno() == errno.ESRCH else None


def _darwin_process_commands() -> Dict[int, str]:
    libc = ctypes.CDLL(None, use_errno=True)
    # Room for pids that appear between the count and the listing.
    listed = libc.proc_listallpids(None, 0)
    if listed < 0:
        return {}
    pids = (ctypes.c_int * (listed * 2))()
    count = libc.proc_listallpids(pids, ctypes.sizeof(pids))
    # Every command fits in ARGMAX, so one buffer serves the whole listing.
    arguments = ctypes.create_string_buffer(_darwin_sysctl_int(libc, _KERN_ARGMAX))
    commands = {}
    for pid in pids[:count]:
        command = _darwin_process_command(libc, pid, arguments)
        if command is not None:
            commands[pid] = command
    return commands


def _darwin_sysctl_int(libc: ctypes.CDLL, name: int) -> int:
    mib = (ctypes.c_int * 2)(_CTL_KERN, name)
    value = ctypes.c_int(0)
    size = ctypes.c_size_t(ctypes.sizeof(value))
    if libc.sysctl(mib, 2, ctypes.byref(value), ctypes.byref(size), None, ctypes.c_size_t(0)):
        raise OSError(ctypes.get_errno(), f"sysctl kern {name}")
    return value.value


def _darwin_process_command(
    libc: ctypes.CDLL, pid: int, buffer: ctypes.Array
) -> Optional[str]:
    """None for a process that has exited or belongs to someone else. The
    layout is argc, the executable's path, padding, then argv and environ."""
    mib = (ctypes.c_int * 3)(_CTL_KERN, _KERN_PROCARGS2, pid)
    size = ctypes.c_size_t(ctypes.sizeof(buffer))
    if libc.sysctl(mib, 3, buffer, ctypes.byref(size), None, ctypes.c_size_t(0)):
        return None
    raw = buffer.raw[: size.value]
    argc = struct.unpack_from("=i", raw)[0]
    after_path = raw[struct.calcsize("=i"):].partition(b"\0")[2].lstrip(b"\0")
    return b" ".join(after_path.split(b"\0")[:argc]).decode("utf-8", "replace")
