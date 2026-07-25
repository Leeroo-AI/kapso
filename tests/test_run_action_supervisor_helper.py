from __future__ import annotations

import struct

import pytest

from kapso.cross_run.launch import run_action_supervisor_helper as helper_module
from kapso.cross_run.launch.run_action_supervisor_helper import (
    RunActionSupervisorHelperError,
)

_CONTAINER_ID = "a" * 64


def test_static_elf_program_table_is_admitted():
    helper_module._require_static_elf(_minimal_elf64(1), "test executable")


@pytest.mark.parametrize("program_type", (2, 3))
def test_elf_dynamic_or_interpreter_program_header_is_rejected(program_type):
    payload = _minimal_elf64(program_type)

    with pytest.raises(RunActionSupervisorHelperError, match="dynamic"):
        helper_module._require_static_elf(payload, "test executable")


@pytest.mark.parametrize(
    "payload",
    (
        b"",
        b"not-elf",
        b"\x7fELF",
        b"\x7fELF\xff\xff\x01" + b"\x00" * 128,
    ),
)
def test_malformed_elf_fails_loud(payload):
    with pytest.raises(RunActionSupervisorHelperError):
        helper_module._require_static_elf(payload, "test executable")


def test_container_cgroup_payload_is_parsed_exactly():
    payload = f"0::/kapso.slice/docker-{_CONTAINER_ID}.scope\n".encode("ascii")

    assert (
        helper_module._parse_run_action_process_cgroup_path(
            payload,
            _CONTAINER_ID,
        )
        == f"/kapso.slice/docker-{_CONTAINER_ID}.scope"
    )


@pytest.mark.parametrize(
    "payload",
    (
        b"",
        f"1::/kapso.slice/docker-{_CONTAINER_ID}.scope\n".encode("ascii"),
        f"0::relative/docker-{_CONTAINER_ID}.scope\n".encode("ascii"),
        f"0::/kapso.slice/docker-{'b' * 64}.scope\n".encode("ascii"),
        (
            f"0::/kapso.slice/docker-{_CONTAINER_ID}.scope\n"
            f"0::/other.slice/docker-{_CONTAINER_ID}.scope\n"
        ).encode("ascii"),
        f"0::/kapso.slice/docker-{_CONTAINER_ID}.scope".encode("ascii"),
        f"0::/kapso.slice/../docker-{_CONTAINER_ID}.scope\n".encode("ascii"),
    ),
)
def test_container_cgroup_payload_rejects_ambiguous_identity(payload):
    with pytest.raises(RunActionSupervisorHelperError):
        helper_module._parse_run_action_process_cgroup_path(
            payload,
            _CONTAINER_ID,
        )


def _minimal_elf64(program_type):
    ident = b"\x7fELF" + bytes((2, 1, 1)) + b"\x00" * 9
    header_size = 64
    program_header_size = 56
    header = struct.pack(
        "<HHIQQQIHHHHHH",
        2,
        62,
        1,
        0,
        header_size,
        0,
        0,
        header_size,
        program_header_size,
        1,
        0,
        0,
        0,
    )
    program_header = struct.pack("<I", program_type) + b"\x00" * (
        program_header_size - 4
    )
    return ident + header + program_header
