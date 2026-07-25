from __future__ import annotations

import struct

import pytest

from kapso.cross_run.launch import run_action_keeper_helper as helper_module
from kapso.cross_run.launch.run_action_keeper_helper import (
    RunActionKeeperHelperError,
)


def test_static_elf_program_table_is_admitted():
    helper_module._require_static_elf(_minimal_elf64(1))


@pytest.mark.parametrize("program_type", (2, 3))
def test_elf_dynamic_or_interpreter_program_header_is_rejected(program_type):
    payload = _minimal_elf64(program_type)

    with pytest.raises(RunActionKeeperHelperError, match="dynamic"):
        helper_module._require_static_elf(payload)


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
    with pytest.raises(RunActionKeeperHelperError):
        helper_module._require_static_elf(payload)


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
