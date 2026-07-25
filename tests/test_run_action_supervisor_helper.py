from __future__ import annotations

import os
import struct
from contextlib import ExitStack

import pytest

from kapso.cross_run.launch import run_action_supervisor_helper as helper_module
from kapso.cross_run.launch.run_action_supervisor_helper import (
    RunActionSupervisorHelperError,
)

_CONTAINER_ID = "a" * 64


def _process_stat_payload(
    process_id,
    *,
    command="keeper",
    state="S",
    parent_process_id="1",
    start_time_ticks="123456",
):
    fields = (
        state,
        parent_process_id,
        *(("0",) * 17),
        start_time_ticks,
    )
    return f"{process_id} ({command}) {' '.join(fields)}\n".encode("ascii")


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
        f"0::/kapso.slice/docker-{_CONTAINER_ID}.scope\r\n".encode("ascii"),
        f"0::/kapso.slice/docker-{_CONTAINER_ID}.scope\n\n".encode("ascii"),
        f"0::/kapso.slice/../docker-{_CONTAINER_ID}.scope\n".encode("ascii"),
    ),
)
def test_container_cgroup_payload_rejects_ambiguous_identity(payload):
    with pytest.raises(RunActionSupervisorHelperError):
        helper_module._parse_run_action_process_cgroup_path(
            payload,
            _CONTAINER_ID,
        )


def test_process_stat_parser_preserves_parent_and_generation_with_right_paren_in_comm():
    payload = _process_stat_payload(
        42,
        command="keeper ) command",
        state="S",
        parent_process_id="7",
        start_time_ticks="123456",
    )

    observation = helper_module._parse_run_action_process_stat(payload, 42)

    assert observation == helper_module.RunActionProcessStatObservation(
        process_id=42,
        state="S",
        parent_process_id=7,
        start_time_ticks=123456,
    )


@pytest.mark.parametrize("state", ("Z", "X", "Q", "SS"))
def test_process_stat_parser_rejects_non_live_or_malformed_state(state):
    with pytest.raises(
        RunActionSupervisorHelperError,
        match="not one live process generation",
    ):
        helper_module._parse_run_action_process_stat(
            _process_stat_payload(42, state=state),
            42,
        )


@pytest.mark.parametrize("parent_process_id", ("-1", "+1", "parent"))
def test_process_stat_parser_rejects_malformed_parent_process_id(
    parent_process_id,
):
    with pytest.raises(
        RunActionSupervisorHelperError,
        match="not one live process generation",
    ):
        helper_module._parse_run_action_process_stat(
            _process_stat_payload(
                42,
                parent_process_id=parent_process_id,
            ),
            42,
        )


@pytest.mark.parametrize("start_time_ticks", ("0", "-1", "+1", "ticks"))
def test_process_stat_parser_rejects_malformed_start_time(start_time_ticks):
    with pytest.raises(
        RunActionSupervisorHelperError,
        match="not one live process generation",
    ):
        helper_module._parse_run_action_process_stat(
            _process_stat_payload(
                42,
                start_time_ticks=start_time_ticks,
            ),
            42,
        )


@pytest.mark.parametrize(
    "payload",
    (
        _process_stat_payload(41),
        _process_stat_payload(42).removesuffix(b"\n"),
        _process_stat_payload(42).replace(b") S ", b")  S "),
        _process_stat_payload(42).replace(b") S ", b")\tS "),
        _process_stat_payload(42).replace(b"keeper", b"keep\ner"),
        _process_stat_payload(42).replace(b"keeper", b"keep\x00er"),
        b"42 (keeper) S 1\n",
    ),
)
def test_process_stat_parser_rejects_ambiguous_payload(payload):
    with pytest.raises(RunActionSupervisorHelperError):
        helper_module._parse_run_action_process_stat(payload, 42)


def test_process_command_line_parser_preserves_every_argument_byte():
    payload = b"/sbin/docker-init\x00--\x00hello world\x00\xff\x00\x00"

    assert helper_module._parse_run_action_process_command_line(payload) == (
        b"/sbin/docker-init",
        b"--",
        b"hello world",
        b"\xff",
        b"",
    )


def test_process_stat_and_command_line_readers_use_one_open_process_descriptor():
    process_id = os.getpid()
    with ExitStack() as descriptors:
        process_descriptor = os.open(
            f"/proc/{process_id}",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        descriptors.callback(os.close, process_descriptor)

        process_stat = helper_module.read_run_action_process_stat_from_descriptor(
            process_descriptor,
            process_id,
        )
        command_line = (
            helper_module.read_run_action_process_command_line_from_descriptor(
                process_descriptor,
            )
        )

    assert process_stat.process_id == process_id
    assert process_stat.parent_process_id == os.getppid()
    assert process_stat.start_time_ticks > 0
    assert command_line
    assert command_line[0]


@pytest.mark.parametrize(
    "payload",
    (
        b"",
        b"/sbin/docker-init",
        bytearray(b"/sbin/docker-init\x00"),
    ),
)
def test_process_command_line_parser_rejects_non_exact_payload(payload):
    with pytest.raises(
        RunActionSupervisorHelperError,
        match="not exact NUL-separated argv",
    ):
        helper_module._parse_run_action_process_command_line(payload)


def test_process_root_executable_and_namespace_metadata_are_descriptor_bound():
    process_id = os.getpid()
    with ExitStack() as descriptors:
        process_descriptor = os.open(
            f"/proc/{process_id}",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        descriptors.callback(os.close, process_descriptor)
        root_descriptor, root_metadata = (
            helper_module.open_run_action_process_root_descriptor(
                descriptors,
                process_descriptor,
            )
        )
        executable_descriptor, executable_metadata = (
            helper_module.open_run_action_process_executable_descriptor(
                descriptors,
                process_descriptor,
            )
        )
        mount_namespace_descriptor, mount_namespace_metadata = (
            helper_module.open_run_action_process_namespace_descriptor(
                descriptors,
                process_descriptor,
                "mnt",
            )
        )
        pid_namespace_descriptor, pid_namespace_metadata = (
            helper_module.open_run_action_process_namespace_descriptor(
                descriptors,
                process_descriptor,
                "pid",
            )
        )

        assert os.fstat(root_descriptor).st_ino == root_metadata.inode
        assert root_metadata.descriptor_name == "root"
        assert root_metadata.file_type == "directory"
        assert os.fstat(executable_descriptor).st_ino == executable_metadata.inode
        assert executable_metadata.descriptor_name == "exe"
        assert executable_metadata.file_type == "regular"
        assert (
            os.fstat(mount_namespace_descriptor).st_ino
            == mount_namespace_metadata.inode
        )
        assert mount_namespace_metadata.descriptor_name == "ns/mnt"
        assert mount_namespace_metadata.file_type == "regular"
        assert os.fstat(pid_namespace_descriptor).st_ino == pid_namespace_metadata.inode
        assert pid_namespace_metadata.descriptor_name == "ns/pid"
        assert pid_namespace_metadata.file_type == "regular"
        assert (
            mount_namespace_metadata.device,
            mount_namespace_metadata.inode,
        ) != (
            pid_namespace_metadata.device,
            pid_namespace_metadata.inode,
        )


@pytest.mark.parametrize("namespace_name", ("net", "../root", "", 1))
def test_process_namespace_open_rejects_unadmitted_name(namespace_name):
    process_id = os.getpid()
    with ExitStack() as descriptors:
        process_descriptor = os.open(
            f"/proc/{process_id}",
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        descriptors.callback(os.close, process_descriptor)

        with pytest.raises(
            RunActionSupervisorHelperError,
            match="namespace name is not admitted",
        ):
            helper_module.open_run_action_process_namespace_descriptor(
                descriptors,
                process_descriptor,
                namespace_name,
            )


def test_process_descriptor_metadata_rejects_wrong_file_type(tmp_path):
    descriptor = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with ExitStack() as descriptors:
        descriptors.callback(os.close, descriptor)
        with pytest.raises(
            RunActionSupervisorHelperError,
            match="wrong type",
        ):
            helper_module._observe_run_action_process_descriptor_metadata(
                descriptor,
                "exe",
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
