from __future__ import annotations

import os
import struct
from contextlib import ExitStack
from pathlib import Path

import pytest

from kapso.core.config import load_config
from kapso.cross_run.launch import run_action_supervisor_helper as helper_module
from kapso.cross_run.launch.run_action_supervisor_helper import (
    RunActionSupervisorHelperError,
)
from kapso.cross_run.settings import CrossRunSettings

_CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"
_CONTAINER_ID = "a" * 64
_PROCESS_SNAPSHOT_SIZE_BYTES = CrossRunSettings.from_dict(
    load_config(_CANONICAL_CONFIG_PATH)["cross_run"]
).launch.run_action_process_snapshot_size_bytes


@pytest.fixture(scope="module")
def docker_settings():
    return CrossRunSettings.from_dict(
        load_config(_CANONICAL_CONFIG_PATH)["cross_run"]
    ).docker


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


@pytest.mark.parametrize(
    "parent_process_id",
    (
        "-1",
        "+1",
        "parent",
        str((1 << 64)),
        "9" * 4301,
    ),
)
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


@pytest.mark.parametrize(
    "start_time_ticks",
    (
        "0",
        "-1",
        "+1",
        "ticks",
        str((1 << 64)),
        "9" * 4301,
    ),
)
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
            _PROCESS_SNAPSHOT_SIZE_BYTES,
        )
        command_line = (
            helper_module.read_run_action_process_command_line_from_descriptor(
                process_descriptor,
                _PROCESS_SNAPSHOT_SIZE_BYTES,
            )
        )

    assert process_stat.process_id == process_id
    assert process_stat.parent_process_id == os.getppid()
    assert process_stat.start_time_ticks > 0
    assert command_line
    assert command_line[0]


@pytest.mark.parametrize(
    ("file_name", "reader"),
    (
        (
            "stat",
            lambda descriptor, limit: (
                helper_module.read_run_action_process_stat_from_descriptor(
                    descriptor,
                    42,
                    limit,
                )
            ),
        ),
        (
            "cmdline",
            lambda descriptor, limit: (
                helper_module.read_run_action_process_command_line_from_descriptor(
                    descriptor,
                    limit,
                )
            ),
        ),
        (
            "cgroup",
            lambda descriptor, limit: (
                helper_module.read_run_action_process_cgroup_path_from_descriptor(
                    descriptor,
                    _CONTAINER_ID,
                    limit,
                )
            ),
        ),
    ),
)
def test_variable_process_readers_reject_one_byte_above_budget(
    tmp_path,
    file_name,
    reader,
):
    payloads = {
        "stat": _process_stat_payload(42),
        "cmdline": b"/kapso/helper\x00",
        "cgroup": f"0::/kapso.slice/docker-{_CONTAINER_ID}.scope\n".encode("ascii"),
    }
    payload = payloads[file_name]
    (tmp_path / file_name).write_bytes(payload)
    process_descriptor = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with ExitStack() as descriptors:
        descriptors.callback(os.close, process_descriptor)
        assert reader(process_descriptor, len(payload)) is not None
        with pytest.raises(
            RunActionSupervisorHelperError,
            match="complete-payload byte limit",
        ):
            reader(process_descriptor, len(payload) - 1)


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


def test_direct_child_reader_uses_retained_process_descriptor(tmp_path):
    process_id = 42
    process_directory = tmp_path / "retained-process"
    task_directory = process_directory / "task" / str(process_id)
    task_directory.mkdir(parents=True)
    (task_directory / "children").write_bytes(b"917 ")
    process_descriptor = os.open(
        process_directory,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with ExitStack() as descriptors:
        descriptors.callback(os.close, process_descriptor)

        assert (
            helper_module.read_run_action_process_direct_child_from_descriptor(
                process_descriptor,
                process_id,
                len(b"917 "),
            )
            == 917
        )


@pytest.mark.parametrize(
    "payload",
    (
        b"",
        b"0 ",
        b"17",
        b"17\n",
        b"17 18 ",
        b"17 17 ",
        b"+17 ",
        b"-17 ",
        b" 17 ",
        b"17  ",
        b"17 \n",
        b"17\x00 ",
        b"18446744073709551616 ",
        b"100000000000000000000 ",
    ),
)
def test_direct_child_parser_rejects_empty_malformed_multiple_or_duplicate_snapshot(
    payload,
):
    with pytest.raises(
        RunActionSupervisorHelperError,
        match="exactly one direct child",
    ):
        helper_module._parse_run_action_direct_child(payload)


def test_direct_child_reader_refuses_symlinked_task_component(tmp_path):
    process_directory = tmp_path / "retained-process"
    alternate_task_directory = tmp_path / "alternate-task"
    (alternate_task_directory / "42").mkdir(parents=True)
    (alternate_task_directory / "42" / "children").write_bytes(b"917 ")
    process_directory.mkdir()
    (process_directory / "task").symlink_to(alternate_task_directory)
    process_descriptor = os.open(
        process_directory,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with ExitStack() as descriptors:
        descriptors.callback(os.close, process_descriptor)
        with pytest.raises(OSError):
            helper_module.read_run_action_process_direct_child_from_descriptor(
                process_descriptor,
                42,
                len(b"917 "),
            )


def test_direct_child_reader_fails_when_full_snapshot_exceeds_byte_limit(tmp_path):
    payload = b"917 "
    process_id = 42
    process_directory = tmp_path / "retained-process"
    task_directory = process_directory / "task" / str(process_id)
    task_directory.mkdir(parents=True)
    (task_directory / "children").write_bytes(payload)
    process_descriptor = os.open(
        process_directory,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with ExitStack() as descriptors:
        descriptors.callback(os.close, process_descriptor)
        with pytest.raises(
            RunActionSupervisorHelperError,
            match="complete-payload byte limit",
        ):
            helper_module.read_run_action_process_direct_child_from_descriptor(
                process_descriptor,
                process_id,
                len(payload) - 1,
            )


@pytest.mark.parametrize("byte_limit", (0, -1, True, "4"))
def test_direct_child_reader_requires_positive_integer_byte_limit(
    tmp_path,
    byte_limit,
):
    process_directory = tmp_path / "retained-process"
    process_directory.mkdir()
    process_descriptor = os.open(
        process_directory,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with ExitStack() as descriptors:
        descriptors.callback(os.close, process_descriptor)
        with pytest.raises(
            RunActionSupervisorHelperError,
            match="direct-child identity is malformed",
        ):
            helper_module.read_run_action_process_direct_child_from_descriptor(
                process_descriptor,
                42,
                byte_limit,
            )


def test_mountinfo_reader_returns_exact_full_eof_bytes(tmp_path):
    payload = b"41 28 0:37 / /workspace rw - tmpfs tmpfs rw\n"
    process_directory = tmp_path / "retained-process"
    process_directory.mkdir()
    (process_directory / "mountinfo").write_bytes(payload)
    process_descriptor = os.open(
        process_directory,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with ExitStack() as descriptors:
        descriptors.callback(os.close, process_descriptor)

        assert (
            helper_module.read_run_action_process_mount_info_from_descriptor(
                process_descriptor,
                len(payload),
            )
            == payload
        )


def test_descriptor_mount_id_reader_honors_exact_fdinfo_byte_bound(tmp_path):
    file_path = tmp_path / "observed"
    file_path.write_bytes(b"fdinfo")
    descriptor = os.open(
        file_path,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with ExitStack() as descriptors:
        descriptors.callback(os.close, descriptor)
        payload = Path(f"/proc/{os.getpid()}/fdinfo/{descriptor}").read_bytes()

        assert (
            helper_module.read_run_action_descriptor_mount_id(
                descriptor,
                len(payload),
            )
            > 0
        )
        with pytest.raises(
            RunActionSupervisorHelperError,
            match="complete-payload byte limit",
        ):
            helper_module.read_run_action_descriptor_mount_id(
                descriptor,
                len(payload) - 1,
            )


def test_mountinfo_reader_fails_when_full_payload_exceeds_byte_limit(tmp_path):
    payload = b"41 28 0:37 / /workspace rw - tmpfs tmpfs rw\n"
    process_directory = tmp_path / "retained-process"
    process_directory.mkdir()
    (process_directory / "mountinfo").write_bytes(payload)
    process_descriptor = os.open(
        process_directory,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with ExitStack() as descriptors:
        descriptors.callback(os.close, process_descriptor)
        with pytest.raises(
            RunActionSupervisorHelperError,
            match="complete-payload byte limit",
        ):
            helper_module.read_run_action_process_mount_info_from_descriptor(
                process_descriptor,
                len(payload) - 1,
            )


@pytest.mark.parametrize("byte_limit", (0, -1, True, "4096"))
def test_mountinfo_reader_requires_positive_integer_byte_limit(
    tmp_path,
    byte_limit,
):
    process_directory = tmp_path / "retained-process"
    process_directory.mkdir()
    process_descriptor = os.open(
        process_directory,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with ExitStack() as descriptors:
        descriptors.callback(os.close, process_descriptor)
        with pytest.raises(
            RunActionSupervisorHelperError,
            match="mountinfo read authority is malformed",
        ):
            helper_module.read_run_action_process_mount_info_from_descriptor(
                process_descriptor,
                byte_limit,
            )


@pytest.mark.parametrize(
    "payload",
    (
        b"",
        b"41 28 0:37 / /workspace rw - tmpfs tmpfs rw",
        b"41 28 0:37 / /workspace rw - tmpfs tmpfs rw\x00\n",
    ),
)
def test_mountinfo_reader_rejects_payload_unsuitable_for_exact_snapshot(
    tmp_path,
    payload,
):
    process_directory = tmp_path / "retained-process"
    process_directory.mkdir()
    (process_directory / "mountinfo").write_bytes(payload)
    process_descriptor = os.open(
        process_directory,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with ExitStack() as descriptors:
        descriptors.callback(os.close, process_descriptor)
        with pytest.raises(
            RunActionSupervisorHelperError,
            match="full-EOF bytes",
        ):
            helper_module.read_run_action_process_mount_info_from_descriptor(
                process_descriptor,
                max(len(payload), 1),
            )


def test_host_boot_id_reader_uses_fixed_no_follow_components(tmp_path):
    boot_id = "01234567-89ab-4def-8123-456789abcdef"
    boot_id_directory = tmp_path / "sys" / "kernel" / "random"
    boot_id_directory.mkdir(parents=True)
    (boot_id_directory / "boot_id").write_text(f"{boot_id}\n", encoding="ascii")
    proc_root_descriptor = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with ExitStack() as descriptors:
        descriptors.callback(os.close, proc_root_descriptor)

        assert (
            helper_module.read_run_action_host_boot_id(proc_root_descriptor) == boot_id
        )


@pytest.mark.parametrize(
    "payload",
    (
        b"",
        b"01234567-89ab-4def-8123-456789abcdef",
        b"01234567-89AB-4DEF-8123-456789ABCDEF\n",
        b"not-a-boot-id\n",
    ),
)
def test_host_boot_id_reader_rejects_noncanonical_payload(tmp_path, payload):
    boot_id_directory = tmp_path / "sys" / "kernel" / "random"
    boot_id_directory.mkdir(parents=True)
    (boot_id_directory / "boot_id").write_bytes(payload)
    proc_root_descriptor = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with ExitStack() as descriptors:
        descriptors.callback(os.close, proc_root_descriptor)
        with pytest.raises(
            RunActionSupervisorHelperError,
            match="boot ID payload is malformed",
        ):
            helper_module.read_run_action_host_boot_id(proc_root_descriptor)


def test_host_boot_id_reader_rejects_structural_overflow(tmp_path):
    boot_id_directory = tmp_path / "sys" / "kernel" / "random"
    boot_id_directory.mkdir(parents=True)
    (boot_id_directory / "boot_id").write_bytes(
        b"01234567-89ab-4def-8123-456789abcdef\nextra\n"
    )
    proc_root_descriptor = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with ExitStack() as descriptors:
        descriptors.callback(os.close, proc_root_descriptor)
        with pytest.raises(
            RunActionSupervisorHelperError,
            match="complete-payload byte limit",
        ):
            helper_module.read_run_action_host_boot_id(proc_root_descriptor)


def test_host_boot_id_reader_refuses_symlinked_path_component(tmp_path):
    alternate_sys_directory = tmp_path / "alternate-sys"
    boot_id_directory = alternate_sys_directory / "kernel" / "random"
    boot_id_directory.mkdir(parents=True)
    (boot_id_directory / "boot_id").write_text(
        "01234567-89ab-4def-8123-456789abcdef\n",
        encoding="ascii",
    )
    (tmp_path / "sys").symlink_to(alternate_sys_directory)
    proc_root_descriptor = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with ExitStack() as descriptors:
        descriptors.callback(os.close, proc_root_descriptor)
        with pytest.raises(OSError):
            helper_module.read_run_action_host_boot_id(proc_root_descriptor)


def test_executable_descriptor_verification_is_stable_and_nonclosing(
    docker_settings,
):
    executable_path = docker_settings.init_executable_path
    expected_digest = docker_settings.init_executable_digest
    descriptor = os.open(
        executable_path,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with ExitStack() as descriptors:
        descriptors.callback(os.close, descriptor)
        os.lseek(descriptor, 11, os.SEEK_SET)

        observation = helper_module.verify_run_action_executable_descriptor(
            descriptor,
            expected_digest,
            _PROCESS_SNAPSHOT_SIZE_BYTES,
        )

        assert observation.executable_digest == expected_digest
        assert observation.device == os.fstat(descriptor).st_dev
        assert observation.inode == os.fstat(descriptor).st_ino
        assert os.lseek(descriptor, 0, os.SEEK_CUR) == 11


def test_executable_descriptor_remains_open_after_digest_failure(docker_settings):
    descriptor = os.open(
        docker_settings.init_executable_path,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with ExitStack() as descriptors:
        descriptors.callback(os.close, descriptor)
        with pytest.raises(
            RunActionSupervisorHelperError,
            match="changed while proving",
        ):
            helper_module.verify_run_action_executable_descriptor(
                descriptor,
                "sha256:" + "0" * 64,
                _PROCESS_SNAPSHOT_SIZE_BYTES,
            )

        assert os.fstat(descriptor).st_ino > 0


def test_executable_descriptor_mount_identity_race_fails_without_closing(
    monkeypatch,
    docker_settings,
):
    executable_path = docker_settings.init_executable_path
    expected_digest = docker_settings.init_executable_digest
    descriptor = os.open(
        executable_path,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    mount_ids = iter((41, 42))
    monkeypatch.setattr(
        helper_module,
        "read_run_action_descriptor_mount_id",
        lambda _descriptor, _byte_limit: next(mount_ids),
    )
    with ExitStack() as descriptors:
        descriptors.callback(os.close, descriptor)
        with pytest.raises(
            RunActionSupervisorHelperError,
            match="changed while proving",
        ):
            helper_module.verify_run_action_executable_descriptor(
                descriptor,
                expected_digest,
                _PROCESS_SNAPSHOT_SIZE_BYTES,
            )

        assert os.fstat(descriptor).st_ino > 0


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
                _PROCESS_SNAPSHOT_SIZE_BYTES,
            )
        )
        executable_descriptor, executable_metadata = (
            helper_module.open_run_action_process_executable_descriptor(
                descriptors,
                process_descriptor,
                _PROCESS_SNAPSHOT_SIZE_BYTES,
            )
        )
        mount_namespace_descriptor, mount_namespace_metadata = (
            helper_module.open_run_action_process_namespace_descriptor(
                descriptors,
                process_descriptor,
                "mnt",
                _PROCESS_SNAPSHOT_SIZE_BYTES,
            )
        )
        pid_namespace_descriptor, pid_namespace_metadata = (
            helper_module.open_run_action_process_namespace_descriptor(
                descriptors,
                process_descriptor,
                "pid",
                _PROCESS_SNAPSHOT_SIZE_BYTES,
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
                _PROCESS_SNAPSHOT_SIZE_BYTES,
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
                _PROCESS_SNAPSHOT_SIZE_BYTES,
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
