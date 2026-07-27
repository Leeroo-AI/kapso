from __future__ import annotations

import os
import stat
from contextlib import ExitStack
from dataclasses import replace

import pytest

from kapso.core.config import load_config
from kapso.cross_run.launch import run_action_runtime_volume as volume_module
from kapso.cross_run.launch.run_action_docker_inspect import (
    observe_runtime_volume,
)
from kapso.cross_run.launch.run_action_supervisor_helper import (
    RunActionProcessStatObservation,
    RunActionSupervisorHelperError,
    read_run_action_process_stat_from_descriptor,
)
from kapso.cross_run.launch.run_action_runtime_volume import (
    DockerRunActionEmptyVolumeObservation,
    RunActionRuntimeVolumeError,
    _parse_mount_info_payload,
    _parse_size_option,
    _require_mount_authority,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
    issue_runtime_volume_authority,
)
from kapso.cross_run.settings import CrossRunSettings
from test_run_action_docker_inspect import _volume_raw
from test_run_action_docker_projection import _policy
from test_run_action_supervisor_contracts import _claim, _prepared_execution

_CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"
_KEEPER_CONTAINER_ID = "a" * 64
_PROCESS_SNAPSHOT_SIZE_BYTES = CrossRunSettings.from_dict(
    load_config(_CANONICAL_CONFIG_PATH)["cross_run"]
).launch.run_action_process_snapshot_size_bytes


@pytest.fixture(scope="module")
def runtime_volume_context():
    settings = CrossRunSettings.from_dict(
        load_config(_CANONICAL_CONFIG_PATH)["cross_run"]
    ).docker
    claim = _claim(policy=_policy(settings))
    authority = issue_runtime_volume_authority(claim, "a" * 32)
    volume = observe_runtime_volume(
        _volume_raw(authority, settings),
        claim,
        authority,
        settings,
    )
    return claim, authority, volume


def _mountinfo_payload(authority) -> bytes:
    return (
        "1232 1223 0:73 / /kapso/runtime-volume "
        "rw,nosuid,nodev,relatime master:595 - tmpfs tmpfs "
        f"rw,size={authority.size_limit_bytes},"
        f"nr_inodes={authority.inode_limit},"
        f"mode={authority.root_mode:o},"
        f"uid={authority.owner_user_id},gid={authority.owner_group_id},"
        "inode64,noswap\n"
    ).encode("ascii")


def _root_metadata(authority) -> os.stat_result:
    return os.stat_result(
        (
            stat.S_IFDIR | authority.root_mode,
            71,
            os.makedev(0, 73),
            2,
            authority.owner_user_id,
            authority.owner_group_id,
            40,
            0,
            0,
            0,
        )
    )


def _empty_observation(authority, volume):
    return DockerRunActionEmptyVolumeObservation(
        runtime_volume_authority=authority,
        docker_volume_observation=volume,
        keeper_container_id=_KEEPER_CONTAINER_ID,
        keeper_process_id=101,
        keeper_process_start_time_ticks=123456,
        process_cgroup_path=(
            f"/test.kapso.run_action.slice/docker-{_KEEPER_CONTAINER_ID}.scope"
        ),
        mount_id=1232,
        device=os.makedev(0, 73),
        root_inode=71,
        filesystem_type="tmpfs",
        observed_mount_flags=("nodev", "nosuid", "noswap"),
        owner_user_id=authority.owner_user_id,
        owner_group_id=authority.owner_group_id,
        root_mode=authority.root_mode,
        allocation_block_size_bytes=4096,
        effective_block_count=128,
        effective_size_bytes=524288,
        effective_inode_limit=64,
        used_block_count=0,
        used_size_bytes=0,
        used_inode_count=1,
        available_block_count=128,
        available_size_bytes=524288,
        available_inode_count=63,
        empty_entry_count=0,
        empty_size_bytes=0,
    )


def test_runtime_volume_mountinfo_parses_and_proves_exact_tmpfs_authority(
    runtime_volume_context,
):
    _, authority, _ = runtime_volume_context

    mount_info = _parse_mount_info_payload(
        _mountinfo_payload(authority),
        1232,
        RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
    )

    assert mount_info.mount_id == 1232
    assert mount_info.parent_mount_id == 1223
    assert mount_info.device_major == 0
    assert mount_info.device_minor == 73
    assert mount_info.mount_options == ("nodev", "nosuid", "relatime", "rw")
    assert mount_info.optional_fields == ("master:595",)
    assert mount_info.filesystem_type == "tmpfs"
    assert mount_info.source == "tmpfs"
    _require_mount_authority(mount_info, _root_metadata(authority), authority)


@pytest.mark.parametrize(
    ("payload_mutation", "mount_id", "destination"),
    (
        (lambda payload: payload[:-1], 1232, "/kapso/runtime-volume"),
        (lambda payload: payload + b"\x00", 1232, "/kapso/runtime-volume"),
        (
            lambda payload: payload.replace(b" - ", b" + ", 1),
            1232,
            "/kapso/runtime-volume",
        ),
        (
            lambda payload: payload.replace(b"0:73", b"invalid", 1),
            1232,
            "/kapso/runtime-volume",
        ),
        (
            lambda payload: payload.replace(
                b"1232 1223",
                b"1232 " + b"9" * 4301,
                1,
            ),
            1232,
            "/kapso/runtime-volume",
        ),
        (
            lambda payload: payload.replace(
                b"/ /kapso/runtime-volume",
                b"/subtree /kapso/runtime-volume",
                1,
            ),
            1232,
            "/kapso/runtime-volume",
        ),
        (lambda payload: payload, 1233, "/kapso/runtime-volume"),
        (lambda payload: payload, 1232, "/kapso/other"),
    ),
)
def test_runtime_volume_mountinfo_rejects_malformed_or_unissued_mounts(
    runtime_volume_context,
    payload_mutation,
    mount_id,
    destination,
):
    _, authority, _ = runtime_volume_context

    with pytest.raises(RunActionRuntimeVolumeError):
        _parse_mount_info_payload(
            payload_mutation(_mountinfo_payload(authority)),
            mount_id,
            destination,
        )


@pytest.mark.parametrize(
    "payload_mutation",
    (
        lambda payload: payload.replace(b"rw,nosuid", b"rw,exec,nosuid", 1),
        lambda payload: payload.replace(b" master:595", b" shared:595", 1),
        lambda payload: payload.replace(b"tmpfs tmpfs", b"ext4 tmpfs", 1),
        lambda payload: payload.replace(b"tmpfs tmpfs", b"tmpfs none", 1),
        lambda payload: payload.replace(b",noswap\n", b"\n", 1),
        lambda payload: payload.replace(b",noswap\n", b",exec,noswap\n", 1),
        lambda payload: payload.replace(b"size=", b"size=1,size=", 1),
        lambda payload: payload.replace(b"nr_inodes=", b"nr_inodes=1,", 1),
        lambda payload: payload.replace(b"mode=700", b"mode=755", 1),
        lambda payload: payload.replace(b"uid=1000", b"uid=1001", 1),
        lambda payload: payload.replace(b"gid=1000", b"gid=1001", 1),
    ),
)
def test_runtime_volume_mount_authority_rejects_every_policy_substitution(
    runtime_volume_context,
    payload_mutation,
):
    _, authority, _ = runtime_volume_context
    mount_info = _parse_mount_info_payload(
        payload_mutation(_mountinfo_payload(authority)),
        1232,
        RUN_ACTION_RUNTIME_VOLUME_KEEPER_DESTINATION,
    )

    with pytest.raises(
        RunActionRuntimeVolumeError,
        match="differs from issued tmpfs authority",
    ):
        _require_mount_authority(mount_info, _root_metadata(authority), authority)


@pytest.mark.parametrize(
    "value",
    (
        "0",
        "-1",
        "1kb",
        "1K",
        "1.5m",
        " 1m",
        "1m ",
        "1,m",
        "",
        "9" * 4301,
    ),
)
def test_runtime_volume_size_parser_rejects_ambiguous_values(value):
    with pytest.raises(
        RunActionRuntimeVolumeError,
        match="size option is malformed",
    ):
        _parse_size_option(value)


def test_runtime_volume_process_lease_parses_one_live_generation(tmp_path):
    process_id = 42
    fields = ("S", *(("0",) * 18), "123456")
    (tmp_path / "stat").write_bytes(
        f"{process_id} (keeper ) command) {' '.join(fields)}\n".encode("ascii")
    )
    process_descriptor = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with ExitStack() as descriptors:
        descriptors.callback(os.close, process_descriptor)
        assert (
            read_run_action_process_stat_from_descriptor(
                process_descriptor,
                process_id,
                _PROCESS_SNAPSHOT_SIZE_BYTES,
            ).start_time_ticks
            == 123456
        )

    (tmp_path / "stat").write_bytes(
        f"{process_id} (keeper) Z {' '.join(fields[1:])}\n".encode("ascii")
    )
    process_descriptor = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with ExitStack() as descriptors:
        descriptors.callback(os.close, process_descriptor)
        with pytest.raises(
            RunActionSupervisorHelperError,
            match="not one live process generation",
        ):
            read_run_action_process_stat_from_descriptor(
                process_descriptor,
                process_id,
                _PROCESS_SNAPSHOT_SIZE_BYTES,
            )


@pytest.mark.parametrize(
    ("rebound", "process_states"),
    (
        (False, ("S", "R")),
        (True, ("S", "S")),
    ),
)
def test_mounted_volume_lease_reopens_current_process_root(
    tmp_path,
    monkeypatch,
    rebound,
    process_states,
):
    keeper = _prepared_execution().volume_keeper_evidence
    process_path = tmp_path / "process"
    process_path.mkdir()
    process_root_path = tmp_path / "process-root"
    current_root_path = process_root_path / "kapso" / "runtime-volume"
    current_root_path.mkdir(parents=True)
    retained_root_path = current_root_path
    if rebound:
        retained_root_path = tmp_path / "retained"
        retained_root_path.mkdir()
    (process_path / "root").symlink_to(process_root_path)
    with ExitStack() as descriptors:
        process_descriptor = os.open(
            process_path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        descriptors.callback(os.close, process_descriptor)
        retained_root_descriptor = os.open(
            retained_root_path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        descriptors.callback(os.close, retained_root_descriptor)
        retained_metadata = os.fstat(retained_root_descriptor)
        retained_mount_id = volume_module.read_run_action_descriptor_mount_id(
            retained_root_descriptor,
            _PROCESS_SNAPSHOT_SIZE_BYTES,
        )
        lease = volume_module._MountedRuntimeVolumeLease(
            process_descriptor=process_descriptor,
            root_descriptor=retained_root_descriptor,
            keeper_container_id=keeper.container_id,
            keeper_process_id=keeper.process_id,
            process_start_time_ticks=keeper.process_start_time_ticks,
            process_cgroup_path=(keeper.mounted_helper_evidence.process_cgroup_path),
            root_mount_id=retained_mount_id,
            root_device=retained_metadata.st_dev,
            root_inode=retained_metadata.st_ino,
            process_snapshot_size_limit_bytes=_PROCESS_SNAPSHOT_SIZE_BYTES,
        )
        observed_process_states = iter(process_states)

        def observe_process_stat(_descriptor, _process_id, _byte_limit):
            return RunActionProcessStatObservation(
                process_id=keeper.process_id,
                state=next(observed_process_states),
                parent_process_id=0,
                start_time_ticks=keeper.process_start_time_ticks,
            )

        monkeypatch.setattr(
            volume_module,
            "read_run_action_process_stat_from_descriptor",
            observe_process_stat,
        )
        monkeypatch.setattr(
            volume_module,
            "read_run_action_process_cgroup_path_from_descriptor",
            lambda _descriptor, _container_id, _byte_limit: (
                keeper.mounted_helper_evidence.process_cgroup_path
            ),
        )
        mount_info = object()
        monkeypatch.setattr(
            volume_module,
            "_read_mount_info",
            lambda _descriptor, _mount_id, _destination, _byte_limit: mount_info,
        )
        monkeypatch.setattr(
            volume_module,
            "_require_mount_authority",
            lambda _mount_info, _metadata, _authority: None,
        )

        if rebound:
            with pytest.raises(
                RunActionRuntimeVolumeError,
                match="changed process or physical root",
            ):
                volume_module._require_same_mounted_runtime_volume(lease, keeper)
        else:
            volume_module._require_same_mounted_runtime_volume(lease, keeper)


def test_empty_volume_observation_closes_identity_and_capacity_accounting(
    runtime_volume_context,
):
    _, authority, volume = runtime_volume_context

    observation = _empty_observation(authority, volume)

    assert observation.runtime_volume_authority is authority
    assert observation.empty_entry_count == 0
    assert observation.used_size_bytes + observation.available_size_bytes == (
        observation.effective_size_bytes
    )
    assert observation.used_inode_count + observation.available_inode_count == (
        observation.effective_inode_limit
    )

    invalid_changes = (
        {"keeper_container_id": "short"},
        {"process_cgroup_path": "not-a-cgroup"},
        {"process_cgroup_path": (f"/\x00/docker-{_KEEPER_CONTAINER_ID}.scope")},
        {"mount_id": 0},
        {"filesystem_type": "ext4"},
        {"observed_mount_flags": ("nodev", "nosuid")},
        {"effective_size_bytes": 524289},
        {"effective_inode_limit": authority.inode_limit + 1},
        {"used_block_count": 1},
        {"available_block_count": 127},
        {"used_inode_count": 2},
        {"available_inode_count": 62},
        {"empty_entry_count": 1},
        {"empty_size_bytes": 1},
    )
    for changes in invalid_changes:
        with pytest.raises(
            RunActionRuntimeVolumeError,
            match="observation is incomplete or unsafe",
        ):
            replace(observation, **changes)

    unrelated_volume = replace(volume, volume_name="unrelated-volume")
    with pytest.raises(
        RunActionRuntimeVolumeError,
        match="observation is incomplete or unsafe",
    ):
        replace(
            observation,
            docker_volume_observation=unrelated_volume,
        )
