"""Crash-atomic descriptor-only delivery of committed run-action payloads."""

from __future__ import annotations

import errno
import os
import stat
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import pytest

import kapso.cross_run.launch.run_action_activation_delivery as activation_delivery
import kapso.cross_run.launch.run_action_atomic_publication as atomic_publication
from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.launch.run_action_activation_delivery import (
    RunActionActivationDeliveryError,
    publish_or_adopt_run_action_delivery as _publish_or_adopt_run_action_delivery,
)
from kapso.cross_run.launch.run_action_atomic_publication import (
    RunActionAtomicPublicationError,
)
from kapso.cross_run.launch.run_action_supervisor_helper import (
    read_run_action_descriptor_mount_id,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionPreparedDeliverySlot,
    RunActionPreparedFileKind,
)
from test_run_action_supervisor_contracts import (
    _RUN_ACTION_PROCESS_SNAPSHOT_SIZE_BYTES,
)

_INPUT_PAYLOAD = b'{"action":"research"}'
_CREDENTIAL_PAYLOAD = b"provider-token"


def publish_or_adopt_run_action_delivery(
    slot: RunActionPreparedDeliverySlot,
    slot_directory_descriptor: int,
    payload: bytes,
):
    return _publish_or_adopt_run_action_delivery(
        slot,
        slot_directory_descriptor,
        payload,
        _RUN_ACTION_PROCESS_SNAPSHOT_SIZE_BYTES,
    )


@dataclass(frozen=True)
class _PreparedSlotFixture:
    root_descriptor: int
    slot_descriptor: int
    slot: RunActionPreparedDeliverySlot


def _open_prepared_slot(root_path: Path) -> _PreparedSlotFixture:
    root_descriptor = os.open(
        root_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
    )
    os.mkdir(
        "slot",
        mode=0o700,
        dir_fd=root_descriptor,
    )
    slot_descriptor = os.open(
        "slot",
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=root_descriptor,
    )
    os.fchmod(slot_descriptor, 0o700)
    return _PreparedSlotFixture(
        root_descriptor=root_descriptor,
        slot_descriptor=slot_descriptor,
        slot=_mint_slot(
            slot_descriptor,
            RunActionPreparedFileKind.INPUT,
        ),
    )


@pytest.fixture
def prepared_slot(tmp_path):
    fixture = _open_prepared_slot(tmp_path)
    yield fixture
    os.close(fixture.slot_descriptor)
    os.close(fixture.root_descriptor)


def _mint_slot(
    slot_descriptor: int,
    kind: RunActionPreparedFileKind,
) -> RunActionPreparedDeliverySlot:
    metadata = os.fstat(slot_descriptor)
    paths = {
        RunActionPreparedFileKind.INPUT: ("input", "request.blob"),
        RunActionPreparedFileKind.CREDENTIAL: ("credential", "credentials"),
    }
    return RunActionPreparedDeliverySlot.mint(
        preparation_claim_id=content_id(
            "run-action-preparation-claim",
            {"fixture": "claim"},
        ),
        runtime_volume_authority_id=content_id(
            "run-action-runtime-volume-authority",
            {"fixture": "volume"},
        ),
        generation_nonce="1" * 32,
        kind=kind,
        directory_relative_path=paths[kind][0],
        final_file_name=paths[kind][1],
        directory_type="directory",
        owner_user_id=metadata.st_uid,
        owner_group_id=metadata.st_gid,
        mode=0o700,
        observed_entry_count=0,
        payload_size_limit_bytes=128,
        mount_id=read_run_action_descriptor_mount_id(
            slot_descriptor,
            _RUN_ACTION_PROCESS_SNAPSHOT_SIZE_BYTES,
        ),
        device=metadata.st_dev,
        inode=metadata.st_ino,
    )


def _remint_slot(
    slot: RunActionPreparedDeliverySlot,
    **changes,
) -> RunActionPreparedDeliverySlot:
    values = {
        key: value
        for key, value in slot.to_dict().items()
        if key != slot.IDENTITY_FIELD
    }
    values.update(changes)
    return RunActionPreparedDeliverySlot.mint(**values)


def _create_named_payload(
    directory_descriptor: int,
    name: str,
    payload: bytes,
    *,
    mode: int = 0o400,
) -> os.stat_result:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
        dir_fd=directory_descriptor,
    )
    with os.fdopen(descriptor, "wb", buffering=0) as handle:
        written_size = 0
        while written_size < len(payload):
            written_size += os.write(
                handle.fileno(),
                payload[written_size:],
            )
        os.fchmod(handle.fileno(), mode)
        os.fsync(handle.fileno())
        metadata = os.fstat(handle.fileno())
    return metadata


def test_publish_and_exact_retry_adoption_return_the_same_physical_file(
    prepared_slot,
):
    with publish_or_adopt_run_action_delivery(
        prepared_slot.slot,
        prepared_slot.slot_descriptor,
        _INPUT_PAYLOAD,
    ) as published:
        observation = published.observation
        assert published.require_final_path(_INPUT_PAYLOAD) == observation

    assert tuple(os.listdir(prepared_slot.slot_descriptor)) == ("request.blob",)
    assert observation.prepared_delivery_slot_id == (
        prepared_slot.slot.prepared_delivery_slot_id
    )
    assert observation.relative_path == "input/request.blob"
    assert observation.mode == 0o400
    assert observation.link_count == 1
    assert observation.size_bytes == len(_INPUT_PAYLOAD)
    assert observation.content_digest == tree_or_blob_digest(_INPUT_PAYLOAD)

    with publish_or_adopt_run_action_delivery(
        prepared_slot.slot,
        prepared_slot.slot_descriptor,
        _INPUT_PAYLOAD,
    ) as adopted:
        assert adopted.observation == observation
        assert adopted.require_final_path(_INPUT_PAYLOAD) == observation


def test_credential_observation_never_contains_a_digest(prepared_slot):
    credential_slot = _mint_slot(
        prepared_slot.slot_descriptor,
        RunActionPreparedFileKind.CREDENTIAL,
    )

    with publish_or_adopt_run_action_delivery(
        credential_slot,
        prepared_slot.slot_descriptor,
        _CREDENTIAL_PAYLOAD,
    ) as delivered:
        observation = delivered.observation
        assert delivered.require_final_path(_CREDENTIAL_PAYLOAD) == observation

    assert observation.relative_path == "credential/credentials"
    assert observation.content_digest is None


def test_late_candidate_validation_rejects_same_size_path_inode_substitution(
    prepared_slot,
):
    with publish_or_adopt_run_action_delivery(
        prepared_slot.slot,
        prepared_slot.slot_descriptor,
        _INPUT_PAYLOAD,
    ) as delivered:
        retained_observation = delivered.observation
        replacement_metadata = _create_named_payload(
            prepared_slot.root_descriptor,
            "replacement",
            _INPUT_PAYLOAD,
        )
        os.rename(
            "replacement",
            prepared_slot.slot.final_file_name,
            src_dir_fd=prepared_slot.root_descriptor,
            dst_dir_fd=prepared_slot.slot_descriptor,
        )

        assert replacement_metadata.st_size == retained_observation.size_bytes
        assert replacement_metadata.st_ino != retained_observation.inode
        with pytest.raises(
            RunActionActivationDeliveryError,
            match="retained delivered file descriptor changed",
        ):
            delivered.require_final_path(_INPUT_PAYLOAD)


def test_adoption_rejects_wrong_payload(prepared_slot):
    substituted_payload = b"x" * len(_INPUT_PAYLOAD)
    _create_named_payload(
        prepared_slot.slot_descriptor,
        prepared_slot.slot.final_file_name,
        substituted_payload,
    )

    with pytest.raises(
        RunActionAtomicPublicationError,
        match="bytes differ",
    ):
        publish_or_adopt_run_action_delivery(
            prepared_slot.slot,
            prepared_slot.slot_descriptor,
            _INPUT_PAYLOAD,
        )


def test_adoption_rejects_wrong_mode(prepared_slot):
    _create_named_payload(
        prepared_slot.slot_descriptor,
        prepared_slot.slot.final_file_name,
        _INPUT_PAYLOAD,
        mode=0o600,
    )

    with pytest.raises(
        RunActionActivationDeliveryError,
        match="unsafe or substituted",
    ):
        publish_or_adopt_run_action_delivery(
            prepared_slot.slot,
            prepared_slot.slot_descriptor,
            _INPUT_PAYLOAD,
        )


def test_adoption_rejects_a_second_hard_link(prepared_slot):
    _create_named_payload(
        prepared_slot.slot_descriptor,
        prepared_slot.slot.final_file_name,
        _INPUT_PAYLOAD,
    )
    os.link(
        prepared_slot.slot.final_file_name,
        "outside-hardlink",
        src_dir_fd=prepared_slot.slot_descriptor,
        dst_dir_fd=prepared_slot.root_descriptor,
        follow_symlinks=False,
    )

    with pytest.raises(
        RunActionActivationDeliveryError,
        match="unsafe or substituted",
    ):
        publish_or_adopt_run_action_delivery(
            prepared_slot.slot,
            prepared_slot.slot_descriptor,
            _INPUT_PAYLOAD,
        )


def test_adoption_rejects_inode_replacement_during_reopen(
    prepared_slot,
    monkeypatch,
):
    _create_named_payload(
        prepared_slot.slot_descriptor,
        prepared_slot.slot.final_file_name,
        _INPUT_PAYLOAD,
    )
    _create_named_payload(
        prepared_slot.root_descriptor,
        "replacement",
        _INPUT_PAYLOAD,
    )
    original_open = activation_delivery.os.open
    final_open_count = 0

    def substituting_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal final_open_count
        if (
            path == prepared_slot.slot.final_file_name
            and dir_fd == prepared_slot.slot_descriptor
        ):
            final_open_count += 1
            if final_open_count == 2:
                os.rename(
                    "replacement",
                    prepared_slot.slot.final_file_name,
                    src_dir_fd=prepared_slot.root_descriptor,
                    dst_dir_fd=prepared_slot.slot_descriptor,
                )
        if dir_fd is None:
            return original_open(path, flags, mode)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(activation_delivery.os, "open", substituting_open)

    with pytest.raises(
        RunActionActivationDeliveryError,
        match="physical state changed",
    ):
        publish_or_adopt_run_action_delivery(
            prepared_slot.slot,
            prepared_slot.slot_descriptor,
            _INPUT_PAYLOAD,
        )


def test_adoption_rejects_same_inode_mutate_and_restore_through_held_writer(
    prepared_slot,
    monkeypatch,
):
    held_writer = os.open(
        prepared_slot.slot.final_file_name,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
        dir_fd=prepared_slot.slot_descriptor,
    )
    with ExitStack() as resources:
        resources.callback(os.close, held_writer)
        written_size = 0
        while written_size < len(_INPUT_PAYLOAD):
            written_size += os.write(
                held_writer,
                _INPUT_PAYLOAD[written_size:],
            )
        os.fchmod(held_writer, 0o400)
        os.fsync(held_writer)
        original_metadata = os.fstat(held_writer)
        original_fsync = activation_delivery.os.fsync
        mutation_performed = False

        def mutate_and_restore_before_fsync(descriptor):
            nonlocal mutation_performed
            if descriptor != prepared_slot.slot_descriptor and not mutation_performed:
                os.pwrite(
                    held_writer,
                    b"x" * len(_INPUT_PAYLOAD),
                    0,
                )
                os.pwrite(
                    held_writer,
                    _INPUT_PAYLOAD,
                    0,
                )
                os.utime(
                    held_writer,
                    ns=(
                        original_metadata.st_atime_ns,
                        original_metadata.st_mtime_ns + 1_000_000_000,
                    ),
                )
                original_fsync(held_writer)
                mutation_performed = True
            original_fsync(descriptor)

        monkeypatch.setattr(
            activation_delivery.os,
            "fsync",
            mutate_and_restore_before_fsync,
        )

        with pytest.raises(
            RunActionActivationDeliveryError,
            match="physical state changed",
        ):
            publish_or_adopt_run_action_delivery(
                prepared_slot.slot,
                prepared_slot.slot_descriptor,
                _INPUT_PAYLOAD,
            )

        final_metadata = os.stat(
            prepared_slot.slot.final_file_name,
            dir_fd=prepared_slot.slot_descriptor,
            follow_symlinks=False,
        )
        assert mutation_performed is True
        assert final_metadata.st_ino == original_metadata.st_ino


def test_publication_rejects_wrong_slot_inode_or_parent(
    prepared_slot,
):
    wrong_inode = _remint_slot(
        prepared_slot.slot,
        inode=prepared_slot.slot.inode + 1,
    )
    with pytest.raises(
        RunActionActivationDeliveryError,
        match="slot directory",
    ):
        publish_or_adopt_run_action_delivery(
            wrong_inode,
            prepared_slot.slot_descriptor,
            _INPUT_PAYLOAD,
        )

    os.mkdir(
        "other",
        mode=0o700,
        dir_fd=prepared_slot.root_descriptor,
    )
    other_descriptor = os.open(
        "other",
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=prepared_slot.root_descriptor,
    )
    with ExitStack() as descriptors:
        descriptors.callback(os.close, other_descriptor)
        with pytest.raises(
            RunActionActivationDeliveryError,
            match="slot directory",
        ):
            publish_or_adopt_run_action_delivery(
                prepared_slot.slot,
                other_descriptor,
                _INPUT_PAYLOAD,
            )


def test_publication_rejects_any_extra_entry(prepared_slot):
    _create_named_payload(
        prepared_slot.slot_descriptor,
        "unexpected",
        b"x",
    )

    with pytest.raises(
        RunActionActivationDeliveryError,
        match="unexpected entry",
    ):
        publish_or_adopt_run_action_delivery(
            prepared_slot.slot,
            prepared_slot.slot_descriptor,
            _INPUT_PAYLOAD,
        )


@pytest.mark.parametrize("payload", (b"", b"x" * 129))
def test_publication_rejects_empty_or_oversized_payload(
    prepared_slot,
    payload,
):
    with pytest.raises(
        RunActionActivationDeliveryError,
        match="bounded nonempty",
    ):
        publish_or_adopt_run_action_delivery(
            prepared_slot.slot,
            prepared_slot.slot_descriptor,
            payload,
        )

    assert tuple(os.listdir(prepared_slot.slot_descriptor)) == ()


def test_short_writes_make_full_progress(prepared_slot, monkeypatch):
    original_write = activation_delivery.os.write

    def short_write(descriptor, payload):
        return original_write(descriptor, payload[:2])

    monkeypatch.setattr(activation_delivery.os, "write", short_write)

    with publish_or_adopt_run_action_delivery(
        prepared_slot.slot,
        prepared_slot.slot_descriptor,
        _INPUT_PAYLOAD,
    ) as delivered:
        observation = delivered.observation

    assert observation.size_bytes == len(_INPUT_PAYLOAD)


def test_no_progress_write_fails_and_anonymous_inode_disappears(
    prepared_slot,
    monkeypatch,
):
    def no_progress_write(descriptor, payload):
        return 0

    monkeypatch.setattr(activation_delivery.os, "write", no_progress_write)

    with pytest.raises(
        RunActionAtomicPublicationError,
        match="no valid progress",
    ):
        publish_or_adopt_run_action_delivery(
            prepared_slot.slot,
            prepared_slot.slot_descriptor,
            _INPUT_PAYLOAD,
        )

    assert tuple(os.listdir(prepared_slot.slot_descriptor)) == ()


def test_link_failure_leaves_no_named_or_anonymous_delivery(
    prepared_slot,
    monkeypatch,
):
    def fail_before_link(
        anonymous_descriptor,
        slot_directory_descriptor,
        final_file_name,
    ):
        raise OSError(errno.EIO, os.strerror(errno.EIO), final_file_name)

    monkeypatch.setattr(
        activation_delivery,
        "link_run_action_anonymous_file_no_replace",
        fail_before_link,
    )

    with pytest.raises(OSError) as raised:
        publish_or_adopt_run_action_delivery(
            prepared_slot.slot,
            prepared_slot.slot_descriptor,
            _INPUT_PAYLOAD,
        )

    assert raised.value.errno == errno.EIO
    assert tuple(os.listdir(prepared_slot.slot_descriptor)) == ()


def test_eexist_collision_is_not_replaced_or_adopted_in_the_same_call(
    prepared_slot,
    monkeypatch,
):
    original_link = activation_delivery.link_run_action_anonymous_file_no_replace

    def collide_then_link(
        anonymous_descriptor,
        slot_directory_descriptor,
        final_file_name,
    ):
        _create_named_payload(
            slot_directory_descriptor,
            final_file_name,
            b"collision",
        )
        original_link(
            anonymous_descriptor,
            slot_directory_descriptor,
            final_file_name,
        )

    monkeypatch.setattr(
        activation_delivery,
        "link_run_action_anonymous_file_no_replace",
        collide_then_link,
    )

    with pytest.raises(OSError) as raised:
        publish_or_adopt_run_action_delivery(
            prepared_slot.slot,
            prepared_slot.slot_descriptor,
            _INPUT_PAYLOAD,
        )

    assert raised.value.errno == errno.EEXIST
    descriptor = os.open(
        prepared_slot.slot.final_file_name,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=prepared_slot.slot_descriptor,
    )
    with os.fdopen(descriptor, "rb") as handle:
        assert handle.read() == b"collision"


def test_concurrent_publishers_have_one_atomic_winner(
    prepared_slot,
    monkeypatch,
):
    second_slot_descriptor = os.open(
        ".",
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
        dir_fd=prepared_slot.slot_descriptor,
    )
    barrier = threading.Barrier(2)
    original_link = activation_delivery.link_run_action_anonymous_file_no_replace

    def synchronized_link(
        anonymous_descriptor,
        slot_directory_descriptor,
        final_file_name,
    ):
        barrier.wait()
        original_link(
            anonymous_descriptor,
            slot_directory_descriptor,
            final_file_name,
        )

    monkeypatch.setattr(
        activation_delivery,
        "link_run_action_anonymous_file_no_replace",
        synchronized_link,
    )
    with ExitStack() as resources:
        resources.callback(os.close, second_slot_descriptor)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = (
                executor.submit(
                    publish_or_adopt_run_action_delivery,
                    prepared_slot.slot,
                    descriptor,
                    _INPUT_PAYLOAD,
                )
                for descriptor in (
                    prepared_slot.slot_descriptor,
                    second_slot_descriptor,
                )
            )
            completed = tuple(futures)
            exceptions = tuple(future.exception() for future in completed)
            delivered_leases = tuple(
                future.result() for future in completed if future.exception() is None
            )
        winning_observation = delivered_leases[0].require_final_path(_INPUT_PAYLOAD)
        delivered_leases[0].close()

    errors = tuple(error for error in exceptions if error is not None)
    assert len(delivered_leases) == 1
    assert len(errors) == 1
    assert type(errors[0]) is FileExistsError
    assert errors[0].errno == errno.EEXIST
    with publish_or_adopt_run_action_delivery(
        prepared_slot.slot,
        prepared_slot.slot_descriptor,
        _INPUT_PAYLOAD,
    ) as adopted:
        assert adopted.observation == winning_observation


def test_directory_fsync_failure_retains_linked_final_for_retry_adoption(
    prepared_slot,
    monkeypatch,
):
    original_fsync = activation_delivery.os.fsync

    def fail_directory_fsync(descriptor):
        if descriptor == prepared_slot.slot_descriptor:
            raise OSError(errno.EIO, os.strerror(errno.EIO))
        original_fsync(descriptor)

    monkeypatch.setattr(
        activation_delivery.os,
        "fsync",
        fail_directory_fsync,
    )

    with pytest.raises(OSError) as raised:
        publish_or_adopt_run_action_delivery(
            prepared_slot.slot,
            prepared_slot.slot_descriptor,
            _INPUT_PAYLOAD,
        )

    assert raised.value.errno == errno.EIO
    assert tuple(os.listdir(prepared_slot.slot_descriptor)) == ("request.blob",)
    with pytest.raises(OSError) as repeated:
        publish_or_adopt_run_action_delivery(
            prepared_slot.slot,
            prepared_slot.slot_descriptor,
            _INPUT_PAYLOAD,
        )

    assert repeated.value.errno == errno.EIO
    synchronized_file_types = []

    def observe_fsync(descriptor):
        synchronized_file_types.append(stat.S_IFMT(os.fstat(descriptor).st_mode))
        original_fsync(descriptor)

    monkeypatch.setattr(
        activation_delivery.os,
        "fsync",
        observe_fsync,
    )
    with publish_or_adopt_run_action_delivery(
        prepared_slot.slot,
        prepared_slot.slot_descriptor,
        _INPUT_PAYLOAD,
    ) as adopted:
        assert adopted.observation.size_bytes == len(_INPUT_PAYLOAD)
        adopted.require_final_path(_INPUT_PAYLOAD)
    assert stat.S_IFREG in synchronized_file_types
    assert stat.S_IFDIR in synchronized_file_types


def test_missing_linkat_support_fails_before_mutation(
    prepared_slot,
    monkeypatch,
):
    monkeypatch.setattr(atomic_publication, "_LINK_AT", None)

    with pytest.raises(
        RunActionAtomicPublicationError,
        match="requires linkat",
    ):
        publish_or_adopt_run_action_delivery(
            prepared_slot.slot,
            prepared_slot.slot_descriptor,
            _INPUT_PAYLOAD,
        )

    assert tuple(os.listdir(prepared_slot.slot_descriptor)) == ()


def test_unsupported_otmpfile_filesystem_error_propagates_without_fallback(
    prepared_slot,
    monkeypatch,
):
    original_open = activation_delivery.os.open

    def unsupported_anonymous_open(path, flags, mode=0o777, *, dir_fd=None):
        if (flags & os.O_TMPFILE) == os.O_TMPFILE:
            raise OSError(
                errno.EOPNOTSUPP,
                os.strerror(errno.EOPNOTSUPP),
                path,
            )
        if dir_fd is None:
            return original_open(path, flags, mode)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(
        activation_delivery.os,
        "open",
        unsupported_anonymous_open,
    )

    with pytest.raises(OSError) as raised:
        publish_or_adopt_run_action_delivery(
            prepared_slot.slot,
            prepared_slot.slot_descriptor,
            _INPUT_PAYLOAD,
        )

    assert raised.value.errno == errno.EOPNOTSUPP
    assert tuple(os.listdir(prepared_slot.slot_descriptor)) == ()


@pytest.mark.skipif(
    not Path("/dev/shm").is_dir() or not os.access("/dev/shm", os.W_OK),
    reason="writable tmpfs is unavailable",
)
def test_real_tmpfs_supports_atomic_publication_and_adoption():
    with tempfile.TemporaryDirectory(dir="/dev/shm") as root_path:
        fixture = _open_prepared_slot(Path(root_path))
        with ExitStack() as resources:
            resources.callback(os.close, fixture.root_descriptor)
            resources.callback(os.close, fixture.slot_descriptor)
            with publish_or_adopt_run_action_delivery(
                fixture.slot,
                fixture.slot_descriptor,
                _INPUT_PAYLOAD,
            ) as published:
                published_observation = published.observation
            with publish_or_adopt_run_action_delivery(
                fixture.slot,
                fixture.slot_descriptor,
                _INPUT_PAYLOAD,
            ) as adopted:
                adopted_observation = adopted.require_final_path(_INPUT_PAYLOAD)

    assert adopted_observation == published_observation
