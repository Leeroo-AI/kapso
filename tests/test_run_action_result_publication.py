"""Crash-atomic publication tests for the previously absent result path."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import pytest

import kapso.cross_run.launch.run_action_atomic_publication as atomic_publication
import kapso.cross_run.launch.run_action_result_publication as result_publication
from kapso.core.config import load_config
from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.launch.run_action_atomic_publication import (
    RunActionAtomicPublicationError,
)
from kapso.cross_run.launch.run_action_result_publication import (
    publish_or_adopt_run_action_result,
    RunActionResultPublicationError,
)
from kapso.cross_run.settings import CrossRunSettings
from test_run_action_supervisor_contracts import (
    _RUN_ACTION_PROCESS_SNAPSHOT_SIZE_BYTES,
)

_MAXIMUM_RESULT_SIZE_BYTES = CrossRunSettings.from_dict(
    load_config("src/kapso/config.yaml")["cross_run"]
).launch.run_action_result_size_bytes
_RESULT_PAYLOAD = b'{"status":"succeeded"}'


@dataclass(frozen=True)
class _ResultDirectoryFixture:
    root_descriptor: int
    result_descriptor: int
    result_path: Path


@pytest.fixture
def result_directory(tmp_path: Path):
    root_descriptor = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
    )
    os.mkdir("result", mode=0o700, dir_fd=root_descriptor)
    result_descriptor = os.open(
        "result",
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=root_descriptor,
    )
    os.fchmod(result_descriptor, 0o700)
    yield _ResultDirectoryFixture(
        root_descriptor=root_descriptor,
        result_descriptor=result_descriptor,
        result_path=tmp_path / "result",
    )
    os.close(result_descriptor)
    os.close(root_descriptor)


def _publish(
    result_descriptor: int,
    payload: bytes = _RESULT_PAYLOAD,
    maximum_size_bytes: int = _MAXIMUM_RESULT_SIZE_BYTES,
):
    return publish_or_adopt_run_action_result(
        result_descriptor,
        payload,
        maximum_size_bytes=maximum_size_bytes,
        process_snapshot_size_limit_bytes=_RUN_ACTION_PROCESS_SNAPSHOT_SIZE_BYTES,
    )


def test_result_publication_links_one_complete_nonempty_inode(result_directory):
    observation = _publish(result_directory.result_descriptor)

    assert os.listdir(result_directory.result_descriptor) == ["result.blob"]
    assert observation.relative_path == "result/result.blob"
    assert observation.file_type == "regular"
    assert observation.mode == 0o600
    assert observation.link_count == 1
    assert observation.size_bytes == len(_RESULT_PAYLOAD)
    assert observation.content_digest == tree_or_blob_digest(_RESULT_PAYLOAD)
    assert observation.parent_inode != observation.inode
    result_descriptor = os.open(
        "result.blob",
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=result_directory.result_descriptor,
    )
    os.close(result_descriptor)


def test_result_publication_adopts_only_the_exact_completed_payload(result_directory):
    first = _publish(result_directory.result_descriptor)
    second = _publish(result_directory.result_descriptor)

    assert second == first
    with pytest.raises(RunActionAtomicPublicationError):
        _publish(result_directory.result_descriptor, b'{"status":"different"}')


def test_concurrent_exact_publishers_serialize_and_adopt_one_inode(
    result_directory,
):
    def publish_from_distinct_open_description():
        with ExitStack() as descriptors:
            descriptor = os.open(
                result_directory.result_path,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
            descriptors.callback(os.close, descriptor)
            return _publish(descriptor)

    with ThreadPoolExecutor(max_workers=2) as executor:
        observations = tuple(
            executor.map(
                lambda _invocation: publish_from_distinct_open_description(),
                range(2),
            )
        )

    assert observations[0] == observations[1]


@pytest.mark.parametrize(
    ("payload", "maximum_size_bytes"),
    (
        (b"", _MAXIMUM_RESULT_SIZE_BYTES),
        (b"xx", 1),
    ),
)
def test_result_publication_rejects_empty_or_oversized_payload(
    result_directory,
    payload,
    maximum_size_bytes,
):
    with pytest.raises(RunActionResultPublicationError):
        _publish(
            result_directory.result_descriptor,
            payload,
            maximum_size_bytes,
        )

    assert os.listdir(result_directory.result_descriptor) == []


def test_failure_before_link_leaves_final_path_absent(
    result_directory,
    monkeypatch,
):
    def fail_link(*_arguments):
        raise RuntimeError("simulated publication loss")

    monkeypatch.setattr(
        result_publication,
        "link_run_action_anonymous_file_no_replace",
        fail_link,
    )

    with pytest.raises(RuntimeError, match="simulated publication loss"):
        _publish(result_directory.result_descriptor)

    assert os.listdir(result_directory.result_descriptor) == []


def test_partial_anonymous_write_never_exposes_final_path(
    result_directory,
    monkeypatch,
):
    original_write = atomic_publication.os.write
    write_calls = 0

    def fail_after_partial_write(descriptor, payload):
        nonlocal write_calls
        write_calls += 1
        if write_calls == 1:
            return original_write(descriptor, payload[:1])
        raise OSError("simulated write loss")

    monkeypatch.setattr(atomic_publication.os, "write", fail_after_partial_write)

    with pytest.raises(OSError, match="simulated write loss"):
        _publish(result_directory.result_descriptor)

    assert os.listdir(result_directory.result_descriptor) == []


def test_loss_after_link_leaves_only_complete_adoptable_result(
    result_directory,
    monkeypatch,
):
    original_fsync = result_publication.os.fsync
    fsync_calls = 0

    def fail_directory_sync(descriptor):
        nonlocal fsync_calls
        fsync_calls += 1
        if fsync_calls == 2:
            raise OSError("simulated directory sync loss")
        return original_fsync(descriptor)

    with monkeypatch.context() as publication_loss:
        publication_loss.setattr(
            result_publication.os,
            "fsync",
            fail_directory_sync,
        )
        with pytest.raises(OSError, match="simulated directory sync loss"):
            _publish(result_directory.result_descriptor)

    result_descriptor = os.open(
        "result.blob",
        os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
        dir_fd=result_directory.result_descriptor,
    )
    with os.fdopen(result_descriptor, "rb", buffering=0) as result_file:
        assert result_file.read() == _RESULT_PAYLOAD
    assert _publish(result_directory.result_descriptor).content_digest == (
        tree_or_blob_digest(_RESULT_PAYLOAD)
    )


def test_named_temporary_or_extra_entry_is_rejected(result_directory):
    descriptor = os.open(
        "partial",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
        dir_fd=result_directory.result_descriptor,
    )
    os.close(descriptor)

    with pytest.raises(
        RunActionResultPublicationError,
        match="unexpected entry",
    ):
        _publish(result_directory.result_descriptor)
