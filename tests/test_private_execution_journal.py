import os
import stat

import pytest

import kapso.cross_run.expert.private_execution_journal as journal_module
from kapso.cross_run.expert.private_execution_journal import (
    ExecutionJournalFilesystem,
    ExecutionJournalLock,
    ExecutionJournalResultBlob,
    ExecutionJournalStoreError,
    NumberedExecutionJournalPayload,
)


def _filesystem(tmp_path, *, staging_limit=2):
    trusted_root = (tmp_path / "trusted").resolve()
    trusted_root.mkdir(mode=0o700)
    filesystem = ExecutionJournalFilesystem(
        trusted_root / "executions",
        trusted_root,
        maximum_event_size_bytes=64,
        maximum_result_size_bytes=64,
        maximum_staging_entry_count=staging_limit,
    )
    return trusted_root, filesystem


def test_layout_is_private_and_rejects_untrusted_paths_and_identifiers(tmp_path):
    trusted_root, filesystem = _filesystem(tmp_path)

    for path in (
        filesystem.root,
        filesystem.lock_root,
        filesystem.reservation_root,
    ):
        assert stat.S_IMODE(path.stat().st_mode) == 0o700
    assert filesystem.root.parent == trusted_root

    for invalid_digest in ("a" * 63, "A" * 64, "../" + "a" * 64):
        with pytest.raises(ExecutionJournalStoreError, match="64 lowercase hex"):
            filesystem.ensure_reservation_layout(invalid_digest)
    with pytest.raises(ExecutionJournalStoreError, match="lock filename"):
        filesystem.lock("../foreign.lock")
    with pytest.raises(ExecutionJournalStoreError, match="event number"):
        filesystem.event_path("a" * 64, 0)
    with pytest.raises(ExecutionJournalStoreError, match="must be bytes"):
        filesystem.publish_numbered_event("a" * 64, 1, "not-bytes")

    trusted_root.chmod(0o777)
    with pytest.raises(ExecutionJournalStoreError, match="owner-private"):
        ExecutionJournalFilesystem(
            trusted_root / "other",
            trusted_root,
            maximum_event_size_bytes=1,
            maximum_result_size_bytes=1,
            maximum_staging_entry_count=1,
        )


def test_layout_query_distinguishes_absent_empty_complete_and_corrupt(tmp_path):
    _, filesystem = _filesystem(tmp_path)
    absent_digest = "a" * 64
    empty_digest = "b" * 64
    nonempty_digest = "c" * 64
    symlink_digest = "d" * 64

    assert not filesystem.has_complete_reservation_layout(absent_digest)

    empty_root = filesystem.reservation_path(empty_digest)
    empty_root.mkdir(mode=0o700)
    (empty_root / "events").mkdir(mode=0o700)
    assert not filesystem.has_complete_reservation_layout(empty_digest)
    filesystem.ensure_reservation_layout(empty_digest)
    assert filesystem.has_complete_reservation_layout(empty_digest)

    nonempty_root = filesystem.reservation_path(nonempty_digest)
    nonempty_root.mkdir(mode=0o700)
    nonempty_events = nonempty_root / "events"
    nonempty_events.mkdir(mode=0o700)
    durable_child = nonempty_events / "durable"
    durable_child.write_bytes(b"state")
    durable_child.chmod(0o400)
    with pytest.raises(ExecutionJournalStoreError, match="durable state"):
        filesystem.has_complete_reservation_layout(nonempty_digest)

    os.symlink(tmp_path / "missing", filesystem.reservation_path(symlink_digest))
    with pytest.raises(ExecutionJournalStoreError, match="real directory"):
        filesystem.has_complete_reservation_layout(symlink_digest)


def test_numbered_event_publication_is_bounded_private_and_create_only(tmp_path):
    _, filesystem = _filesystem(tmp_path)
    digest = "a" * 64
    filesystem.ensure_reservation_layout(digest)

    filesystem.publish_numbered_event(digest, 1, b"first")
    event_path = filesystem.event_path(digest, 1)
    assert stat.S_IMODE(event_path.stat().st_mode) == 0o400
    assert filesystem.read_numbered_event_payloads(digest, 1) == (
        NumberedExecutionJournalPayload(event_number=1, payload=b"first"),
    )

    with pytest.raises(OSError):
        filesystem.publish_numbered_event(digest, 1, b"fork")
    assert event_path.read_bytes() == b"first"
    with pytest.raises(ExecutionJournalStoreError, match="configured bound"):
        filesystem.publish_numbered_event(digest, 2, b"x" * 65)
    with pytest.raises(ExecutionJournalStoreError, match="structural event bound"):
        filesystem.read_numbered_event_payloads(digest, 0)

    event_path.chmod(0o600)
    with pytest.raises(ExecutionJournalStoreError, match="private independent"):
        filesystem.read_numbered_event_payloads(digest, 1)


def test_post_rename_failure_leaves_one_readable_create_only_event(
    tmp_path,
    monkeypatch,
):
    _, filesystem = _filesystem(tmp_path)
    digest = "a" * 64
    filesystem.ensure_reservation_layout(digest)
    events_root = filesystem.events_path(digest)
    original_fsync = journal_module._fsync_directory

    def fsync_then_fail(path):
        original_fsync(path)
        if path == events_root:
            raise OSError("post-rename response lost")

    monkeypatch.setattr(journal_module, "_fsync_directory", fsync_then_fail)
    with pytest.raises(OSError, match="response lost"):
        filesystem.publish_numbered_event(digest, 1, b"durable")

    monkeypatch.setattr(journal_module, "_fsync_directory", original_fsync)
    assert filesystem.read_numbered_event_payloads(digest, 1) == (
        NumberedExecutionJournalPayload(event_number=1, payload=b"durable"),
    )
    with pytest.raises(OSError):
        filesystem.publish_numbered_event(digest, 1, b"fork")


def test_result_blobs_are_content_addressed_bounded_and_integrity_checked(tmp_path):
    _, filesystem = _filesystem(tmp_path)
    digest = "a" * 64
    filesystem.ensure_reservation_layout(digest)

    blob = filesystem.publish_result(digest, b"result")
    assert type(blob) is ExecutionJournalResultBlob
    assert filesystem.read_result(digest, blob) == b"result"
    assert stat.S_IMODE(filesystem.result_path(digest, blob).stat().st_mode) == 0o400
    filesystem.validate_results(digest, 1)

    with pytest.raises(OSError):
        filesystem.publish_result(digest, b"result")
    with pytest.raises(ExecutionJournalStoreError, match="structural bound"):
        filesystem.validate_results(digest, 0)
    with pytest.raises(ExecutionJournalStoreError, match="configured bound"):
        filesystem.publish_result(digest, b"x" * 65)
    with pytest.raises(ExecutionJournalStoreError, match="not exact"):
        filesystem.read_result(digest, object())

    result_path = filesystem.result_path(digest, blob)
    result_path.chmod(0o600)
    result_path.write_bytes(b"mutated")
    result_path.chmod(0o400)
    with pytest.raises(ExecutionJournalStoreError, match="differs"):
        filesystem.read_result(digest, blob)


def test_staging_cleanup_validates_the_whole_bounded_set_before_unlink(tmp_path):
    _, filesystem = _filesystem(tmp_path, staging_limit=2)
    digest = "a" * 64
    filesystem.ensure_reservation_layout(digest)
    staging_root = filesystem.staging_path(digest)
    valid = staging_root / f".event-{'a' * 32}.tmp"
    invalid = staging_root / "unexpected"
    valid.write_bytes(b"orphan")
    valid.chmod(0o600)
    invalid.write_bytes(b"unsafe")
    invalid.chmod(0o600)

    with pytest.raises(ExecutionJournalStoreError, match="unexpected entry"):
        filesystem.clean_staging(digest)
    assert valid.exists()

    invalid.unlink()
    filesystem.clean_staging(digest)
    assert not valid.exists()

    for token in ("a", "b", "c"):
        path = staging_root / f".result-{token * 32}.tmp"
        path.write_bytes(b"orphan")
        path.chmod(0o600)
    with pytest.raises(ExecutionJournalStoreError, match="configured bound"):
        filesystem.clean_staging(digest)
    assert len(tuple(staging_root.iterdir())) == 3


def test_locks_are_private_single_use_creator_process_authority(tmp_path):
    _, filesystem = _filesystem(tmp_path)
    digest = "a" * 64
    lock = filesystem.reservation_lock(digest)

    with pytest.raises(ExecutionJournalStoreError, match="not held"):
        lock.require_acquired()
    with lock:
        assert isinstance(lock, ExecutionJournalLock)
        lock.require_acquired()
        assert lock.owner_process_id == os.getpid()
        assert stat.S_IMODE(lock.path.stat().st_mode) == 0o600
        with pytest.raises(ExecutionJournalStoreError, match="entered twice"):
            lock.__enter__()
    with pytest.raises(ExecutionJournalStoreError, match="not held"):
        lock.require_acquired()

    named_lock = filesystem.lock(f"candidate-stage-{'b' * 64}.lock")
    with named_lock:
        named_lock.require_acquired()
