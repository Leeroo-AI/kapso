"""Closed physical publication for release and timeout control files."""

from __future__ import annotations

import os
import stat
from contextlib import ExitStack
from pathlib import Path
from threading import Thread

import pytest

import kapso.cross_run.launch.run_action_control_candidate as candidate_module
from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.launch.run_action_atomic_publication import (
    open_run_action_anonymous_file,
)
from kapso.cross_run.launch.run_action_control_candidate import (
    _CONTROL_FILE_CANDIDATE_ISSUANCE_AUTHORITY,
    _CONTROL_FILE_CANDIDATE_PUBLICATION_AUTHORITY,
    _RunActionControlFileTransition,
    _RunActionFrozenControlFileCandidate,
    RunActionControlCandidateError,
)

_PAYLOAD_SIZE_LIMIT_BYTES = 4096


def _write_release(control: Path) -> None:
    release = control / "release"
    release.write_bytes(b'{"release":"retained"}')
    release.chmod(0o400)


def _candidate(
    opened: ExitStack,
    control: Path,
    transition: _RunActionControlFileTransition,
    payload: bytes,
) -> _RunActionFrozenControlFileCandidate:
    control_descriptor = os.open(
        control,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    opened.callback(os.close, control_descriptor)
    predecessor_descriptor = None
    if transition is _RunActionControlFileTransition.TIMEOUT:
        predecessor_descriptor = os.open(
            "release",
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=control_descriptor,
        )
        opened.callback(os.close, predecessor_descriptor)
    anonymous_descriptor = open_run_action_anonymous_file(
        control_descriptor,
        0o600,
    )
    opened.callback(os.close, anonymous_descriptor)
    candidate = _RunActionFrozenControlFileCandidate(
        transition=transition,
        control_directory_descriptor=control_descriptor,
        anonymous_file_descriptor=anonymous_descriptor,
        predecessor_file_descriptor=predecessor_descriptor,
        owner_user_id=os.getuid(),
        owner_group_id=os.getgid(),
        payload_size_limit_bytes=_PAYLOAD_SIZE_LIMIT_BYTES,
        payload=payload,
        _authority=_CONTROL_FILE_CANDIDATE_ISSUANCE_AUTHORITY,
    )
    opened.callback(candidate.close)
    return candidate


@pytest.mark.parametrize(
    ("transition", "before_entries", "final_file_name"),
    (
        (_RunActionControlFileTransition.RELEASE, (), "release"),
        (_RunActionControlFileTransition.TIMEOUT, ("release",), "timeout"),
    ),
)
def test_closed_transition_publishes_one_exact_durable_inode(
    tmp_path,
    transition,
    before_entries,
    final_file_name,
):
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    if before_entries:
        _write_release(control)
    payload = f'{{"transition":"{transition.value}"}}'.encode("ascii")
    with ExitStack() as opened:
        candidate = _candidate(opened, control, transition, payload)
        assert (
            candidate._begin_publication(
                payload,
                _authority=_CONTROL_FILE_CANDIDATE_PUBLICATION_AUTHORITY,
            )
            == payload
        )
        candidate._prepare_authorized_link_once(
            _authority=_CONTROL_FILE_CANDIDATE_PUBLICATION_AUTHORITY,
        )
        evidence = candidate._link_prepared_once(
            _authority=_CONTROL_FILE_CANDIDATE_PUBLICATION_AUTHORITY,
        )
        final_path = control / final_file_name
        metadata = final_path.stat()
        assert final_path.read_bytes() == payload
        assert tuple(sorted(path.name for path in control.iterdir())) == (
            *before_entries,
            final_file_name,
        )
        assert evidence.transition is transition
        assert evidence.final_file_name == final_file_name
        assert (evidence.device, evidence.inode) == (
            metadata.st_dev,
            metadata.st_ino,
        )
        assert evidence.mode == 0o400
        assert evidence.link_count == 1
        assert evidence.size_bytes == len(payload)
        assert evidence.content_digest == tree_or_blob_digest(payload)
        assert stat.S_IMODE(metadata.st_mode) == 0o400
        assert metadata.st_nlink == 1
        assert metadata.st_ino != control.stat().st_ino
        if transition is _RunActionControlFileTransition.TIMEOUT:
            assert metadata.st_ino != (control / "release").stat().st_ino


@pytest.mark.parametrize(
    ("transition", "predecessor_present"),
    (
        (_RunActionControlFileTransition.RELEASE, True),
        (_RunActionControlFileTransition.TIMEOUT, False),
    ),
)
def test_transition_shape_rejects_the_wrong_predecessor_requirement(
    tmp_path,
    transition,
    predecessor_present,
):
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    control_descriptor = os.open(
        control,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    with ExitStack() as opened:
        opened.callback(os.close, control_descriptor)
        predecessor_descriptor = control_descriptor if predecessor_present else None
        anonymous_descriptor = open_run_action_anonymous_file(
            control_descriptor,
            0o600,
        )
        opened.callback(os.close, anonymous_descriptor)
        with pytest.raises(
            RunActionControlCandidateError,
            match="exact closed transition",
        ):
            _RunActionFrozenControlFileCandidate(
                transition=transition,
                control_directory_descriptor=control_descriptor,
                anonymous_file_descriptor=anonymous_descriptor,
                predecessor_file_descriptor=predecessor_descriptor,
                owner_user_id=os.getuid(),
                owner_group_id=os.getgid(),
                payload_size_limit_bytes=_PAYLOAD_SIZE_LIMIT_BYTES,
                payload=b"candidate",
                _authority=_CONTROL_FILE_CANDIDATE_ISSUANCE_AUTHORITY,
            )


def test_timeout_candidate_detects_release_path_replacement(tmp_path):
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    _write_release(control)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b'{"release":"retained"}')
    replacement.chmod(0o400)
    with ExitStack() as opened:
        candidate = _candidate(
            opened,
            control,
            _RunActionControlFileTransition.TIMEOUT,
            b'{"timeout":"candidate"}',
        )
        replacement.replace(control / "release")
        with pytest.raises(
            RunActionControlCandidateError,
            match="predecessor release",
        ):
            candidate._begin_publication(
                b'{"timeout":"candidate"}',
                _authority=_CONTROL_FILE_CANDIDATE_PUBLICATION_AUTHORITY,
            )


def test_prepared_timeout_link_rejects_release_replacement_after_begin(tmp_path):
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    _write_release(control)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b'{"release":"retained"}')
    replacement.chmod(0o400)
    payload = b'{"timeout":"candidate"}'
    with ExitStack() as opened:
        candidate = _candidate(
            opened,
            control,
            _RunActionControlFileTransition.TIMEOUT,
            payload,
        )
        candidate._begin_publication(
            payload,
            _authority=_CONTROL_FILE_CANDIDATE_PUBLICATION_AUTHORITY,
        )
        replacement.replace(control / "release")

        with pytest.raises(
            RunActionControlCandidateError,
            match="predecessor release",
        ):
            candidate._prepare_authorized_link_once(
                _authority=_CONTROL_FILE_CANDIDATE_PUBLICATION_AUTHORITY,
            )
        assert not (control / "timeout").exists()


def test_prepared_release_link_rejects_topology_replacement_after_begin(tmp_path):
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    payload = b'{"release":"candidate"}'
    with ExitStack() as opened:
        candidate = _candidate(
            opened,
            control,
            _RunActionControlFileTransition.RELEASE,
            payload,
        )
        candidate._begin_publication(
            payload,
            _authority=_CONTROL_FILE_CANDIDATE_PUBLICATION_AUTHORITY,
        )
        (control / "foreign").write_bytes(b"foreign")

        with pytest.raises(
            RunActionControlCandidateError,
            match="topology",
        ):
            candidate._prepare_authorized_link_once(
                _authority=_CONTROL_FILE_CANDIDATE_PUBLICATION_AUTHORITY,
            )
        assert not (control / "release").exists()


def test_candidate_is_owner_thread_bound_and_single_use(tmp_path):
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    payload = b'{"release":"candidate"}'
    with ExitStack() as opened:
        candidate = _candidate(
            opened,
            control,
            _RunActionControlFileTransition.RELEASE,
            payload,
        )
        failures = []

        def foreign_thread():
            with pytest.raises(
                RunActionControlCandidateError,
                match="unissued, spent, or foreign",
            ):
                candidate._begin_publication(
                    payload,
                    _authority=_CONTROL_FILE_CANDIDATE_PUBLICATION_AUTHORITY,
                )
            failures.append("thread")

        thread = Thread(target=foreign_thread)
        thread.start()
        thread.join()
        assert failures == ["thread"]
        candidate._begin_publication(
            payload,
            _authority=_CONTROL_FILE_CANDIDATE_PUBLICATION_AUTHORITY,
        )
        candidate._prepare_authorized_link_once(
            _authority=_CONTROL_FILE_CANDIDATE_PUBLICATION_AUTHORITY,
        )
        candidate._link_prepared_once(
            _authority=_CONTROL_FILE_CANDIDATE_PUBLICATION_AUTHORITY,
        )
        with pytest.raises(
            RunActionControlCandidateError,
            match="unissued, spent, or foreign",
        ):
            candidate._link_prepared_once(
                _authority=_CONTROL_FILE_CANDIDATE_PUBLICATION_AUTHORITY,
            )


def test_link_is_no_replace_and_never_overwrites_a_racing_path(
    tmp_path,
    monkeypatch,
):
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    payload = b'{"release":"candidate"}'
    with ExitStack() as opened:
        candidate = _candidate(
            opened,
            control,
            _RunActionControlFileTransition.RELEASE,
            payload,
        )
        candidate._begin_publication(
            payload,
            _authority=_CONTROL_FILE_CANDIDATE_PUBLICATION_AUTHORITY,
        )
        candidate._prepare_authorized_link_once(
            _authority=_CONTROL_FILE_CANDIDATE_PUBLICATION_AUTHORITY,
        )
        racing_payload = b"racing release"
        original_link = candidate_module.link_run_action_anonymous_file_no_replace

        def race_before_link(
            anonymous_descriptor,
            directory_descriptor,
            final_file_name,
        ):
            (control / "release").write_bytes(racing_payload)
            original_link(
                anonymous_descriptor,
                directory_descriptor,
                final_file_name,
            )

        monkeypatch.setattr(
            candidate_module,
            "link_run_action_anonymous_file_no_replace",
            race_before_link,
        )
        with pytest.raises(FileExistsError):
            candidate._link_prepared_once(
                _authority=_CONTROL_FILE_CANDIDATE_PUBLICATION_AUTHORITY,
            )
        assert (control / "release").read_bytes() == racing_payload


def test_publication_fsyncs_anonymous_linked_and_control_inodes(
    tmp_path,
    monkeypatch,
):
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    fsynced_file_types = []
    original_fsync = os.fsync

    def observe_fsync(descriptor):
        metadata = os.fstat(descriptor)
        fsynced_file_types.append(
            "directory" if stat.S_ISDIR(metadata.st_mode) else "regular"
        )
        original_fsync(descriptor)

    monkeypatch.setattr(candidate_module.os, "fsync", observe_fsync)
    payload = b'{"release":"candidate"}'
    with ExitStack() as opened:
        candidate = _candidate(
            opened,
            control,
            _RunActionControlFileTransition.RELEASE,
            payload,
        )
        candidate._begin_publication(
            payload,
            _authority=_CONTROL_FILE_CANDIDATE_PUBLICATION_AUTHORITY,
        )
        candidate._prepare_authorized_link_once(
            _authority=_CONTROL_FILE_CANDIDATE_PUBLICATION_AUTHORITY,
        )
        candidate._link_prepared_once(
            _authority=_CONTROL_FILE_CANDIDATE_PUBLICATION_AUTHORITY,
        )

    assert fsynced_file_types.count("regular") >= 2
    assert "directory" in fsynced_file_types
