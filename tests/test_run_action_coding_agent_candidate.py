import os
import stat
from contextlib import ExitStack

import pytest

import kapso.cross_run.launch.run_action_coding_agent_candidate as candidate_module
from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.launch.run_action_coding_agent_candidate import (
    RunActionCodingAgentCandidateError,
    publish_coding_agent_result_candidate,
)


def _temporary_descriptor(tmp_path, cleanup):
    tmp_path.chmod(0o700)
    descriptor = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    cleanup.callback(os.close, descriptor)
    return descriptor


def test_candidate_is_published_complete_private_and_no_replace(tmp_path):
    payload = b'{"complete":true}'
    with ExitStack() as cleanup:
        descriptor = _temporary_descriptor(tmp_path, cleanup)

        observed = publish_coding_agent_result_candidate(
            descriptor,
            payload,
            maximum_size_bytes=len(payload),
        )

    candidate_path = tmp_path / "result.candidate"
    metadata = candidate_path.stat(follow_symlinks=False)
    assert candidate_path.read_bytes() == payload
    assert stat.S_IMODE(metadata.st_mode) == 0o600
    assert metadata.st_nlink == 1
    assert observed.size_bytes == len(payload)
    assert observed.content_digest == tree_or_blob_digest(payload)
    assert (observed.device, observed.inode) == (metadata.st_dev, metadata.st_ino)

    with ExitStack() as cleanup:
        descriptor = _temporary_descriptor(tmp_path, cleanup)
        with pytest.raises(
            RunActionCodingAgentCandidateError,
            match="already exists",
        ):
            publish_coding_agent_result_candidate(
                descriptor,
                payload,
                maximum_size_bytes=len(payload),
            )
    assert candidate_path.read_bytes() == payload


@pytest.mark.parametrize(
    ("payload", "maximum_size_bytes"),
    (
        (b"", 1),
        (b"oversized", 1),
        ("not-bytes", 128),
    ),
)
def test_candidate_rejects_empty_oversized_and_non_bytes_payload(
    tmp_path,
    payload,
    maximum_size_bytes,
):
    with ExitStack() as cleanup:
        descriptor = _temporary_descriptor(tmp_path, cleanup)
        with pytest.raises(
            RunActionCodingAgentCandidateError,
            match="invalid or unbounded",
        ):
            publish_coding_agent_result_candidate(
                descriptor,
                payload,
                maximum_size_bytes=maximum_size_bytes,
            )
    assert not (tmp_path / "result.candidate").exists()


@pytest.mark.parametrize("mode", (0o755, 0o750, 0o770))
def test_candidate_rejects_non_private_temporary_directory(tmp_path, mode):
    tmp_path.chmod(mode)
    with ExitStack() as cleanup:
        descriptor = os.open(
            tmp_path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        cleanup.callback(os.close, descriptor)
        with pytest.raises(
            RunActionCodingAgentCandidateError,
            match="lacks private authority",
        ):
            publish_coding_agent_result_candidate(
                descriptor,
                b"payload",
                maximum_size_bytes=7,
            )


def test_pre_link_failure_leaves_no_visible_candidate(tmp_path, monkeypatch):
    def fail_before_link(*_arguments):
        raise RuntimeError("injected pre-link failure")

    monkeypatch.setattr(
        candidate_module,
        "link_run_action_anonymous_file_no_replace",
        fail_before_link,
    )
    with ExitStack() as cleanup:
        descriptor = _temporary_descriptor(tmp_path, cleanup)
        with pytest.raises(RuntimeError, match="injected pre-link failure"):
            publish_coding_agent_result_candidate(
                descriptor,
                b"payload",
                maximum_size_bytes=7,
            )

    assert tuple(tmp_path.iterdir()) == ()


def test_post_link_failure_leaves_only_the_complete_candidate(
    tmp_path,
    monkeypatch,
):
    original_fsync = candidate_module.os.fsync
    call_count = 0

    def fail_directory_fsync(descriptor):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("injected directory-fsync failure")
        return original_fsync(descriptor)

    monkeypatch.setattr(candidate_module.os, "fsync", fail_directory_fsync)
    with ExitStack() as cleanup:
        descriptor = _temporary_descriptor(tmp_path, cleanup)
        with pytest.raises(RuntimeError, match="directory-fsync failure"):
            publish_coding_agent_result_candidate(
                descriptor,
                b"payload",
                maximum_size_bytes=7,
            )

    assert tuple(path.name for path in tmp_path.iterdir()) == ("result.candidate",)
    assert (tmp_path / "result.candidate").read_bytes() == b"payload"


def test_existing_symlink_or_hard_link_candidate_is_never_adopted(tmp_path):
    external = tmp_path / "external"
    external.write_bytes(b"external")
    external.chmod(0o600)
    candidate = tmp_path / "result.candidate"
    candidate.symlink_to(external.name)

    with ExitStack() as cleanup:
        descriptor = _temporary_descriptor(tmp_path, cleanup)
        with pytest.raises(
            RunActionCodingAgentCandidateError,
            match="already exists",
        ):
            publish_coding_agent_result_candidate(
                descriptor,
                b"payload",
                maximum_size_bytes=7,
            )
    assert candidate.is_symlink()
    assert external.read_bytes() == b"external"

    candidate.unlink()
    os.link(external, candidate)
    with ExitStack() as cleanup:
        descriptor = _temporary_descriptor(tmp_path, cleanup)
        with pytest.raises(
            RunActionCodingAgentCandidateError,
            match="already exists",
        ):
            publish_coding_agent_result_candidate(
                descriptor,
                b"payload",
                maximum_size_bytes=7,
            )
    assert external.stat().st_nlink == 2
    assert external.read_bytes() == b"external"
