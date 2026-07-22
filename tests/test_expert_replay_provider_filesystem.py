from __future__ import annotations

import os
import stat
from dataclasses import replace

import pytest

from kapso.cross_run.expert import replay_provider_filesystem
from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.contracts import SourceFileDescriptor
from kapso.cross_run.expert.replay_execution import (
    ExpertSourceReplayMatchedLegInvocation,
    expert_source_replay_execution_provider_key,
    source_replay_provider_execution_handle,
)
from kapso.cross_run.expert.replay_protocol import (
    TaskEvaluatorInvocationAllocation,
    build_task_evaluator_request,
)
from kapso.cross_run.expert.replay_provider_filesystem import (
    SourceReplayProviderFilesystemError,
    cleanup_source_replay_provider_workspace,
    materialize_source_replay_provider_inputs,
    materialize_verified_byte_tree,
    parse_source_replay_result_snapshot,
)
from test_expert_source_replay_request import _prepared, _request_fixture

_TAR_BLOCK_SIZE = 512
_RESULT_PAYLOAD = b'{"completed":true}'


@pytest.fixture(scope="module")
def prepared_replay_request(tmp_path_factory):
    return _prepared(
        _request_fixture(tmp_path_factory.mktemp("expert-replay-provider-filesystem"))
    )


def _matched_invocation(prepared, leg_name):
    materialized_case = prepared.cases[0]
    leg = getattr(materialized_case.request_case, leg_name)
    allocation = TaskEvaluatorInvocationAllocation(
        reservation_id=content_id(
            "expert-source-replay-execution-reservation",
            {"provider_filesystem_leg": leg_name},
        ),
        execution_case_id=materialized_case.request_case.execution_case_id,
        execution_leg_id=leg.execution_leg_id,
        invocation_nonce="1" * 32,
    )
    provider_key = expert_source_replay_execution_provider_key(materialized_case)
    request = build_task_evaluator_request(materialized_case, allocation)
    return ExpertSourceReplayMatchedLegInvocation(
        materialized_case=materialized_case,
        expert_source=(
            prepared.parent if leg_name == "control_leg" else prepared.candidate
        ),
        invocation_allocation=allocation,
        task_evaluator_request=request,
        provider_handle=source_replay_provider_execution_handle(
            provider_key,
            allocation,
        ),
    )


def _source_file(relative_path, payload, mode):
    return SourceFileDescriptor(
        relative_path=relative_path,
        digest=tree_or_blob_digest(payload),
        mode=mode,
        size=len(payload),
    )


def _mode(path):
    return stat.S_IMODE(os.stat(path, follow_symlinks=False).st_mode)


def _tar_octal(value, width):
    encoded = f"{value:0{width - 1}o}".encode("ascii") + b"\x00"
    assert len(encoded) == width
    return encoded


def _tar_header(
    name,
    *,
    owner_id,
    size,
    type_flag,
    link_name=b"",
    magic=b"ustar  \x00",
):
    header = bytearray(_TAR_BLOCK_SIZE)
    encoded_name = name.encode("ascii")
    header[0 : len(encoded_name)] = encoded_name
    header[100:108] = _tar_octal(0o700 if type_flag == b"5" else 0o600, 8)
    header[108:116] = _tar_octal(owner_id, 8)
    header[116:124] = _tar_octal(owner_id, 8)
    header[124:136] = _tar_octal(size, 12)
    header[136:148] = _tar_octal(0, 12)
    header[148:156] = b"        "
    header[156:157] = type_flag
    header[157 : 157 + len(link_name)] = link_name
    header[257:265] = magic
    checksum = sum(header)
    header[148:156] = f"{checksum:06o}".encode("ascii") + b"\x00 "
    return bytes(header)


def _tar_snapshot(entries, *, end_blocks=2):
    archive = bytearray()
    for name, owner_id, type_flag, payload, link_name in entries:
        archive.extend(
            _tar_header(
                name,
                owner_id=owner_id,
                size=len(payload),
                type_flag=type_flag,
                link_name=link_name,
            )
        )
        archive.extend(payload)
        archive.extend(bytes((-len(payload)) % _TAR_BLOCK_SIZE))
    archive.extend(bytes(_TAR_BLOCK_SIZE * end_blocks))
    return bytes(archive)


def _valid_result_snapshot(owner_id):
    return _tar_snapshot(
        (
            ("./", owner_id, b"5", b"", b""),
            ("./result.json", owner_id, b"0", _RESULT_PAYLOAD, b""),
        )
    )


def _parse(snapshot, owner_id):
    return parse_source_replay_result_snapshot(
        snapshot,
        expected_owner_id=owner_id,
        expected_group_id=owner_id,
        maximum_result_bytes=len(_RESULT_PAYLOAD),
        maximum_snapshot_bytes=len(snapshot),
    )


def test_verified_byte_tree_preserves_regular_and_executable_modes(tmp_path):
    trusted_root = (tmp_path / "trusted").resolve()
    trusted_root.mkdir(mode=0o700)
    payloads = {
        "bin/run": b"#!/bin/sh\nexit 0\n",
        "settings.json": b"{}",
    }
    descriptors = tuple(
        _source_file(path, payloads[path], mode)
        for path, mode in (("bin/run", "100755"), ("settings.json", "100644"))
    )
    destination = trusted_root / "tree"

    materialize_verified_byte_tree(
        trusted_root=trusted_root,
        destination_root=destination,
        descriptors=descriptors,
        source_contents=payloads,
    )

    assert (destination / "bin/run").read_bytes() == payloads["bin/run"]
    assert (destination / "settings.json").read_bytes() == payloads["settings.json"]
    assert _mode(destination) == 0o700
    assert _mode(destination / "bin") == 0o700
    assert _mode(destination / "bin/run") == 0o755
    assert _mode(destination / "settings.json") == 0o644


def test_verified_byte_tree_validates_before_creating_any_destination(tmp_path):
    trusted_root = (tmp_path / "trusted").resolve()
    trusted_root.mkdir(mode=0o700)
    destination = trusted_root / "tree"
    payload = b"trusted"
    descriptor = _source_file("file", payload, "100644")

    with pytest.raises(SourceReplayProviderFilesystemError, match="path closure"):
        materialize_verified_byte_tree(
            trusted_root=trusted_root,
            destination_root=destination,
            descriptors=(descriptor,),
            source_contents={"other": payload},
        )
    assert not destination.exists()

    corrupt_descriptor = replace(descriptor, size=descriptor.size + 1)
    with pytest.raises(SourceReplayProviderFilesystemError, match="descriptors"):
        materialize_verified_byte_tree(
            trusted_root=trusted_root,
            destination_root=destination,
            descriptors=(corrupt_descriptor,),
            source_contents={"file": payload},
        )
    assert not destination.exists()


def test_verified_byte_tree_rejects_untrusted_or_existing_roots(tmp_path):
    public_root = (tmp_path / "public").resolve()
    public_root.mkdir(mode=0o755)
    public_root.chmod(0o755)
    descriptor = _source_file("file", b"payload", "100644")

    with pytest.raises(SourceReplayProviderFilesystemError, match="owner-private"):
        materialize_verified_byte_tree(
            trusted_root=public_root,
            destination_root=public_root / "tree",
            descriptors=(descriptor,),
            source_contents={"file": b"payload"},
        )

    public_root.chmod(0o700)
    existing = public_root / "tree"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        materialize_verified_byte_tree(
            trusted_root=public_root,
            destination_root=existing,
            descriptors=(descriptor,),
            source_contents={"file": b"payload"},
        )


def test_provider_workspace_cleanup_removes_frozen_tree_and_is_idempotent(tmp_path):
    trusted_root = (tmp_path / "trusted").resolve()
    trusted_root.mkdir(mode=0o700)
    workspace_root = trusted_root / "workspace"
    workspace_root.mkdir(mode=0o700)
    input_root = workspace_root / "input"
    input_root.mkdir(mode=0o700)
    nested_root = input_root / "nested"
    nested_root.mkdir(mode=0o700)
    frozen_file = nested_root / "source.py"
    frozen_file.write_bytes(b"print('verified')\n")
    frozen_file.chmod(0o444)
    nested_root.chmod(0o555)
    input_root.chmod(0o555)

    cleanup_source_replay_provider_workspace(
        trusted_root=trusted_root,
        workspace_root=workspace_root,
    )
    cleanup_source_replay_provider_workspace(
        trusted_root=trusted_root,
        workspace_root=workspace_root,
    )

    assert not workspace_root.exists()
    assert tuple(trusted_root.iterdir()) == ()


def test_provider_workspace_cleanup_rejects_non_child_target(tmp_path):
    trusted_root = (tmp_path / "trusted").resolve()
    trusted_root.mkdir(mode=0o700)
    outside_workspace = (tmp_path / "workspace").resolve()
    outside_workspace.mkdir(mode=0o700)
    outside_file = outside_workspace / "keep"
    outside_file.write_bytes(b"outside")

    with pytest.raises(SourceReplayProviderFilesystemError, match="direct child"):
        cleanup_source_replay_provider_workspace(
            trusted_root=trusted_root,
            workspace_root=outside_workspace,
        )

    assert outside_file.read_bytes() == b"outside"


@pytest.mark.parametrize("replacement_kind", ("file", "symlink"))
def test_provider_workspace_cleanup_rejects_non_directory_without_following(
    tmp_path,
    replacement_kind,
):
    trusted_root = (tmp_path / "trusted").resolve()
    trusted_root.mkdir(mode=0o700)
    outside_root = (tmp_path / "outside").resolve()
    outside_root.mkdir(mode=0o700)
    outside_file = outside_root / "keep"
    outside_file.write_bytes(b"outside")
    workspace_root = trusted_root / "workspace"
    if replacement_kind == "file":
        workspace_root.write_bytes(b"not a workspace")
    else:
        workspace_root.symlink_to(outside_root, target_is_directory=True)

    with pytest.raises(
        SourceReplayProviderFilesystemError,
        match="real owned directory",
    ):
        cleanup_source_replay_provider_workspace(
            trusted_root=trusted_root,
            workspace_root=workspace_root,
        )

    assert workspace_root.exists()
    assert outside_file.read_bytes() == b"outside"


def test_provider_workspace_cleanup_rejects_nested_symlink_without_deleting_outside(
    tmp_path,
):
    trusted_root = (tmp_path / "trusted").resolve()
    trusted_root.mkdir(mode=0o700)
    workspace_root = trusted_root / "workspace"
    workspace_root.mkdir(mode=0o700)
    outside_root = (tmp_path / "outside").resolve()
    outside_root.mkdir(mode=0o700)
    outside_file = outside_root / "keep"
    outside_file.write_bytes(b"outside")
    nested_link = workspace_root / "escape"
    nested_link.symlink_to(outside_root, target_is_directory=True)

    with pytest.raises(SourceReplayProviderFilesystemError, match="link, special"):
        cleanup_source_replay_provider_workspace(
            trusted_root=trusted_root,
            workspace_root=workspace_root,
        )

    assert nested_link.is_symlink()
    assert outside_file.read_bytes() == b"outside"


def test_provider_workspace_cleanup_rejects_workspace_replaced_while_opening(
    tmp_path,
    monkeypatch,
):
    trusted_root = (tmp_path / "trusted").resolve()
    trusted_root.mkdir(mode=0o700)
    workspace_root = trusted_root / "workspace"
    workspace_root.mkdir(mode=0o700)
    (workspace_root / "original").write_bytes(b"original")
    moved_workspace = trusted_root / "moved-workspace"
    real_open = replay_provider_filesystem.os.open
    replaced = False

    def replace_before_workspace_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal replaced
        if path == workspace_root.name and dir_fd is not None and not replaced:
            replaced = True
            workspace_root.rename(moved_workspace)
            workspace_root.mkdir(mode=0o700)
            (workspace_root / "replacement").write_bytes(b"replacement")
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(
        replay_provider_filesystem.os,
        "open",
        replace_before_workspace_open,
    )

    with pytest.raises(
        SourceReplayProviderFilesystemError, match="changed while opening"
    ):
        cleanup_source_replay_provider_workspace(
            trusted_root=trusted_root,
            workspace_root=workspace_root,
        )

    assert (moved_workspace / "original").read_bytes() == b"original"
    assert (workspace_root / "replacement").read_bytes() == b"replacement"


@pytest.mark.parametrize("leg_name", ("control_leg", "candidate_leg"))
def test_matched_leg_inputs_materialize_exact_closures_and_freeze_read_only(
    tmp_path,
    prepared_replay_request,
    leg_name,
):
    trusted_root = (tmp_path / "provider").resolve()
    trusted_root.mkdir(mode=0o700)
    invocation = _matched_invocation(prepared_replay_request, leg_name)

    layout = materialize_source_replay_provider_inputs(
        invocation=invocation,
        trusted_root=trusted_root,
        workspace_root=trusted_root / leg_name,
    )

    expert = invocation.expert_source
    expert_descriptors = (
        expert.source_tree.files
        if leg_name == "candidate_leg"
        else expert.parent_tree_receipt.source_extraction_receipt.source_tree_files
    )
    for descriptor in expert_descriptors:
        path = layout.expert_root / descriptor.relative_path
        assert path.read_bytes() == expert.source_contents[descriptor.relative_path]
        assert _mode(path) == (0o555 if descriptor.mode == "100755" else 0o444)

    adapter = invocation.materialized_case.task_adapter
    for descriptor in adapter.source_extraction_receipt.source_tree_files:
        path = layout.adapter_root / descriptor.relative_path
        assert path.read_bytes() == adapter.source_contents[descriptor.relative_path]
        assert _mode(path) == (0o555 if descriptor.mode == "100755" else 0o444)

    for artifact in invocation.materialized_case.task_context.starting_artifacts:
        for descriptor in artifact.artifact.source_files:
            path = (
                layout.task_root
                / artifact.artifact.mount_path
                / descriptor.relative_path
            )
            assert (
                path.read_bytes() == artifact.source_contents[descriptor.relative_path]
            )
            assert _mode(path) == (0o555 if descriptor.mode == "100755" else 0o444)

    assert layout.request_path.read_bytes() == (
        invocation.task_evaluator_request.to_json_bytes()
    )
    assert _mode(layout.request_path) == 0o444
    assert _mode(layout.workspace_root) == 0o700
    assert tuple(path.name for path in layout.workspace_root.iterdir()) == ("input",)
    for directory, subdirectories, _filenames in os.walk(layout.input_root):
        assert _mode(directory) == 0o555
        for subdirectory in subdirectories:
            assert _mode(os.path.join(directory, subdirectory)) == 0o555


def test_result_snapshot_accepts_only_the_exact_busybox_result_closure():
    owner_id = os.geteuid()
    snapshot = _valid_result_snapshot(owner_id)

    assert _parse(snapshot, owner_id) == _RESULT_PAYLOAD


@pytest.mark.parametrize("type_flag", (b"1", b"2", b"3", b"4", b"5", b"6", b"7"))
def test_result_snapshot_rejects_links_directories_and_special_results(type_flag):
    owner_id = os.geteuid()
    snapshot = _tar_snapshot(
        (
            ("./", owner_id, b"5", b"", b""),
            (
                "./result.json",
                owner_id,
                type_flag,
                b"",
                b"target" if type_flag in {b"1", b"2"} else b"",
            ),
        )
    )

    with pytest.raises(
        SourceReplayProviderFilesystemError, match="link or special|regular"
    ):
        _parse(snapshot, owner_id)


def test_result_snapshot_rejects_extra_and_duplicate_entries():
    owner_id = os.geteuid()
    extra = _tar_snapshot(
        (
            ("./", owner_id, b"5", b"", b""),
            ("./result.json", owner_id, b"0", _RESULT_PAYLOAD, b""),
            ("./trace", owner_id, b"0", b"secret", b""),
        )
    )
    duplicate = _tar_snapshot(
        (
            ("./", owner_id, b"5", b"", b""),
            ("./result.json", owner_id, b"0", _RESULT_PAYLOAD, b""),
            ("./result.json", owner_id, b"0", _RESULT_PAYLOAD, b""),
        )
    )

    with pytest.raises(SourceReplayProviderFilesystemError, match="extra"):
        _parse(extra, owner_id)
    with pytest.raises(SourceReplayProviderFilesystemError, match="duplicate"):
        _parse(duplicate, owner_id)


def test_result_snapshot_rejects_missing_root_or_result_entries():
    owner_id = os.geteuid()
    missing_root = _tar_snapshot(
        (("./result.json", owner_id, b"0", _RESULT_PAYLOAD, b""),)
    )
    missing_result = _tar_snapshot((("./", owner_id, b"5", b"", b""),))

    with pytest.raises(
        SourceReplayProviderFilesystemError, match="exact result closure"
    ):
        _parse(missing_root, owner_id)
    with pytest.raises(
        SourceReplayProviderFilesystemError, match="exact result closure"
    ):
        _parse(missing_result, owner_id)


def test_result_snapshot_rejects_owner_and_size_bound_mismatches():
    owner_id = os.geteuid()
    snapshot = _valid_result_snapshot(owner_id)

    with pytest.raises(SourceReplayProviderFilesystemError, match="owner"):
        _parse(snapshot, owner_id + 1)
    with pytest.raises(SourceReplayProviderFilesystemError, match="group"):
        parse_source_replay_result_snapshot(
            snapshot,
            expected_owner_id=owner_id,
            expected_group_id=owner_id + 1,
            maximum_result_bytes=len(_RESULT_PAYLOAD),
            maximum_snapshot_bytes=len(snapshot),
        )
    with pytest.raises(SourceReplayProviderFilesystemError, match="result exceeds"):
        parse_source_replay_result_snapshot(
            snapshot,
            expected_owner_id=owner_id,
            expected_group_id=owner_id,
            maximum_result_bytes=len(_RESULT_PAYLOAD) - 1,
            maximum_snapshot_bytes=len(snapshot),
        )
    with pytest.raises(SourceReplayProviderFilesystemError, match="snapshot exceeds"):
        parse_source_replay_result_snapshot(
            snapshot,
            expected_owner_id=owner_id,
            expected_group_id=owner_id,
            maximum_result_bytes=len(_RESULT_PAYLOAD),
            maximum_snapshot_bytes=len(snapshot) - 1,
        )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda snapshot: snapshot + bytes(_TAR_BLOCK_SIZE),
        lambda snapshot: snapshot[:-_TAR_BLOCK_SIZE],
        lambda snapshot: snapshot[:-1],
        lambda snapshot: snapshot[:148] + b"000000\x00 " + snapshot[156:],
    ),
)
def test_result_snapshot_rejects_trailing_truncated_and_noncanonical_archives(mutate):
    owner_id = os.geteuid()
    snapshot = mutate(_valid_result_snapshot(owner_id))

    with pytest.raises(SourceReplayProviderFilesystemError):
        _parse(snapshot, owner_id)


def test_result_snapshot_rejects_nonzero_padding_and_another_tar_dialect():
    owner_id = os.geteuid()
    snapshot = bytearray(_valid_result_snapshot(owner_id))
    result_data_offset = _TAR_BLOCK_SIZE * 2
    snapshot[result_data_offset + len(_RESULT_PAYLOAD)] = 1
    another_dialect = bytearray(_valid_result_snapshot(owner_id))
    another_dialect[257:265] = b"ustar\x0000"

    with pytest.raises(SourceReplayProviderFilesystemError, match="padding"):
        _parse(bytes(snapshot), owner_id)
    with pytest.raises(SourceReplayProviderFilesystemError, match="pinned BusyBox"):
        _parse(bytes(another_dialect), owner_id)
