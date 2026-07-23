import os
import socket
import stat
from pathlib import Path

import pytest

import kapso.cross_run.expert.workspace as expert_workspace_module
from kapso.core.config import load_config
from kapso.cross_run.canonical import source_tree_digest, tree_or_blob_digest
from kapso.cross_run.contracts import SourceFileDescriptor
from kapso.cross_run.expert import (
    EMPTY_EXPERT_TREE_DIGEST,
    ExpertCandidateWorkspaceError,
    ExpertCandidateWorkspaceManager,
    ExpertSourceBaseTreeReceipt,
    ExpertTriggerEvidencePacket,
    compile_expert_semantic_book,
)
from kapso.cross_run.expert.book import (
    EXPERT_BOOK_PATH,
    EXPERT_REPOSITORY_MAP_PATH,
    expert_control_paths,
    expert_module_contract_path,
)
from kapso.cross_run.github.materializer import (
    MaterializedArtifact,
    SourceArchiveExtractionReceipt,
)
from kapso.cross_run.settings import CrossRunSettings
from test_expert_triggers import trigger_packet, trigger_settings

CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"


class FixtureSourceMaterializer:
    def __init__(self, contents):
        self.contents = contents
        self.calls = []

    def extract_verified_source_archive(
        self,
        *,
        materialized,
        expected,
        destination,
        destination_parent_descriptor,
    ):
        named_parent = destination.parent.stat(follow_symlinks=False)
        opened_parent = os.fstat(destination_parent_descriptor)
        if (named_parent.st_dev, named_parent.st_ino) != (
            opened_parent.st_dev,
            opened_parent.st_ino,
        ):
            raise ExpertCandidateWorkspaceError(
                "fixture extraction source base differs from its pinned descriptor"
            )
        self.calls.append(
            (
                materialized,
                expected,
                destination,
                destination_parent_descriptor,
            )
        )
        pinned_destination = (
            Path("/proc/self/fd")
            / str(destination_parent_descriptor)
            / destination.name
        )
        pinned_destination.mkdir(mode=0o700)
        expected_by_path = {
            descriptor.relative_path: descriptor
            for descriptor in expected.source_tree_files
        }
        for relative_path, payload in self.contents.items():
            output = pinned_destination / relative_path
            output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            output.write_bytes(payload)
            output.chmod(
                0o755 if expected_by_path[relative_path].mode == "100755" else 0o644
            )
        return expected


class MismatchedReceiptSourceMaterializer(FixtureSourceMaterializer):
    def extract_verified_source_archive(
        self,
        *,
        materialized,
        expected,
        destination,
        destination_parent_descriptor,
    ):
        super().extract_verified_source_archive(
            materialized=materialized,
            expected=expected,
            destination=destination,
            destination_parent_descriptor=destination_parent_descriptor,
        )
        return SourceArchiveExtractionReceipt.mint(
            artifact_id=expected.artifact_id,
            source_archive_ref=expected.source_archive_ref,
            source_archive_digest=expected.source_archive_digest,
            source_tree_hash=expected.source_tree_hash,
            source_tree_files=expected.source_tree_files,
            extractor_version=f"{expected.extractor_version}-mismatch",
        )


class FailingAfterExtractionSourceMaterializer(FixtureSourceMaterializer):
    def extract_verified_source_archive(
        self,
        *,
        materialized,
        expected,
        destination,
        destination_parent_descriptor,
    ):
        super().extract_verified_source_archive(
            materialized=materialized,
            expected=expected,
            destination=destination,
            destination_parent_descriptor=destination_parent_descriptor,
        )
        raise RuntimeError("injected extraction failure")


def expert_settings():
    return CrossRunSettings.from_dict(
        load_config(CANONICAL_CONFIG_PATH)["cross_run"]
    ).expert


def descriptors(contents):
    return tuple(
        SourceFileDescriptor(
            relative_path=relative_path,
            digest=tree_or_blob_digest(contents[relative_path]),
            mode="100644",
            size=len(contents[relative_path]),
        )
        for relative_path in sorted(contents)
    )


def tree_hash(source_files):
    return source_tree_digest(
        {
            source_file.relative_path: (
                source_file.digest,
                source_file.mode,
                source_file.size,
            )
            for source_file in source_files
        }
    )


def released_workspace_fixture(*, manual_book=False):
    packet = trigger_packet(settings=trigger_settings())
    repository_map = packet.source_base_repository_map
    module = packet.source_base_module_contracts[0]
    generated_book = compile_expert_semantic_book(
        packet.source_base_scope_contract,
        repository_map,
        packet.source_base_module_contracts,
    )
    contents = {
        "src/reproducible_execution/__init__.py": b"def execute(task):\n    return task.run()\n",
        "tests/replay_resume.py": b"def replay():\n    return True\n",
        "tests/test_resume.py": b"def test_resume():\n    assert True\n",
        EXPERT_BOOK_PATH: b"# manual\n" if manual_book else generated_book,
        EXPERT_REPOSITORY_MAP_PATH: repository_map.to_json_bytes(),
        expert_module_contract_path(module.module_contract_id): module.to_json_bytes(),
    }
    source_files = descriptors(contents)
    parent_hash = tree_hash(source_files)
    extraction = SourceArchiveExtractionReceipt.mint(
        artifact_id=packet.source_base_release.release_id,
        source_archive_ref=packet.source_base_release.source_archive_ref,
        source_archive_digest=packet.source_base_release.checksums[
            packet.source_base_release.source_archive_ref
        ],
        source_tree_hash=parent_hash,
        source_tree_files=source_files,
        extractor_version=(
            packet.source_base_tree_receipt.source_extraction_receipt.extractor_version
        ),
    )
    source_base_receipt = ExpertSourceBaseTreeReceipt.mint(
        release_id=packet.source_base_release.release_id,
        cache_verification_receipt=(
            packet.source_base_tree_receipt.cache_verification_receipt
        ),
        source_extraction_receipt=extraction,
        source_base_tree_hash=parent_hash,
        repository_map_id=repository_map.repository_map_id,
        module_contract_ids=tuple(
            module.module_contract_id for module in packet.source_base_module_contracts
        ),
        materializer_version=packet.source_base_tree_receipt.materializer_version,
    )
    released_packet = ExpertTriggerEvidencePacket.mint(
        knowledge_snapshot_manifest=packet.knowledge_snapshot_manifest,
        knowledge_record_closure_digest=packet.knowledge_record_closure_digest,
        configuration_fingerprint=packet.configuration_fingerprint,
        scope_contract=packet.scope_contract,
        source_base_scope_contract=packet.source_base_scope_contract,
        source_base_release=packet.source_base_release,
        source_base_tree_receipt=source_base_receipt,
        source_base_tree_hash=parent_hash,
        source_base_repository_map=packet.source_base_repository_map,
        source_base_module_contracts=packet.source_base_module_contracts,
        episodes=packet.episodes,
        claims=packet.claims,
        trigger_observations=packet.trigger_observations,
        active_task_bindings=packet.active_task_bindings,
        proof_reference_ids=packet.proof_reference_ids,
        recovery_barrier_basis_packet_id=None,
    )
    materialized = MaterializedArtifact(
        root=Path("/unused/materialized"),
        content=Path("/unused/materialized/content"),
        assets=Path("/unused/materialized/assets"),
        receipt=source_base_receipt.cache_verification_receipt,
        reused=True,
    )
    return released_packet, materialized, contents


def workspace_manager(tmp_path, materializer):
    tmp_path.chmod(0o700)
    return ExpertCandidateWorkspaceManager(
        (tmp_path / "workspaces").resolve(),
        tmp_path.resolve(),
        expert_settings(),
        materializer,
    )


def test_bootstrap_workspace_is_exact_empty_and_removed_after_lease(tmp_path):
    packet = trigger_packet(settings=trigger_settings(), bootstrap=True)
    materializer = FixtureSourceMaterializer({})
    manager = workspace_manager(tmp_path, materializer)

    with manager.lease(
        trigger_packet=packet,
        materialized_source_base=None,
    ) as prepared:
        workspace_path = prepared.path
        assert prepared.source_base_tree_hash == EMPTY_EXPERT_TREE_DIGEST
        assert prepared.source_base_files == ()
        assert prepared.editable_snapshot.tree_hash == EMPTY_EXPERT_TREE_DIGEST
        assert tuple(workspace_path.iterdir()) == ()

    assert not workspace_path.exists()
    assert materializer.calls == []


def test_concurrent_workspace_leases_have_distinct_private_roots(tmp_path):
    packet = trigger_packet(settings=trigger_settings(), bootstrap=True)
    manager = workspace_manager(tmp_path, FixtureSourceMaterializer({}))
    first_lease = manager.lease(
        trigger_packet=packet,
        materialized_source_base=None,
    )
    second_lease = manager.lease(
        trigger_packet=packet,
        materialized_source_base=None,
    )

    with first_lease as first, second_lease as second:
        assert first.path != second.path
        assert first.path.parent == second.path.parent == manager.root
        assert first.path.stat().st_mode & 0o077 == 0
        assert second.path.stat().st_mode & 0o077 == 0

    assert tuple(manager.root.iterdir()) == ()


def test_workspace_authority_descriptor_exists_only_during_active_lease(tmp_path):
    packet = trigger_packet(settings=trigger_settings(), bootstrap=True)
    manager = workspace_manager(tmp_path, FixtureSourceMaterializer({}))
    lease = manager.lease(
        trigger_packet=packet,
        materialized_source_base=None,
    )

    with pytest.raises(ExpertCandidateWorkspaceError, match="active lease"):
        _ = lease.workspace_authority_descriptor
    with lease as prepared:
        descriptor_metadata = os.fstat(lease.workspace_authority_descriptor)
        path_metadata = prepared.path.stat(follow_symlinks=False)
        assert (descriptor_metadata.st_dev, descriptor_metadata.st_ino) == (
            path_metadata.st_dev,
            path_metadata.st_ino,
        )
    with pytest.raises(ExpertCandidateWorkspaceError, match="active lease"):
        _ = lease.workspace_authority_descriptor


def test_released_workspace_validates_controls_then_exposes_editable_parent(tmp_path):
    packet, materialized, contents = released_workspace_fixture()
    materializer = FixtureSourceMaterializer(contents)
    manager = workspace_manager(tmp_path, materializer)

    with manager.lease(
        trigger_packet=packet,
        materialized_source_base=materialized,
    ) as prepared:
        workspace_path = prepared.path
        editable_paths = tuple(
            file.descriptor.relative_path for file in prepared.editable_snapshot.files
        )
        assert prepared.source_base_tree_hash == packet.source_base_tree_hash
        assert prepared.source_base_files == (
            packet.source_base_tree_receipt.source_extraction_receipt.source_tree_files
        )
        assert editable_paths == (
            "src/reproducible_execution/__init__.py",
            "tests/replay_resume.py",
            "tests/test_resume.py",
        )
        assert not any(
            (workspace_path / relative_path).exists()
            for relative_path in expert_control_paths(
                packet.source_base_module_contracts
            )
        )

    assert not workspace_path.exists()
    assert materializer.calls[0][1] == (
        packet.source_base_tree_receipt.source_extraction_receipt
    )


def test_invalid_released_controls_fail_before_lease_and_leave_no_workspace(tmp_path):
    packet, materialized, contents = released_workspace_fixture(manual_book=True)
    manager = workspace_manager(tmp_path, FixtureSourceMaterializer(contents))

    with pytest.raises(
        ExpertCandidateWorkspaceError,
        match="control bytes differ",
    ):
        manager.lease(
            trigger_packet=packet,
            materialized_source_base=materialized,
        )

    assert tuple(manager.root.iterdir()) == ()


@pytest.mark.parametrize(
    ("materializer_type", "error_type", "message"),
    (
        (
            MismatchedReceiptSourceMaterializer,
            ExpertCandidateWorkspaceError,
            "trigger receipt",
        ),
        (
            FailingAfterExtractionSourceMaterializer,
            RuntimeError,
            "injected extraction failure",
        ),
    ),
)
def test_released_workspace_construction_failures_remove_created_destination(
    tmp_path,
    materializer_type,
    error_type,
    message,
):
    packet, materialized, contents = released_workspace_fixture()
    manager = workspace_manager(tmp_path, materializer_type(contents))

    with pytest.raises(error_type, match=message):
        manager.lease(
            trigger_packet=packet,
            materialized_source_base=materialized,
        )

    assert tuple(manager.root.iterdir()) == ()


def test_generated_control_removal_rejects_raced_symlink_without_following_it(
    tmp_path,
    monkeypatch,
):
    packet, materialized, contents = released_workspace_fixture()
    materializer = FixtureSourceMaterializer(contents)
    manager = workspace_manager(tmp_path, materializer)
    outside = tmp_path / "outside-controls"
    outside.mkdir(mode=0o755)
    outside.chmod(0o755)
    sentinel = outside / "sentinel.txt"
    sentinel.write_bytes(b"preserve me")
    original_open = expert_workspace_module._open_real_directory_at

    def swap_control_root(
        parent_descriptor,
        name,
        descriptors_stack,
        purpose,
    ):
        if name == ".kapso":
            workspace_path = materializer.calls[-1][2]
            control_root = workspace_path / ".kapso"
            control_root.rename(workspace_path / ".kapso-original")
            control_root.symlink_to(outside, target_is_directory=True)
        return original_open(
            parent_descriptor,
            name,
            descriptors_stack,
            purpose,
        )

    monkeypatch.setattr(
        expert_workspace_module,
        "_open_real_directory_at",
        swap_control_root,
    )

    with pytest.raises(ExpertCandidateWorkspaceError, match="real directory"):
        manager.lease(
            trigger_packet=packet,
            materialized_source_base=materialized,
        )

    assert tuple(manager.root.iterdir()) == ()
    assert sentinel.read_bytes() == b"preserve me"
    assert stat.S_IMODE(outside.stat().st_mode) == 0o755


def test_released_workspace_rejects_source_mutation_during_control_removal(
    tmp_path,
    monkeypatch,
):
    packet, materialized, contents = released_workspace_fixture()
    materializer = FixtureSourceMaterializer(contents)
    manager = workspace_manager(tmp_path, materializer)
    original_remove = ExpertCandidateWorkspaceManager._remove_generated_controls

    def mutate_source(workspace_descriptor, trigger_packet):
        workspace_path = materializer.calls[-1][2]
        source = workspace_path / "src/reproducible_execution/__init__.py"
        source.write_bytes(b"def execute(task):\n    return 'mutated'\n")
        original_remove(workspace_descriptor, trigger_packet)

    monkeypatch.setattr(
        ExpertCandidateWorkspaceManager,
        "_remove_generated_controls",
        staticmethod(mutate_source),
    )

    with pytest.raises(
        ExpertCandidateWorkspaceError,
        match="outside generated controls",
    ):
        manager.lease(
            trigger_packet=packet,
            materialized_source_base=materialized,
        )

    assert tuple(manager.root.iterdir()) == ()


def test_workspace_manager_rejects_root_replacement_during_preparation(
    tmp_path,
    monkeypatch,
):
    packet = trigger_packet(settings=trigger_settings(), bootstrap=True)
    manager = workspace_manager(tmp_path, FixtureSourceMaterializer({}))
    moved_root = tmp_path / "moved-workspaces"
    original_inspect = expert_workspace_module.inspect_coding_agent_workspace_descriptor
    swapped = False

    def replace_root(workspace_descriptor, *, maximum_entries, maximum_bytes):
        nonlocal swapped
        if not swapped:
            swapped = True
            manager.root.rename(moved_root)
            manager.root.mkdir(mode=0o700)
            workspace_name = tuple(moved_root.iterdir())[0].name
            (manager.root / workspace_name).mkdir(mode=0o700)
        return original_inspect(
            workspace_descriptor,
            maximum_entries=maximum_entries,
            maximum_bytes=maximum_bytes,
        )

    monkeypatch.setattr(
        expert_workspace_module,
        "inspect_coding_agent_workspace_descriptor",
        replace_root,
    )

    with pytest.raises(ExpertCandidateWorkspaceError, match="root identity changed"):
        manager.lease(
            trigger_packet=packet,
            materialized_source_base=None,
        )

    assert tuple(moved_root.iterdir()) == ()
    replacement_workspace = tuple(manager.root.iterdir())
    assert len(replacement_workspace) == 1
    replacement_workspace[0].rmdir()
    manager.root.rmdir()
    moved_root.rmdir()


def test_released_extraction_rejects_root_replacement_before_writing(
    tmp_path,
    monkeypatch,
):
    packet, materialized, contents = released_workspace_fixture()
    source_materializer = FixtureSourceMaterializer(contents)
    manager = workspace_manager(tmp_path, source_materializer)
    moved_root = tmp_path / "moved-workspaces"
    original_extract = source_materializer.extract_verified_source_archive

    def replace_root_before_extraction(**arguments):
        manager.root.rename(moved_root)
        manager.root.mkdir(mode=0o700)
        return original_extract(**arguments)

    monkeypatch.setattr(
        source_materializer,
        "extract_verified_source_archive",
        replace_root_before_extraction,
    )

    with pytest.raises(ExpertCandidateWorkspaceError, match="pinned descriptor"):
        manager.lease(
            trigger_packet=packet,
            materialized_source_base=materialized,
        )

    assert tuple(moved_root.iterdir()) == ()
    assert tuple(manager.root.iterdir()) == ()
    manager.root.rmdir()
    moved_root.rmdir()


def test_workspace_cleanup_unlinks_hostile_entries_without_following_them(tmp_path):
    packet = trigger_packet(settings=trigger_settings(), bootstrap=True)
    manager = workspace_manager(tmp_path, FixtureSourceMaterializer({}))
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"preserve me")

    with manager.lease(
        trigger_packet=packet,
        materialized_source_base=None,
    ) as prepared:
        workspace_path = prepared.path
        (workspace_path / "linked").symlink_to(outside)
        os.link(outside, workspace_path / "hardlinked")
        os.mkfifo(workspace_path / "pipe")
        unix_socket = socket.socket(socket.AF_UNIX)
        workspace_descriptor = os.open(
            workspace_path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        unix_socket.bind(f"/proc/self/fd/{workspace_descriptor}/socket")
        os.close(workspace_descriptor)
        unix_socket.close()
        nested = workspace_path / "nested"
        nested.mkdir()
        (nested / "payload").write_bytes(b"payload")
        nested.chmod(0o000)
        workspace_path.chmod(0o000)

    assert not workspace_path.exists()
    assert outside.read_bytes() == b"preserve me"
    assert outside.stat().st_nlink == 1


def test_workspace_cleanup_rejects_root_replacement_without_deleting_it(tmp_path):
    packet = trigger_packet(settings=trigger_settings(), bootstrap=True)
    manager = workspace_manager(tmp_path, FixtureSourceMaterializer({}))
    lease = manager.lease(
        trigger_packet=packet,
        materialized_source_base=None,
    )
    prepared = lease.__enter__()
    moved = prepared.path.with_name("moved-original")
    prepared.path.rename(moved)
    prepared.path.mkdir(mode=0o700)

    with pytest.raises(ExpertCandidateWorkspaceError, match="identity changed"):
        lease.close()

    assert prepared.path.is_dir()
    prepared.path.rmdir()
    moved.rmdir()


def test_workspace_cleanup_does_not_chmod_a_raced_directory_symlink(
    tmp_path,
    monkeypatch,
):
    packet = trigger_packet(settings=trigger_settings(), bootstrap=True)
    manager = workspace_manager(tmp_path, FixtureSourceMaterializer({}))
    outside = tmp_path / "outside-directory"
    outside.mkdir(mode=0o755)
    outside.chmod(0o755)
    lease = manager.lease(
        trigger_packet=packet,
        materialized_source_base=None,
    )
    prepared = lease.__enter__()
    nested = prepared.path / "nested"
    nested.mkdir()
    (nested / "payload").write_bytes(b"payload")
    moved = prepared.path / "moved-nested"
    original_open = expert_workspace_module._open_cleanup_directory_at

    def swap_nested_directory(
        parent_descriptor,
        name,
        expected_identity,
        descriptors_stack,
    ):
        if name == "nested":
            nested.rename(moved)
            nested.symlink_to(outside, target_is_directory=True)
        return original_open(
            parent_descriptor,
            name,
            expected_identity,
            descriptors_stack,
        )

    monkeypatch.setattr(
        expert_workspace_module,
        "_open_cleanup_directory_at",
        swap_nested_directory,
    )

    with pytest.raises(OSError):
        lease.close()

    assert stat.S_IMODE(outside.stat().st_mode) == 0o755
    nested.unlink()
    (moved / "payload").unlink()
    moved.rmdir()
    prepared.path.rmdir()


def test_active_lease_rejects_root_substitution_before_successful_exit(tmp_path):
    packet = trigger_packet(settings=trigger_settings(), bootstrap=True)
    manager = workspace_manager(tmp_path, FixtureSourceMaterializer({}))
    moved_root = tmp_path / "moved-workspaces"
    lease = manager.lease(
        trigger_packet=packet,
        materialized_source_base=None,
    )

    with pytest.raises(ExpertCandidateWorkspaceError, match="root identity changed"):
        with lease as prepared:
            manager.root.rename(moved_root)
            manager.root.mkdir(mode=0o700)
            prepared.path.mkdir(mode=0o700)
            (prepared.path / "substituted.py").write_bytes(b"candidate")

    assert tuple(moved_root.iterdir()) == ()
    assert (lease.prepared.path / "substituted.py").read_bytes() == b"candidate"
    (lease.prepared.path / "substituted.py").unlink()
    lease.prepared.path.rmdir()
    manager.root.rmdir()
    moved_root.rmdir()


def test_failed_lease_entry_closes_descriptors_and_removes_original_workspace(
    tmp_path,
):
    packet = trigger_packet(settings=trigger_settings(), bootstrap=True)
    manager = workspace_manager(tmp_path, FixtureSourceMaterializer({}))
    moved_root = tmp_path / "moved-workspaces"
    lease = manager.lease(
        trigger_packet=packet,
        materialized_source_base=None,
    )
    manager.root.rename(moved_root)
    manager.root.mkdir(mode=0o700)
    lease.prepared.path.mkdir(mode=0o700)

    with pytest.raises(ExpertCandidateWorkspaceError, match="root identity changed"):
        with lease:
            raise AssertionError("unreachable")

    for descriptor in (
        lease._state_descriptor,
        lease._root_descriptor,
        lease._workspace_descriptor,
    ):
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert tuple(moved_root.iterdir()) == ()
    lease.prepared.path.rmdir()
    manager.root.rmdir()
    moved_root.rmdir()
