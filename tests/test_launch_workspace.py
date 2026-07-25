import os
import shutil
import stat
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace

import pytest

import kapso.cross_run.launch.workspace as workspace_module
from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.contracts import ContractValidationError
from kapso.cross_run.git_command import BoundedGitCommand
from kapso.cross_run.github.command import CommandOutputKind
from kapso.cross_run.launch.contracts import (
    BootstrapPin,
    LaunchContractError,
    LaunchManifest,
    WorkspaceInstallationReceipt,
)
from kapso.cross_run.launch.resolver import LaunchResolutionError
from kapso.cross_run.launch.workspace import (
    LaunchWorkspaceError,
    PreparedLaunchWorkspace,
    StarterWorkspaceBuilder,
)
from test_launch_resolver import resolver_case


def _build(resolver_case, run_root, *, run_id="run-1"):
    resolved = resolver_case["resolver"].resolve(resolver_case["request"])
    builder = StarterWorkspaceBuilder(resolver_case["resolver"]._settings)
    return builder.build(
        resolved,
        run_root,
        run_id=run_id,
        campaign_id="campaign-1",
    )


def test_builder_atomically_publishes_self_contained_launch(resolver_case, tmp_path):
    prepared = _build(resolver_case, (tmp_path / "run").absolute())
    pin = prepared.bootstrap_pin
    receipt = pin.installation_receipt
    settings = resolver_case["resolver"]._settings
    git = BoundedGitCommand(
        timeout_seconds=settings.github.command_timeout_seconds,
        maximum_output_bytes=settings.capture.git_command_output_bytes,
    )

    assert prepared.run_root.is_dir()
    assert prepared.workspace.is_dir()
    assert prepared.knowledge_snapshot.is_dir()
    assert prepared.task_adapter.is_dir()
    assert BootstrapPin.from_json_bytes(prepared.bootstrap_pin_path.read_bytes()) == pin
    assert (
        LaunchManifest.from_json_bytes(prepared.launch_manifest_path.read_bytes())
        == pin.launch_manifest
    )
    assert tree_or_blob_digest(prepared.launch_manifest_path.read_bytes()) == (
        pin.launch_manifest_full_digest
    )
    assert receipt.layout.workspace_relative_path == "workspace"
    checkpoint_journal = (
        prepared.run_root / receipt.layout.run_checkpoint_journal_relative_path
    )
    checkpoint_lock = (
        prepared.run_root / receipt.layout.run_checkpoint_lock_relative_path
    )
    derived_state_store = (
        prepared.run_root / receipt.layout.run_derived_state_store_relative_path
    )
    derived_state_staging = (
        prepared.run_root / receipt.layout.run_derived_state_staging_relative_path
    )
    action_workspace_staging = (
        prepared.run_root / receipt.layout.run_action_workspace_staging_relative_path
    )
    assert (
        checkpoint_journal.stat().st_dev,
        checkpoint_journal.stat().st_ino,
    ) == (
        receipt.run_checkpoint_journal_device,
        receipt.run_checkpoint_journal_inode,
    )
    assert (
        checkpoint_lock.stat().st_dev,
        checkpoint_lock.stat().st_ino,
    ) == (
        receipt.run_checkpoint_lock_device,
        receipt.run_checkpoint_lock_inode,
    )
    for directory in (derived_state_store, derived_state_staging):
        assert directory.is_dir()
        assert stat.S_IMODE(directory.stat(follow_symlinks=False).st_mode) == 0o700
        assert tuple(directory.iterdir()) == ()
    assert (
        action_workspace_staging.stat(follow_symlinks=False).st_dev,
        action_workspace_staging.stat(follow_symlinks=False).st_ino,
    ) == (
        receipt.run_action_workspace_staging_device,
        receipt.run_action_workspace_staging_inode,
    )
    assert (
        stat.S_IMODE(action_workspace_staging.stat(follow_symlinks=False).st_mode)
        == 0o700
    )
    assert tuple(action_workspace_staging.iterdir()) == ()
    for projection_path in (
        receipt.layout.run_idea_archive_relative_path,
        receipt.layout.run_experiment_history_relative_path,
        receipt.layout.run_execution_journal_relative_path,
    ):
        assert not (prepared.run_root / projection_path).exists()
    assert not (prepared.workspace / ".kapso" / "bootstrap_pin.json").exists()
    assert not (prepared.workspace / ".gitmodules").exists()
    assert not (prepared.workspace / ".git" / "hooks").exists()
    assert not (
        prepared.workspace / ".git" / "objects" / "info" / "alternates"
    ).exists()

    status = git.run(
        prepared.workspace,
        (
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ),
        output_kind=CommandOutputKind.TEXT,
    )
    assert status.output == ""
    assert (
        git.run(
            prepared.workspace,
            ("rev-parse", "HEAD"),
            output_kind=CommandOutputKind.TEXT,
        ).output.strip()
        == receipt.workspace_baseline_commit_sha
    )
    assert (
        git.run(
            prepared.workspace,
            ("rev-parse", "HEAD^{tree}"),
            output_kind=CommandOutputKind.TEXT,
        ).output.strip()
        == receipt.workspace_baseline_tree_sha
    )
    git.run(
        prepared.workspace,
        ("fsck", "--strict", "--no-dangling"),
        output_kind=CommandOutputKind.TEXT,
    )


def test_builder_materializes_private_read_only_copies(resolver_case, tmp_path):
    resolved = resolver_case["resolver"].resolve(resolver_case["request"])
    source_knowledge_path = resolved.knowledge_artifact.content
    builder = StarterWorkspaceBuilder(resolver_case["resolver"]._settings)
    prepared = builder.build(
        resolved,
        (tmp_path / "run").absolute(),
        run_id="run-1",
        campaign_id="campaign-1",
    )

    immutable_roots = (
        prepared.knowledge_snapshot,
        prepared.task_adapter,
        *prepared.starting_artifacts.values(),
    )
    for root in immutable_roots:
        assert stat.S_IMODE(root.stat(follow_symlinks=False).st_mode) == 0o555
        for path in root.rglob("*"):
            assert not path.is_symlink()
            mode = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
            assert mode in ({0o555} if path.is_dir() else {0o444, 0o555})
    source_manifest = source_knowledge_path / "snapshot.json"
    installed_manifest = prepared.knowledge_snapshot / "snapshot.json"
    assert source_manifest.read_bytes() == installed_manifest.read_bytes()
    assert source_manifest.stat().st_ino != installed_manifest.stat().st_ino
    assert not tuple(prepared.task_adapter.glob("release_matrix_assets"))
    assert str(tmp_path).encode("utf-8") not in prepared.bootstrap_pin.to_json_bytes()


def test_builder_is_deterministic_across_run_roots(resolver_case, tmp_path):
    first = _build(
        resolver_case,
        (tmp_path / "run-one").absolute(),
        run_id="run-one",
    )
    second = _build(
        resolver_case,
        (tmp_path / "run-two").absolute(),
        run_id="run-two",
    )

    assert (
        first.bootstrap_pin.installation_receipt.workspace_baseline_tree_sha
        == second.bootstrap_pin.installation_receipt.workspace_baseline_tree_sha
    )
    assert (
        first.bootstrap_pin.installation_receipt.workspace_baseline_commit_sha
        == second.bootstrap_pin.installation_receipt.workspace_baseline_commit_sha
    )
    assert (
        first.bootstrap_pin.installation_receipt.workspace_git_index_digest
        == second.bootstrap_pin.installation_receipt.workspace_git_index_digest
    )
    assert (first.workspace / ".git" / "index").read_bytes() == (
        second.workspace / ".git" / "index"
    ).read_bytes()


def test_builder_accepts_valid_nested_shared_layout_ancestors(
    resolver_case,
    tmp_path,
):
    settings = resolver_case["resolver"]._settings
    launch = replace(
        settings.launch,
        workspace_path="materialized/workspace",
        immutable_root_path="materialized/readonly",
        knowledge_snapshot_path="materialized/readonly/knowledge",
        task_adapter_path="materialized/readonly/task_adapter",
        starting_artifacts_path="materialized/readonly/starting_artifacts",
    )
    builder = StarterWorkspaceBuilder(replace(settings, launch=launch))
    resolved = resolver_case["resolver"].resolve(resolver_case["request"])

    prepared = builder.build(
        resolved,
        (tmp_path / "run").absolute(),
        run_id="nested-run",
        campaign_id="campaign-1",
    )

    assert prepared.workspace == prepared.run_root / launch.workspace_path
    assert prepared.knowledge_snapshot == (
        prepared.run_root / launch.knowledge_snapshot_path
    )
    active = prepared.activate()
    materialized = prepared.run_root / "materialized"
    replacement = tmp_path / "replacement-materialized"
    shutil.copytree(materialized, replacement)
    materialized.rename(tmp_path / "retired-materialized")
    replacement.rename(materialized)

    with pytest.raises(LaunchWorkspaceError, match="directories changed"):
        active.require_control_authority()
    active.close()


def test_nested_layout_reopens_after_workspace_leaf_replacement(
    resolver_case,
    tmp_path,
):
    settings = resolver_case["resolver"]._settings
    launch = replace(
        settings.launch,
        workspace_path="materialized/workspace",
        immutable_root_path="materialized/readonly",
        knowledge_snapshot_path="materialized/readonly/knowledge",
        task_adapter_path="materialized/readonly/task_adapter",
        starting_artifacts_path="materialized/readonly/starting_artifacts",
    )
    nested_settings = replace(settings, launch=launch)
    resolved = resolver_case["resolver"].resolve(resolver_case["request"])
    prepared = StarterWorkspaceBuilder(nested_settings).build(
        resolved,
        (tmp_path / "run").absolute(),
        run_id="nested-run",
        campaign_id="campaign-1",
    )
    active = prepared.activate()
    replacement = tmp_path / "replacement-workspace"
    retired = tmp_path / "retired-workspace"
    shutil.copytree(active.workspace, replacement)
    active.workspace.rename(retired)
    replacement.rename(active.workspace)

    with ExitStack() as descriptors:
        descriptor, identity = active._open_execution_workspace(descriptors)
        active._require_execution_workspace(descriptor, identity)
    active.close()

    resumed = StarterWorkspaceBuilder(nested_settings).reopen(prepared.run_root)
    with ExitStack() as descriptors:
        descriptor, identity = resumed._open_execution_workspace(descriptors)
        resumed._require_execution_workspace(descriptor, identity)
    resumed.close()


def test_builder_consumes_resolver_authority_once_under_concurrency(
    resolver_case,
    tmp_path,
):
    resolved = resolver_case["resolver"].resolve(resolver_case["request"])
    builder = StarterWorkspaceBuilder(resolver_case["resolver"]._settings)

    def build(position):
        return builder.build(
            resolved,
            (tmp_path / f"run-{position}").absolute(),
            run_id=f"run-{position}",
            campaign_id="campaign-1",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = tuple(pool.submit(build, position) for position in range(2))
    outcomes = tuple(
        future.exception() if future.exception() is not None else future.result()
        for future in futures
    )

    assert sum(not isinstance(outcome, Exception) for outcome in outcomes) == 1
    assert sum(isinstance(outcome, LaunchResolutionError) for outcome in outcomes) == 1


def test_prepublication_failure_never_exposes_final_run_root(
    resolver_case,
    tmp_path,
    monkeypatch,
):
    resolved = resolver_case["resolver"].resolve(resolver_case["request"])
    builder = StarterWorkspaceBuilder(resolver_case["resolver"]._settings)
    run_root = (tmp_path / "run").absolute()

    def fail_rename(*_arguments):
        raise LaunchWorkspaceError("injected prepublication failure")

    monkeypatch.setattr(builder, "_rename_no_replace", fail_rename)
    with pytest.raises(LaunchWorkspaceError, match="injected"):
        builder.build(
            resolved,
            run_root,
            run_id="run-1",
            campaign_id="campaign-1",
        )

    assert not os.path.lexists(run_root)
    assert tuple(tmp_path.glob(".launch-staging-*"))
    with pytest.raises(LaunchResolutionError, match="authority"):
        builder.build(
            resolved,
            (tmp_path / "retry").absolute(),
            run_id="run-1",
            campaign_id="campaign-1",
        )


def test_postpublication_failure_leaves_complete_resumable_container(
    resolver_case,
    tmp_path,
    monkeypatch,
):
    resolved = resolver_case["resolver"].resolve(resolver_case["request"])
    builder = StarterWorkspaceBuilder(resolver_case["resolver"]._settings)
    run_root = (tmp_path / "run").absolute()

    def fail_reopen(published_root, expected_pin, **_arguments):
        raise LaunchWorkspaceError("injected postpublication failure")

    monkeypatch.setattr(builder, "_verify_published", fail_reopen)
    with pytest.raises(LaunchWorkspaceError, match="postpublication"):
        builder.build(
            resolved,
            run_root,
            run_id="run-1",
            campaign_id="campaign-1",
        )

    assert run_root.is_dir()
    pin_path = run_root / resolver_case["resolver"]._settings.launch.bootstrap_pin_path
    manifest_path = (
        run_root / resolver_case["resolver"]._settings.launch.launch_manifest_path
    )
    pin = BootstrapPin.from_json_bytes(pin_path.read_bytes())
    assert LaunchManifest.from_json_bytes(manifest_path.read_bytes()) == (
        pin.launch_manifest
    )


def test_published_byte_tamper_fails_complete_reopen(resolver_case, tmp_path):
    prepared = _build(resolver_case, (tmp_path / "run").absolute())
    target = prepared.knowledge_snapshot / "snapshot.json"
    target.chmod(0o644)
    target.write_bytes(b"tampered")
    builder = StarterWorkspaceBuilder(resolver_case["resolver"]._settings)

    with pytest.raises((LaunchWorkspaceError, ValueError)):
        builder._verify_published(
            prepared.run_root,
            prepared.bootstrap_pin,
            requires_initial_state=True,
        )


def test_published_git_control_plane_tamper_fails_complete_reopen(
    resolver_case,
    tmp_path,
):
    prepared = _build(resolver_case, (tmp_path / "run").absolute())
    builder = StarterWorkspaceBuilder(resolver_case["resolver"]._settings)
    workspace_relative_path = (
        prepared.bootstrap_pin.installation_receipt.layout.workspace_relative_path
    )

    def change_config(root):
        path = root / workspace_relative_path / ".git" / "config"
        path.write_bytes(path.read_bytes() + b"\n")

    def change_index(root):
        path = root / workspace_relative_path / ".git" / "index"
        payload = path.read_bytes()
        path.write_bytes(payload[:-1] + bytes((payload[-1] ^ 1,)))

    def add_reference(root):
        path = root / workspace_relative_path / ".git" / "refs" / "replace" / ("a" * 40)
        path.parent.mkdir(mode=0o700)
        path.write_text("b" * 40 + "\n", encoding="ascii")
        path.chmod(0o600)

    def add_empty_hooks_directory(root):
        (root / workspace_relative_path / ".git" / "hooks").mkdir(mode=0o700)

    def make_bootstrap_pin_writable(root):
        (
            root
            / prepared.bootstrap_pin.installation_receipt.layout.bootstrap_pin_relative_path
        ).chmod(0o644)

    def oversize_git_object(root):
        object_id = (
            prepared.bootstrap_pin.installation_receipt.workspace_git_object_ids[0]
        )
        path = (
            root
            / workspace_relative_path
            / ".git"
            / "objects"
            / object_id[:2]
            / object_id[2:]
        )
        path.chmod(0o644)
        with path.open("wb") as handle:
            handle.truncate(
                resolver_case["resolver"]._settings.github.source_tree_size_bytes
                + resolver_case[
                    "resolver"
                ]._settings.github.git_tree_metadata_size_bytes
                + 1
            )
        path.chmod(0o444)

    def oversize_source_file(root):
        descriptor = prepared.bootstrap_pin.launch_manifest.expert_source.extraction_receipt.source_tree_files[
            0
        ]
        path = root / workspace_relative_path / descriptor.relative_path
        with path.open("wb") as handle:
            handle.truncate(
                resolver_case["resolver"]._settings.github.source_tree_size_bytes + 1
            )

    def oversize_knowledge_file(root):
        path = (
            root
            / prepared.bootstrap_pin.installation_receipt.layout.knowledge_snapshot_relative_path
            / "snapshot.json"
        )
        path.chmod(0o644)
        with path.open("wb") as handle:
            handle.truncate(
                resolver_case[
                    "resolver"
                ]._settings.launch.knowledge_snapshot_file_size_bytes
                + 1
            )
        path.chmod(0o444)

    def oversize_git_index(root):
        path = root / workspace_relative_path / ".git" / "index"
        with path.open("r+b") as handle:
            handle.truncate(
                resolver_case["resolver"]._settings.github.git_tree_metadata_size_bytes
                + 1
            )

    def oversize_git_config(root):
        path = root / workspace_relative_path / ".git" / "config"
        with path.open("r+b") as handle:
            handle.truncate(
                resolver_case["resolver"]._settings.github.control_blob_size_bytes + 1
            )

    for name, mutation in (
        ("config", change_config),
        ("index", change_index),
        ("reference", add_reference),
        ("directory", add_empty_hooks_directory),
        ("bootstrap-mode", make_bootstrap_pin_writable),
        ("oversize-object", oversize_git_object),
        ("oversize-source", oversize_source_file),
        ("oversize-knowledge", oversize_knowledge_file),
        ("oversize-index", oversize_git_index),
        ("oversize-config", oversize_git_config),
    ):
        root = (tmp_path / f"tampered-{name}").absolute()
        shutil.copytree(prepared.run_root, root)
        mutation(root)
        with pytest.raises(LaunchWorkspaceError):
            builder._verify_published(
                root,
                prepared.bootstrap_pin,
                requires_initial_state=True,
            )


def test_published_immutable_tree_rejects_shared_files_and_writable_parents(
    resolver_case,
    tmp_path,
):
    prepared = _build(resolver_case, (tmp_path / "run").absolute())
    builder = StarterWorkspaceBuilder(resolver_case["resolver"]._settings)
    snapshot = prepared.knowledge_snapshot / "snapshot.json"
    external_link = tmp_path / "shared-snapshot.json"
    os.link(snapshot, external_link)

    with pytest.raises(LaunchWorkspaceError, match="shared"):
        builder._verify_published(
            prepared.run_root,
            prepared.bootstrap_pin,
            requires_initial_state=True,
        )

    external_link.unlink()
    immutable_root = (
        prepared.run_root
        / prepared.bootstrap_pin.installation_receipt.layout.immutable_root_relative_path
    )
    immutable_root.chmod(0o755)
    with pytest.raises(LaunchWorkspaceError, match="writable"):
        builder._verify_published(
            prepared.run_root,
            prepared.bootstrap_pin,
            requires_initial_state=True,
        )


def test_publication_rejects_staging_path_substitution(
    resolver_case,
    tmp_path,
    monkeypatch,
):
    for replacement_kind in ("symlink", "directory"):
        resolved = resolver_case["resolver"].resolve(resolver_case["request"])
        builder = StarterWorkspaceBuilder(resolver_case["resolver"]._settings)
        original_rename = builder._rename_no_replace
        run_root = (tmp_path / f"run-{replacement_kind}").absolute()

        def substitute_then_rename(
            parent_descriptor,
            source_name,
            destination_name,
            expected_source_identity,
            *,
            kind=replacement_kind,
        ):
            original_name = f"{source_name}-original"
            os.rename(
                source_name,
                original_name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            if kind == "symlink":
                os.symlink(
                    original_name,
                    source_name,
                    dir_fd=parent_descriptor,
                )
            else:
                os.mkdir(source_name, mode=0o700, dir_fd=parent_descriptor)
            return original_rename(
                parent_descriptor,
                source_name,
                destination_name,
                expected_source_identity,
            )

        with monkeypatch.context() as patch:
            patch.setattr(builder, "_rename_no_replace", substitute_then_rename)
            with pytest.raises(LaunchWorkspaceError, match="identity changed"):
                builder.build(
                    resolved,
                    run_root,
                    run_id=f"run-{replacement_kind}",
                    campaign_id="campaign-1",
                )
        assert not os.path.lexists(run_root)


def test_prepared_workspace_authority_is_bound_and_one_shot(
    resolver_case,
    tmp_path,
):
    prepared = _build(resolver_case, (tmp_path / "run").absolute())
    clone = replace(prepared)

    with pytest.raises(LaunchWorkspaceError, match="builder authority"):
        replace(prepared, _builder_authority=object())
    with pytest.raises(LaunchWorkspaceError, match="layout"):
        replace(
            prepared,
            workspace=(tmp_path / "forged-workspace").absolute(),
        )
    with pytest.raises(LaunchWorkspaceError, match="authority"):
        clone.activate()

    receipt_payload = prepared.bootstrap_pin.installation_receipt.to_dict()
    receipt_payload.pop("workspace_installation_receipt_id")
    receipt_payload["run_id"] = "spliced-run"
    spliced_receipt = WorkspaceInstallationReceipt.mint(**receipt_payload)
    spliced_pin = BootstrapPin.mint(
        launch_manifest=prepared.bootstrap_pin.launch_manifest,
        launch_manifest_full_digest=(
            prepared.bootstrap_pin.launch_manifest_full_digest
        ),
        installation_receipt=spliced_receipt,
        exact_dependency_ids=tuple(
            sorted(
                {
                    prepared.bootstrap_pin.launch_manifest.launch_manifest_id,
                    spliced_receipt.workspace_installation_receipt_id,
                }
            )
        ),
    )
    spliced_clone = replace(prepared, bootstrap_pin=spliced_pin)
    with pytest.raises(LaunchWorkspaceError, match="authority"):
        spliced_clone.activate()

    assert type(prepared) is PreparedLaunchWorkspace
    prepared.activate()
    with pytest.raises(LaunchWorkspaceError, match="authority"):
        prepared.activate()


def test_prepared_workspace_consumption_rechecks_pinned_inodes(
    resolver_case,
    tmp_path,
):
    for replacement_kind in ("run-root", "workspace", "workspace-bytes"):
        prepared = _build(
            resolver_case,
            (tmp_path / f"run-{replacement_kind}").absolute(),
            run_id=f"run-{replacement_kind}",
        )
        if replacement_kind == "run-root":
            prepared.run_root.rename(tmp_path / f"original-{replacement_kind}")
            prepared.run_root.mkdir(mode=0o700)
        elif replacement_kind == "workspace":
            prepared.workspace.rename(prepared.run_root / "original-workspace")
            prepared.workspace.mkdir(mode=0o700)
        else:
            descriptor = prepared.bootstrap_pin.launch_manifest.expert_source.extraction_receipt.source_tree_files[
                0
            ]
            target = prepared.workspace / descriptor.relative_path
            payload = target.read_bytes()
            target.write_bytes(bytes((payload[0] ^ 1,)) + payload[1:])

        with pytest.raises((LaunchWorkspaceError, OSError)):
            prepared.activate()
        with pytest.raises(LaunchWorkspaceError, match="authority"):
            prepared.activate()


def test_prepared_workspace_consumption_has_terminal_identity_check(
    resolver_case,
    tmp_path,
    monkeypatch,
):
    prepared = _build(resolver_case, (tmp_path / "run").absolute())
    builder = prepared._builder_verifier
    original_verify = builder._verify_published

    def verify_then_replace(*arguments, **keywords):
        verified = original_verify(*arguments, **keywords)
        prepared.workspace.rename(prepared.run_root / "verified-workspace")
        prepared.workspace.mkdir(mode=0o700)
        return verified

    monkeypatch.setattr(builder, "_verify_published", verify_then_replace)
    with pytest.raises(LaunchWorkspaceError, match="filesystem closure"):
        prepared.activate()
    with pytest.raises(LaunchWorkspaceError, match="authority"):
        prepared.activate()


def test_prepared_workspace_consumption_has_terminal_byte_check(
    resolver_case,
    tmp_path,
    monkeypatch,
):
    prepared = _build(resolver_case, (tmp_path / "run").absolute())
    builder = prepared._builder_verifier
    original_verify = builder._verify_published

    def verify_then_mutate(*arguments, **keywords):
        verified = original_verify(*arguments, **keywords)
        descriptor = prepared.bootstrap_pin.launch_manifest.expert_source.extraction_receipt.source_tree_files[
            0
        ]
        target = prepared.workspace / descriptor.relative_path
        payload = target.read_bytes()
        target.write_bytes(bytes((payload[0] ^ 1,)) + payload[1:])
        return verified

    monkeypatch.setattr(builder, "_verify_published", verify_then_mutate)
    with pytest.raises(LaunchWorkspaceError, match="verified descriptor"):
        prepared.activate()
    with pytest.raises(LaunchWorkspaceError, match="authority"):
        prepared.activate()


def test_active_workspace_admits_only_the_current_replaceable_leaf(
    resolver_case,
    tmp_path,
):
    prepared = _build(resolver_case, (tmp_path / "run").absolute())
    active = prepared.activate()
    replacement = (tmp_path / "replacement-workspace").absolute()
    retired = (tmp_path / "retired-workspace").absolute()
    shutil.copytree(active.workspace, replacement)

    with ExitStack() as staging_descriptors:
        _staging_descriptor, staging_identity = (
            active._open_run_action_workspace_staging(staging_descriptors)
        )
        receipt = active.bootstrap_pin.installation_receipt
        assert staging_identity == (
            receipt.run_action_workspace_staging_device,
            receipt.run_action_workspace_staging_inode,
        )

    with ExitStack() as stale_descriptors:
        stale_descriptor, stale_identity = active._open_execution_workspace(
            stale_descriptors
        )
        active.workspace.rename(retired)
        replacement.rename(active.workspace)

        active.require_control_authority()
        with ExitStack() as current_descriptors:
            current_descriptor, current_identity = active._open_execution_workspace(
                current_descriptors
            )
            assert current_identity != stale_identity
            active._require_execution_workspace(
                current_descriptor,
                current_identity,
            )
        with pytest.raises(LaunchWorkspaceError, match="no longer public"):
            active._require_execution_workspace(
                stale_descriptor,
                stale_identity,
            )
    active.close()


def test_active_workspace_rejects_leaf_outside_promotion_filesystem(
    resolver_case,
    tmp_path,
    monkeypatch,
):
    prepared = _build(resolver_case, (tmp_path / "run").absolute())
    active = prepared.activate()
    workspace_identity = (
        active.workspace.stat(follow_symlinks=False).st_dev,
        active.workspace.stat(follow_symlinks=False).st_ino,
    )
    original = workspace_module._require_owner_private_directory

    def substitute_workspace_device(descriptor, name):
        identity = original(descriptor, name)
        return (
            (identity[0] + 1, identity[1])
            if identity == workspace_identity
            else identity
        )

    monkeypatch.setattr(
        workspace_module,
        "_require_owner_private_directory",
        substitute_workspace_device,
    )
    with ExitStack() as descriptors:
        with pytest.raises(
            LaunchWorkspaceError,
            match="outside the workspace-promotion filesystem",
        ):
            active._open_execution_workspace(descriptors)
    active.close()


def test_reopen_rejects_workspace_staging_root_substitution(
    resolver_case,
    tmp_path,
):
    settings = resolver_case["resolver"]._settings
    prepared = _build(resolver_case, (tmp_path / "run").absolute())
    layout = prepared.bootstrap_pin.installation_receipt.layout
    active = prepared.activate()
    active.close()
    workspace_staging = (
        prepared.run_root / layout.run_action_workspace_staging_relative_path
    )
    workspace_staging.rename(tmp_path / "original-workspace-staging")
    workspace_staging.mkdir(mode=0o700)

    with pytest.raises(LaunchWorkspaceError, match="workspace staging root differs"):
        StarterWorkspaceBuilder(settings).reopen(prepared.run_root)


def test_reopen_rejects_external_component_aliases_and_extra_root_state(
    resolver_case,
    tmp_path,
):
    prepared = _build(resolver_case, (tmp_path / "run").absolute())
    builder = StarterWorkspaceBuilder(resolver_case["resolver"]._settings)
    layout = prepared.bootstrap_pin.installation_receipt.layout

    def alias_workspace(root, name):
        external = tmp_path / f"{name}-external-workspace"
        (root / layout.workspace_relative_path).rename(external)
        os.symlink(
            external,
            root / layout.workspace_relative_path,
            target_is_directory=True,
        )

    def alias_control_parent(root, name):
        control_parent = (root / layout.launch_manifest_relative_path).parent
        external = tmp_path / f"{name}-external-control"
        control_parent.rename(external)
        os.symlink(external, control_parent, target_is_directory=True)

    def add_extra_root_file(root, _name):
        path = root / "unverified-state"
        path.write_bytes(b"extra")

    for name, mutation in (
        ("workspace-alias", alias_workspace),
        ("control-alias", alias_control_parent),
        ("extra-root", add_extra_root_file),
    ):
        root = (tmp_path / name).absolute()
        shutil.copytree(prepared.run_root, root)
        mutation(root, name)
        with pytest.raises((LaunchWorkspaceError, OSError)):
            builder._verify_published(
                root,
                prepared.bootstrap_pin,
                requires_initial_state=True,
            )


def test_existing_destination_fails_without_consuming_authority(
    resolver_case,
    tmp_path,
):
    resolved = resolver_case["resolver"].resolve(resolver_case["request"])
    builder = StarterWorkspaceBuilder(resolver_case["resolver"]._settings)
    existing = (tmp_path / "existing").absolute()
    existing.mkdir()

    with pytest.raises(LaunchWorkspaceError, match="already exists"):
        builder.build(
            resolved,
            existing,
            run_id="run-1",
            campaign_id="campaign-1",
        )
    prepared = builder.build(
        resolved,
        (tmp_path / "run").absolute(),
        run_id="run-1",
        campaign_id="campaign-1",
    )

    assert prepared.bootstrap_pin_path.is_file()


def test_published_envelope_accepts_bounded_derived_state_files(
    resolver_case,
    tmp_path,
):
    prepared = _build(resolver_case, (tmp_path / "run").absolute())
    layout = prepared.bootstrap_pin.installation_receipt.layout
    builder = prepared._builder_verifier
    projection_payloads = {
        layout.run_idea_archive_relative_path: b"{}",
        layout.run_experiment_history_relative_path: b"{}",
        layout.run_execution_journal_relative_path: b'{"event":"one"}\n',
    }
    for relative_path, payload in projection_payloads.items():
        target = prepared.run_root / relative_path
        target.write_bytes(payload)
        target.chmod(0o400)
    generation = (
        prepared.run_root
        / layout.run_derived_state_store_relative_path
        / f"generation-{'a' * 64}.bundle"
    )
    generation.write_bytes(b"{}")
    generation.chmod(0o400)
    staged = (
        prepared.run_root
        / layout.run_derived_state_staging_relative_path
        / f"generation-{'b' * 64}-{'c' * 32}.tmp"
    )
    staged.write_bytes(b"{}")
    staged.chmod(0o600)

    builder._verify_outer_run_root_closure(
        prepared.run_root,
        layout,
        prepared._published_root_identity,
    )


@pytest.mark.parametrize(
    ("relative_path_name", "entry_name", "mode", "message"),
    [
        (
            "run_derived_state_store_relative_path",
            "unknown.json",
            0o400,
            "derived-state object",
        ),
        (
            "run_derived_state_staging_relative_path",
            "generation-invalid.tmp",
            0o600,
            "derived-state staging",
        ),
    ],
)
def test_published_envelope_rejects_unrecognized_derived_state_entries(
    resolver_case,
    tmp_path,
    relative_path_name,
    entry_name,
    mode,
    message,
):
    prepared = _build(resolver_case, (tmp_path / "run").absolute())
    layout = prepared.bootstrap_pin.installation_receipt.layout
    target = prepared.run_root / getattr(layout, relative_path_name) / entry_name
    target.write_bytes(b"unsafe")
    target.chmod(mode)

    with pytest.raises(LaunchWorkspaceError, match=message):
        prepared._builder_verifier._verify_outer_run_root_closure(
            prepared.run_root,
            layout,
            prepared._published_root_identity,
        )


def test_published_envelope_rejects_legacy_run_action_lock(
    resolver_case,
    tmp_path,
):
    prepared = _build(resolver_case, (tmp_path / "run").absolute())
    layout = prepared.bootstrap_pin.installation_receipt.layout
    target = (
        prepared.run_root
        / layout.run_action_store_relative_path
        / f"operation-{'a' * 64}.lock"
    )
    target.write_bytes(b"")
    target.chmod(0o600)

    with pytest.raises(LaunchWorkspaceError, match="action store entry"):
        prepared._builder_verifier._verify_outer_run_root_closure(
            prepared.run_root,
            layout,
            prepared._published_root_identity,
        )


def test_published_envelope_rejects_unexpected_action_workspace_staging(
    resolver_case,
    tmp_path,
):
    prepared = _build(resolver_case, (tmp_path / "run").absolute())
    layout = prepared.bootstrap_pin.installation_receipt.layout
    target = (
        prepared.run_root
        / layout.run_action_workspace_staging_relative_path
        / "unexpected"
    )
    target.mkdir(mode=0o700)

    with pytest.raises(
        LaunchWorkspaceError,
        match="workspace staging entry is unsafe",
    ):
        prepared._builder_verifier._verify_outer_run_root_closure(
            prepared.run_root,
            layout,
            prepared._published_root_identity,
        )


def test_published_envelope_rejects_special_action_workspace_staging_entry(
    resolver_case,
    tmp_path,
):
    prepared = _build(resolver_case, (tmp_path / "run").absolute())
    layout = prepared.bootstrap_pin.installation_receipt.layout
    staged_workspace = (
        prepared.run_root
        / layout.run_action_workspace_staging_relative_path
        / "workspace"
    )
    staged_workspace.mkdir(mode=0o700)
    os.mkfifo(staged_workspace / "unsafe")

    with pytest.raises(
        LaunchWorkspaceError,
        match="workspace staging entry is unsafe",
    ):
        prepared._builder_verifier._verify_outer_run_root_closure(
            prepared.run_root,
            layout,
            prepared._published_root_identity,
        )


def test_published_envelope_rejects_writable_projection(
    resolver_case,
    tmp_path,
):
    prepared = _build(resolver_case, (tmp_path / "run").absolute())
    layout = prepared.bootstrap_pin.installation_receipt.layout
    archive = prepared.run_root / layout.run_idea_archive_relative_path
    archive.write_bytes(b"{}")
    archive.chmod(0o600)

    with pytest.raises(LaunchWorkspaceError, match="projection"):
        prepared._builder_verifier._verify_outer_run_root_closure(
            prepared.run_root,
            layout,
            prepared._published_root_identity,
        )


def test_published_envelope_rejects_oversized_projection(
    resolver_case,
    tmp_path,
):
    prepared = _build(resolver_case, (tmp_path / "run").absolute())
    layout = prepared.bootstrap_pin.installation_receipt.layout
    settings = resolver_case["resolver"]._settings
    launch = replace(settings.launch, run_idea_archive_size_bytes=1)
    builder = StarterWorkspaceBuilder(replace(settings, launch=launch))
    archive = prepared.run_root / layout.run_idea_archive_relative_path
    archive.write_bytes(b"{}")
    archive.chmod(0o400)

    with pytest.raises(LaunchWorkspaceError, match="projection"):
        builder._verify_outer_run_root_closure(
            prepared.run_root,
            layout,
            prepared._published_root_identity,
        )


def test_published_envelope_rejects_oversized_checkpoint_staging(
    resolver_case,
    tmp_path,
):
    prepared = _build(resolver_case, (tmp_path / "run").absolute())
    layout = prepared.bootstrap_pin.installation_receipt.layout
    settings = resolver_case["resolver"]._settings
    launch = replace(settings.launch, run_checkpoint_size_bytes=1)
    builder = StarterWorkspaceBuilder(replace(settings, launch=launch))
    staging = (
        prepared.run_root
        / layout.run_checkpoint_staging_relative_path
        / f"checkpoint-{'a' * 64}-{'b' * 32}.tmp"
    )
    staging.write_bytes(b"{}")
    staging.chmod(0o600)

    with pytest.raises(LaunchWorkspaceError, match="checkpoint staging"):
        builder._verify_outer_run_root_closure(
            prepared.run_root,
            layout,
            prepared._published_root_identity,
        )


def test_published_envelope_rejects_empty_permanent_generation(
    resolver_case,
    tmp_path,
):
    prepared = _build(resolver_case, (tmp_path / "run").absolute())
    layout = prepared.bootstrap_pin.installation_receipt.layout
    generation = (
        prepared.run_root
        / layout.run_derived_state_store_relative_path
        / f"generation-{'a' * 64}.bundle"
    )
    generation.touch(mode=0o400)

    with pytest.raises(LaunchWorkspaceError, match="derived-state object"):
        prepared._builder_verifier._verify_outer_run_root_closure(
            prepared.run_root,
            layout,
            prepared._published_root_identity,
        )


@pytest.mark.parametrize(
    ("directory_field", "entry_names", "limit_field", "message"),
    [
        (
            "run_derived_state_store_relative_path",
            (
                f"generation-{'a' * 64}.bundle",
                f"generation-{'b' * 64}.bundle",
            ),
            "run_derived_state_store_entry_limit",
            "derived-state store",
        ),
        (
            "run_derived_state_staging_relative_path",
            (
                f"generation-{'a' * 64}-{'b' * 32}.tmp",
                f"generation-{'c' * 64}-{'d' * 32}.tmp",
            ),
            "run_derived_state_staging_entry_limit",
            "derived-state staging",
        ),
    ],
)
def test_published_envelope_enforces_derived_state_entry_bounds(
    resolver_case,
    tmp_path,
    directory_field,
    entry_names,
    limit_field,
    message,
):
    prepared = _build(resolver_case, (tmp_path / "run").absolute())
    layout = prepared.bootstrap_pin.installation_receipt.layout
    settings = resolver_case["resolver"]._settings
    launch = replace(settings.launch, **{limit_field: 1})
    builder = StarterWorkspaceBuilder(replace(settings, launch=launch))
    directory = prepared.run_root / getattr(layout, directory_field)
    for entry_name in entry_names:
        entry = directory / entry_name
        entry.write_bytes(b"{}")
        entry.chmod(0o400)

    with pytest.raises(LaunchWorkspaceError, match=message):
        builder._verify_outer_run_root_closure(
            prepared.run_root,
            layout,
            prepared._published_root_identity,
        )


def test_workspace_contracts_reject_spliced_or_legacy_authority(
    resolver_case,
    tmp_path,
):
    prepared = _build(resolver_case, (tmp_path / "run").absolute())
    pin = prepared.bootstrap_pin
    receipt = pin.installation_receipt

    assert BootstrapPin.from_json_bytes(pin.to_json_bytes()) == pin
    assert (
        WorkspaceInstallationReceipt.from_json_bytes(receipt.to_json_bytes()) == receipt
    )
    legacy = {
        "bootstrap_pin_id": pin.bootstrap_pin_id,
        "launch_manifest_id": pin.launch_manifest.launch_manifest_id,
        "workspace_tree_hash": receipt.expert_source_tree_hash,
    }
    with pytest.raises(ContractValidationError):
        BootstrapPin.from_dict(legacy)
    with pytest.raises(LaunchContractError, match="do not join"):
        replace(
            pin,
            launch_manifest_full_digest=tree_or_blob_digest(b"wrong"),
        )
    with pytest.raises(LaunchContractError, match="do not join"):
        receipt_payload = receipt.to_dict()
        receipt_payload.pop("workspace_installation_receipt_id")
        receipt_payload["knowledge_package_tree_hash"] = tree_or_blob_digest(b"wrong")
        spliced_receipt = WorkspaceInstallationReceipt.mint(**receipt_payload)
        BootstrapPin.mint(
            launch_manifest=pin.launch_manifest,
            launch_manifest_full_digest=pin.launch_manifest_full_digest,
            installation_receipt=spliced_receipt,
            exact_dependency_ids=tuple(
                sorted(
                    {
                        pin.launch_manifest.launch_manifest_id,
                        spliced_receipt.workspace_installation_receipt_id,
                    }
                )
            ),
        )
    with pytest.raises(LaunchContractError, match="prefix-disjoint"):
        replace(
            receipt.layout,
            task_adapter_relative_path=(
                f"{receipt.layout.knowledge_snapshot_relative_path}/adapter"
            ),
        )
    with pytest.raises(LaunchContractError, match="prefix-disjoint"):
        replace(
            receipt.layout,
            bootstrap_pin_relative_path=(
                f"{receipt.layout.launch_manifest_relative_path}/pin.json"
            ),
        )
    with pytest.raises(LaunchContractError, match="prefix-disjoint"):
        replace(
            receipt.layout,
            launch_manifest_relative_path=(
                f"{receipt.layout.bootstrap_pin_relative_path}/manifest.json"
            ),
        )
