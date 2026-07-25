from __future__ import annotations

import os
import stat
from contextlib import ExitStack
from dataclasses import replace

import pytest

from kapso.core.config import load_config
from kapso.cross_run.launch import run_action_runtime_volume as volume_module
from kapso.cross_run.launch.run_action_contracts import (
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_docker_inspect import (
    observe_runtime_volume,
)
from kapso.cross_run.launch.run_action_runtime_volume import (
    DockerRunActionEmptyVolumeObservation,
    DockerRunActionPreparedVolumeObservation,
    RunActionRuntimeVolumeError,
    _materialize_layout_at_descriptor,
    _open_exact_regular_file,
    _plan_runtime_volume_layout,
    _require_same_exact_regular_file,
)
from kapso.cross_run.launch.run_action_keeper_helper import (
    read_run_action_descriptor_mount_id,
)
from kapso.cross_run.launch.run_action_reservation_contracts import (
    RunActionWorkspaceBinding,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionContainerLabel,
    RunActionCredentialMode,
    RunActionPreparationClaim,
)
from kapso.cross_run.launch.workspace_frontier import (
    inspect_run_workspace_frontier,
)
from kapso.cross_run.settings import CrossRunSettings
from test_run_action_docker_inspect import _volume_raw
from test_run_action_docker_projection import _policy
from test_launch_resolver import resolver_case
from test_run_action_supervisor_contracts import (
    _claim,
    _fixture_content_id,
    _prepared_execution,
    _remint_contract,
    _volume_authority,
)
from test_run_state_publisher import publisher_case

_CANONICAL_CONFIG_PATH = "src/kapso/config.yaml"
_GENERATION_NONCE = "9" * 32


@pytest.fixture(scope="module")
def layout_context():
    settings = CrossRunSettings.from_dict(
        load_config(_CANONICAL_CONFIG_PATH)["cross_run"]
    )
    policy = _policy(
        settings.docker,
        workspace_access=RunFrontierWorkspaceAccess.NONE,
        credential_mode=RunActionCredentialMode.NONE,
    )
    claim = _claim(policy=policy)
    authority = _volume_authority(claim, nonce=_GENERATION_NONCE)
    volume = observe_runtime_volume(
        _volume_raw(authority, settings.docker),
        claim,
        authority,
        settings.docker,
    )
    block_size = 4096
    effective_block_count = authority.size_limit_bytes // block_size
    empty = DockerRunActionEmptyVolumeObservation(
        runtime_volume_authority=authority,
        docker_volume_observation=volume,
        keeper_container_id="a" * 64,
        keeper_process_id=101,
        keeper_process_start_time_ticks=123456,
        process_cgroup_path=(f"/test.kapso.run_action.slice/docker-{'a' * 64}.scope"),
        mount_id=1232,
        device=os.makedev(0, 73),
        root_inode=71,
        filesystem_type="tmpfs",
        observed_mount_flags=("nodev", "nosuid", "noswap"),
        owner_user_id=authority.owner_user_id,
        owner_group_id=authority.owner_group_id,
        root_mode=authority.root_mode,
        allocation_block_size_bytes=block_size,
        effective_block_count=effective_block_count,
        effective_size_bytes=authority.size_limit_bytes,
        effective_inode_limit=authority.inode_limit,
        used_block_count=0,
        used_size_bytes=0,
        used_inode_count=1,
        available_block_count=effective_block_count,
        available_size_bytes=authority.size_limit_bytes,
        available_inode_count=authority.inode_limit - 1,
        empty_entry_count=0,
        empty_size_bytes=0,
    )
    return settings, claim, authority, empty


def _open_empty_root(path, descriptors):
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    descriptors.callback(os.close, descriptor)
    return descriptor


def test_descriptor_materializer_publishes_complete_layout_and_sentinel_last(
    layout_context,
    tmp_path,
):
    settings, claim, authority, empty = layout_context
    root_path = tmp_path / "runtime-volume"
    with ExitStack() as descriptors:
        root_descriptor = _open_empty_root(root_path, descriptors)
        plan = _plan_runtime_volume_layout(
            claim,
            empty,
            workspace_descriptor=None,
            settings=settings.launch,
        )

        workspace_frontier = _materialize_layout_at_descriptor(
            root_descriptor,
            claim=claim,
            authority=authority,
            plan=plan,
            workspace_descriptor=None,
            settings=settings.launch,
        )

    assert workspace_frontier is None
    assert tuple(sorted(path.name for path in root_path.iterdir())) == (
        ".kapso-generation",
        "input",
        "result",
        "temporary",
    )
    assert (root_path / ".kapso-generation").read_bytes() == (
        authority.generation_nonce.encode("ascii")
    )
    assert stat.S_IMODE((root_path / ".kapso-generation").stat().st_mode) == 0o400
    assert (root_path / "input" / "request.blob").read_bytes() == b""
    assert (root_path / "result" / "result.blob").read_bytes() == b""
    assert tuple((root_path / "temporary").iterdir()) == ()
    assert all(
        stat.S_IMODE((root_path / name).stat().st_mode) == 0o700
        for name in ("input", "result", "temporary")
    )
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o600
        for path in (
            root_path / "input" / "request.blob",
            root_path / "result" / "result.blob",
        )
    )


@pytest.mark.parametrize(
    ("failure_destination", "staging_present", "pending_present"),
    (
        ("input", True, False),
        ("result", True, False),
        ("temporary", True, False),
        (f".kapso-generation.pending-{_GENERATION_NONCE}", True, False),
        (".kapso-generation", False, True),
    ),
)
def test_descriptor_materializer_leaves_no_published_sentinel_before_final_rename(
    layout_context,
    tmp_path,
    monkeypatch,
    failure_destination,
    staging_present,
    pending_present,
):
    settings, claim, authority, empty = layout_context
    root_path = tmp_path / "runtime-volume"
    original_rename = volume_module._rename_no_replace

    def fail_final_sentinel(
        source_descriptor,
        source_name,
        destination_descriptor,
        destination_name,
    ):
        if destination_name == failure_destination:
            raise OSError("simulated publication crash")
        return original_rename(
            source_descriptor,
            source_name,
            destination_descriptor,
            destination_name,
        )

    monkeypatch.setattr(volume_module, "_rename_no_replace", fail_final_sentinel)
    with ExitStack() as descriptors:
        root_descriptor = _open_empty_root(root_path, descriptors)
        plan = _plan_runtime_volume_layout(
            claim,
            empty,
            workspace_descriptor=None,
            settings=settings.launch,
        )
        with pytest.raises(OSError, match="simulated publication crash"):
            _materialize_layout_at_descriptor(
                root_descriptor,
                claim=claim,
                authority=authority,
                plan=plan,
                workspace_descriptor=None,
                settings=settings.launch,
            )

    assert not (root_path / ".kapso-generation").exists()
    assert (
        root_path / f".kapso-generation.pending-{_GENERATION_NONCE}"
    ).exists() is pending_present
    assert (
        root_path / f".kapso-prepare-{_GENERATION_NONCE}"
    ).exists() is staging_present


def test_staging_removal_failure_cannot_publish_the_final_sentinel(
    layout_context,
    tmp_path,
    monkeypatch,
):
    settings, claim, authority, empty = layout_context
    root_path = tmp_path / "runtime-volume"
    original_rmdir = volume_module.os.rmdir

    def fail_staging_removal(path, *, dir_fd=None):
        if path == f".kapso-prepare-{_GENERATION_NONCE}":
            raise OSError("simulated staging-removal crash")
        return original_rmdir(path, dir_fd=dir_fd)

    monkeypatch.setattr(volume_module.os, "rmdir", fail_staging_removal)
    with ExitStack() as descriptors:
        root_descriptor = _open_empty_root(root_path, descriptors)
        plan = _plan_runtime_volume_layout(
            claim,
            empty,
            workspace_descriptor=None,
            settings=settings.launch,
        )
        with pytest.raises(OSError, match="simulated staging-removal crash"):
            _materialize_layout_at_descriptor(
                root_descriptor,
                claim=claim,
                authority=authority,
                plan=plan,
                workspace_descriptor=None,
                settings=settings.launch,
            )

    assert not (root_path / ".kapso-generation").exists()
    assert (root_path / f".kapso-generation.pending-{_GENERATION_NONCE}").is_file()
    assert (root_path / f".kapso-prepare-{_GENERATION_NONCE}").is_dir()


@pytest.mark.parametrize("resource", ("bytes", "inodes"))
def test_layout_plan_requires_strict_peak_and_execution_headroom(
    layout_context,
    resource,
):
    settings, claim, _authority, empty = layout_context
    admitted = _plan_runtime_volume_layout(
        claim,
        empty,
        workspace_descriptor=None,
        settings=settings.launch,
    )
    limits = claim.execution_policy.docker_resource_limits
    if resource == "bytes":
        future_size_bytes = sum(
            volume_module._allocated_size_bytes(
                file_plan.payload_size_limit_bytes,
                empty.allocation_block_size_bytes,
            )
            for file_plan in admitted.file_plans
        ) + volume_module._allocated_size_bytes(
            limits.runtime_temporary_reservation_size_bytes,
            empty.allocation_block_size_bytes,
        )
        exact_available_size = admitted.preparation_size_bytes + future_size_bytes
        exhausted = replace(
            empty,
            effective_block_count=(
                exact_available_size // empty.allocation_block_size_bytes
            ),
            effective_size_bytes=exact_available_size,
            available_block_count=(
                exact_available_size // empty.allocation_block_size_bytes
            ),
            available_size_bytes=exact_available_size,
        )
    else:
        exact_available_inodes = (
            admitted.preparation_inode_count
            + limits.runtime_temporary_reservation_inode_count
        )
        exhausted = replace(
            empty,
            effective_inode_limit=exact_available_inodes + 1,
            available_inode_count=exact_available_inodes,
        )

    with pytest.raises(
        RunActionRuntimeVolumeError,
        match="lacks peak preparation and execution headroom",
    ):
        _plan_runtime_volume_layout(
            claim,
            exhausted,
            workspace_descriptor=None,
            settings=settings.launch,
        )


def test_descriptor_materializer_rejects_nonempty_or_second_publication(
    layout_context,
    tmp_path,
):
    settings, claim, authority, empty = layout_context
    root_path = tmp_path / "runtime-volume"
    with ExitStack() as descriptors:
        root_descriptor = _open_empty_root(root_path, descriptors)
        plan = _plan_runtime_volume_layout(
            claim,
            empty,
            workspace_descriptor=None,
            settings=settings.launch,
        )
        (root_path / "substituted").write_bytes(b"foreign")
        with pytest.raises(RunActionRuntimeVolumeError, match="no longer empty"):
            _materialize_layout_at_descriptor(
                root_descriptor,
                claim=claim,
                authority=authority,
                plan=plan,
                workspace_descriptor=None,
                settings=settings.launch,
            )


def test_prepared_volume_aggregate_rejects_layout_splices():
    prepared = _prepared_execution()
    observation = DockerRunActionPreparedVolumeObservation(
        preparation_claim=prepared.preparation_claim,
        runtime_volume_evidence=prepared.runtime_volume_evidence,
        input_file=prepared.input_file,
        result_file=prepared.result_file,
        credential_file=prepared.credential_file,
        workspace_proof=prepared.workspace_proof,
        layout_proof=prepared.layout_proof,
    )
    substituted_layout = _prepared_execution(inode_offset=7).layout_proof

    assert observation.layout_proof.runtime_volume_evidence_id == (
        observation.runtime_volume_evidence.runtime_volume_evidence_id
    )
    with pytest.raises(
        RunActionRuntimeVolumeError,
        match="observation is incomplete",
    ):
        replace(observation, layout_proof=substituted_layout)


@pytest.mark.parametrize(
    "mutation",
    (
        "directory_relative_paths",
        "logical_content_size_bytes",
        "logical_entry_count",
        "observed_used_size_bytes",
        "observed_used_inode_count",
    ),
)
def test_prepared_volume_aggregate_rejects_same_graph_layout_lies(mutation):
    prepared = _prepared_execution()
    observation = DockerRunActionPreparedVolumeObservation(
        preparation_claim=prepared.preparation_claim,
        runtime_volume_evidence=prepared.runtime_volume_evidence,
        input_file=prepared.input_file,
        result_file=prepared.result_file,
        credential_file=prepared.credential_file,
        workspace_proof=prepared.workspace_proof,
        layout_proof=prepared.layout_proof,
    )
    original_value = getattr(observation.layout_proof, mutation)
    substituted_value = (
        ("alien",) if mutation == "directory_relative_paths" else original_value + 1
    )
    substituted_layout = _remint_contract(
        observation.layout_proof,
        **{mutation: substituted_value},
    )

    with pytest.raises(
        RunActionRuntimeVolumeError,
        match="observation is incomplete",
    ):
        replace(observation, layout_proof=substituted_layout)


def test_prepared_volume_aggregate_rejects_claim_policy_authority_splice():
    prepared = _prepared_execution()
    observation = DockerRunActionPreparedVolumeObservation(
        preparation_claim=prepared.preparation_claim,
        runtime_volume_evidence=prepared.runtime_volume_evidence,
        input_file=prepared.input_file,
        result_file=prepared.result_file,
        credential_file=prepared.credential_file,
        workspace_proof=prepared.workspace_proof,
        layout_proof=prepared.layout_proof,
    )
    substituted_authority = _remint_contract(
        observation.runtime_volume_evidence.volume_authority,
        labels=tuple(
            RunActionContainerLabel(
                key=label.key,
                value=(
                    _fixture_content_id("run-action-reservation", "foreign")
                    if label.key == "com.kapso.run-action.reservation"
                    else label.value
                ),
            )
            for label in observation.runtime_volume_evidence.volume_authority.labels
        ),
    )
    substituted_sentinel = _remint_contract(
        observation.runtime_volume_evidence.sentinel_evidence,
        runtime_volume_authority_id=(substituted_authority.runtime_volume_authority_id),
    )
    substituted_evidence = _remint_contract(
        observation.runtime_volume_evidence,
        volume_authority=substituted_authority,
        observed_labels=substituted_authority.labels,
        sentinel_evidence=substituted_sentinel,
    )
    substituted_files = tuple(
        _remint_contract(
            prepared_file,
            runtime_volume_authority_id=(
                substituted_authority.runtime_volume_authority_id
            ),
        )
        for prepared_file in (
            observation.input_file,
            observation.result_file,
            observation.credential_file,
        )
        if prepared_file is not None
    )
    substituted_workspace = (
        None
        if observation.workspace_proof is None
        else _remint_contract(
            observation.workspace_proof,
            runtime_volume_authority_id=(
                substituted_authority.runtime_volume_authority_id
            ),
        )
    )
    substituted_layout = _remint_contract(
        observation.layout_proof,
        runtime_volume_authority_id=(substituted_authority.runtime_volume_authority_id),
        runtime_volume_evidence_id=(substituted_evidence.runtime_volume_evidence_id),
        prepared_file_ids=tuple(
            sorted(
                prepared_file.prepared_file_id for prepared_file in substituted_files
            )
        ),
        prepared_workspace_proof_id=(
            None
            if substituted_workspace is None
            else substituted_workspace.prepared_workspace_proof_id
        ),
    )

    with pytest.raises(
        RunActionRuntimeVolumeError,
        match="observation is incomplete",
    ):
        replace(
            observation,
            runtime_volume_evidence=substituted_evidence,
            input_file=substituted_files[0],
            result_file=substituted_files[1],
            credential_file=(
                None if len(substituted_files) == 2 else substituted_files[2]
            ),
            workspace_proof=substituted_workspace,
            layout_proof=substituted_layout,
        )


@pytest.mark.parametrize(
    "changes",
    (
        {"generation_nonce": "f" * 32},
        {"owner_user_id": 1001},
        {"owner_group_id": 1001},
        {"payload_size_limit_bytes": 1},
    ),
)
def test_prepared_volume_aggregate_rejects_file_authority_splices(changes):
    prepared = _prepared_execution()
    observation = DockerRunActionPreparedVolumeObservation(
        preparation_claim=prepared.preparation_claim,
        runtime_volume_evidence=prepared.runtime_volume_evidence,
        input_file=prepared.input_file,
        result_file=prepared.result_file,
        credential_file=prepared.credential_file,
        workspace_proof=prepared.workspace_proof,
        layout_proof=prepared.layout_proof,
    )
    substituted_input = _remint_contract(observation.input_file, **changes)
    substituted_layout = _remint_contract(
        observation.layout_proof,
        prepared_file_ids=tuple(
            sorted(
                (
                    substituted_input.prepared_file_id,
                    observation.result_file.prepared_file_id,
                    *(
                        ()
                        if observation.credential_file is None
                        else (observation.credential_file.prepared_file_id,)
                    ),
                )
            )
        ),
    )

    with pytest.raises(
        RunActionRuntimeVolumeError,
        match="observation is incomplete",
    ):
        replace(
            observation,
            input_file=substituted_input,
            layout_proof=substituted_layout,
        )


def test_prepared_volume_aggregate_rejects_workspace_authority_splice():
    prepared = _prepared_execution()
    assert prepared.workspace_proof is not None
    observation = DockerRunActionPreparedVolumeObservation(
        preparation_claim=prepared.preparation_claim,
        runtime_volume_evidence=prepared.runtime_volume_evidence,
        input_file=prepared.input_file,
        result_file=prepared.result_file,
        credential_file=prepared.credential_file,
        workspace_proof=prepared.workspace_proof,
        layout_proof=prepared.layout_proof,
    )
    substituted_workspace = _remint_contract(
        observation.workspace_proof,
        generation_nonce="f" * 32,
    )
    substituted_layout = _remint_contract(
        observation.layout_proof,
        prepared_workspace_proof_id=(substituted_workspace.prepared_workspace_proof_id),
    )

    with pytest.raises(
        RunActionRuntimeVolumeError,
        match="observation is incomplete",
    ):
        replace(
            observation,
            workspace_proof=substituted_workspace,
            layout_proof=substituted_layout,
        )


@pytest.mark.parametrize("mutation", ("sentinel_content", "prepared_file_mode"))
def test_exact_file_observation_detects_mutation_after_initial_read(
    layout_context,
    tmp_path,
    mutation,
):
    _settings, _claim, authority, _empty = layout_context
    root_path = tmp_path / "observation-root"
    file_path = root_path / "observed"
    with ExitStack() as descriptors:
        root_descriptor = _open_empty_root(root_path, descriptors)
        if mutation == "sentinel_content":
            expected_payload = authority.generation_nonce.encode("ascii")
            expected_mode = 0o400
        else:
            expected_payload = b""
            expected_mode = 0o600
        file_path.write_bytes(expected_payload)
        file_path.chmod(expected_mode)
        observation = _open_exact_regular_file(
            descriptors,
            root_descriptor,
            "observed",
            expected_payload=expected_payload,
            expected_mode=expected_mode,
            authority=authority,
            root_mount_id=read_run_action_descriptor_mount_id(root_descriptor),
            root_device=os.fstat(root_descriptor).st_dev,
        )

        if mutation == "sentinel_content":
            file_path.chmod(0o600)
            file_path.write_bytes(b"f" * len(expected_payload))
            file_path.chmod(expected_mode)
        else:
            file_path.chmod(0o400)

        with pytest.raises(
            RunActionRuntimeVolumeError,
            match="changed during exact observation",
        ):
            _require_same_exact_regular_file(observation)


def test_layout_materialization_copies_complete_workspace_and_git_closure(
    layout_context,
    publisher_case,
    tmp_path,
):
    settings, _claim_without_workspace, _authority, empty_without_workspace = (
        layout_context
    )
    policy = _policy(settings.docker)
    initial_claim = _claim(policy=policy)
    root_path = tmp_path / "runtime-volume"
    with ExitStack() as descriptors:
        workspace_descriptor, _workspace_identity = publisher_case[
            "active"
        ]._open_execution_workspace(descriptors)
        expected_commit = publisher_case[
            "checkpoint"
        ].safety_state.derivative_frontier.evidence.branch_heads[
            publisher_case["settings"].workspace_git_branch
        ]
        source_frontier = inspect_run_workspace_frontier(
            workspace_descriptor,
            settings=publisher_case["settings"],
            expected_commit_sha=expected_commit,
        )
        workspace_binding = RunActionWorkspaceBinding.from_identity(source_frontier)
        frontier = _remint_contract(
            initial_claim.reservation.frontier,
            workspace_before=workspace_binding,
        )
        reservation = _remint_contract(
            initial_claim.reservation,
            frontier=frontier,
            exact_dependency_ids=tuple(
                sorted(
                    (
                        frontier.frontier_binding_id
                        if dependency_id
                        == initial_claim.reservation.frontier.frontier_binding_id
                        else dependency_id
                    )
                    for dependency_id in initial_claim.reservation.exact_dependency_ids
                )
            ),
        )
        claim = RunActionPreparationClaim.mint(
            reservation=reservation,
            execution_policy=policy,
        )
        authority = _volume_authority(claim, nonce=_GENERATION_NONCE)
        volume = observe_runtime_volume(
            _volume_raw(authority, settings.docker),
            claim,
            authority,
            settings.docker,
        )
        empty = replace(
            empty_without_workspace,
            runtime_volume_authority=authority,
            docker_volume_observation=volume,
        )
        root_descriptor = _open_empty_root(root_path, descriptors)
        plan = _plan_runtime_volume_layout(
            claim,
            empty,
            workspace_descriptor=workspace_descriptor,
            settings=publisher_case["settings"],
        )

        copied_frontier = _materialize_layout_at_descriptor(
            root_descriptor,
            claim=claim,
            authority=authority,
            plan=plan,
            workspace_descriptor=workspace_descriptor,
            settings=publisher_case["settings"],
        )

    assert copied_frontier.source_tree_digest == source_frontier.source_tree_digest
    assert copied_frontier.git_closure_digest == source_frontier.git_closure_digest
    assert plan.workspace_copy_plan.physical_entry_count > (
        source_frontier.source_entry_count
    )
    assert (root_path / "workspace" / ".git" / "HEAD").is_file()
    assert (root_path / "credential" / "credentials").read_bytes() == b""
    assert (
        stat.S_IMODE((root_path / "credential" / "credentials").stat().st_mode) == 0o600
    )
