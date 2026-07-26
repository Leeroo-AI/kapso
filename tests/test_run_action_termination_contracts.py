"""Exhaustive outcome and graph joins for typed provider termination."""

from __future__ import annotations

from dataclasses import fields

import pytest

from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.launch.run_action_control_topology import (
    RunActionControlDirectoryTopology,
)
from kapso.cross_run.launch.run_action_supervisor_contracts import (
    RunActionPreparationAllocation,
)
from kapso.cross_run.launch.run_action_recovery import (
    RunActionProviderResult,
    RunActionRecoveryError,
)
from kapso.cross_run.launch.run_action_termination_contracts import (
    provider_termination_matches_durable_activation,
    run_action_pre_release_main_loss_observation_token,
    run_action_pre_release_main_terminal_observation_token,
    RunActionPreReleaseMainLossObservation,
    RunActionPreReleaseMainTerminalObservation,
    RunActionPreReleaseTerminalContainerObservation,
    RunActionProviderTerminationDisposition,
    RunActionProviderTerminationReason,
    RunActionProviderTerminationReceipt,
    RunActionTerminationContractError,
    RunActionTimeoutDirective,
    RunActionTimeoutDirectivePublicationReceipt,
    run_action_running_container_occurrence_matches,
    run_action_timeout_directive_evidence_matches,
    run_action_timeout_publication_evidence_matches,
)
from test_run_action_result_authority import _result_graph
from test_run_action_supervisor_contracts import (
    _remint_contract,
    _result_capture_receipt,
)


def _remint(contract, **changes):
    values = {
        field.name: getattr(contract, field.name)
        for field in fields(contract)
        if field.name != contract.IDENTITY_FIELD
    }
    values.update(changes)
    return type(contract).mint(**values)


def _timeout_publication(activation, adoption):
    release = adoption.workload_release_receipt
    control = activation.prepared_execution.control_directory
    authority = activation.prepared_execution.runtime_volume_authority
    running = _remint_contract(
        release.resolved_workload_observation.running_container_observation,
        complete_inspection_digest=tree_or_blob_digest(b"fresh timeout inspection"),
    )
    directive = RunActionTimeoutDirective.mint(
        activation_event_id=release.activation_event_id,
        workload_release_receipt_id=release.workload_release_receipt_id,
        workload_release_adoption_id=adoption.workload_release_adoption_id,
        host_boot_id=release.host_boot_id,
        execution_deadline_boottime_nanoseconds=(
            release.execution_deadline_boottime_nanoseconds
        ),
        containment_deadline_boottime_nanoseconds=(
            release.containment_deadline_boottime_nanoseconds
        ),
        observed_before_boottime_nanoseconds=(
            release.execution_deadline_boottime_nanoseconds
        ),
        running_container_observation=running,
        observed_after_boottime_nanoseconds=(
            release.execution_deadline_boottime_nanoseconds + 1
        ),
    )
    payload = directive.to_json_bytes()
    return RunActionTimeoutDirectivePublicationReceipt.mint(
        timeout_directive=directive,
        workload_release_adoption_id=adoption.workload_release_adoption_id,
        prepared_control_directory_id=control.prepared_runtime_directory_id,
        control_mount_id=control.mount_id,
        control_device=control.device,
        control_inode=control.inode,
        release_mount_id=adoption.release_mount_id,
        release_device=adoption.release_device,
        release_inode=adoption.release_inode,
        relative_path="control/timeout",
        file_type="regular",
        owner_user_id=authority.owner_user_id,
        owner_group_id=authority.owner_group_id,
        mode=0o400,
        link_count=1,
        size_bytes=len(payload),
        content_digest=tree_or_blob_digest(payload),
        timeout_mount_id=control.mount_id,
        timeout_device=control.device,
        timeout_inode=adoption.release_inode + 1,
    )


def _pre_release_loss(activation, activation_event_id):
    prepared = activation.prepared_execution
    inventory_digest = tree_or_blob_digest(b"stable pre-release inventory")
    allocation = RunActionPreparationAllocation.mint(
        preparation_claim=prepared.preparation_claim,
        runtime_volume_authority=prepared.runtime_volume_authority,
    )
    control = prepared.control_directory
    return RunActionPreReleaseMainLossObservation.mint(
        activation_event_id=activation_event_id,
        preparation_allocation=allocation,
        activation_revalidation_receipt=activation,
        host_boot_id="123e4567-e89b-42d3-a456-426614174000",
        observed_before_boottime_nanoseconds=80_000_000_000,
        first_complete_inventory_digest=inventory_digest,
        reobserved_volume_evidence=activation.reobserved_volume_evidence,
        reobserved_keeper_evidence=activation.reobserved_keeper_evidence,
        second_complete_inventory_digest=inventory_digest,
        observed_after_boottime_nanoseconds=80_000_000_001,
        observed_runtime_volume_names=(prepared.runtime_volume_authority.volume_name,),
        observed_keeper_container_ids=(prepared.volume_keeper_evidence.container_id,),
        observed_main_container_ids=(),
        missing_provider_execution_id=activation.spawn_commit.provider_execution_id,
        control_mount_id=control.mount_id,
        control_device=control.device,
        control_inode=control.inode,
        control_entry_count=0,
        control_directory_topology=RunActionControlDirectoryTopology.EMPTY,
    )


def _pre_release_terminal(
    activation,
    activation_event_id,
    released_terminal,
):
    prepared = activation.prepared_execution
    inventory_digest = tree_or_blob_digest(b"stable present terminal inventory")
    allocation = RunActionPreparationAllocation.mint(
        preparation_claim=prepared.preparation_claim,
        runtime_volume_authority=prepared.runtime_volume_authority,
    )
    control = prepared.control_directory
    terminal = RunActionPreReleaseTerminalContainerObservation.mint(
        prepared_execution_id=released_terminal.prepared_execution_id,
        spawn_commit_id=released_terminal.spawn_commit_id,
        provider_execution_id=released_terminal.provider_execution_id,
        runtime_volume_authority_id=(released_terminal.runtime_volume_authority_id),
        generation_nonce=released_terminal.generation_nonce,
        activation_revalidation_receipt_id=(
            released_terminal.activation_revalidation_receipt_id
        ),
        observed_inspect_projection=(released_terminal.observed_inspect_projection),
        complete_inspection_digest=released_terminal.complete_inspection_digest,
        container_status=released_terminal.container_status,
        process_id=released_terminal.process_id,
        restart_count=released_terminal.restart_count,
        paused=released_terminal.paused,
        restarting=released_terminal.restarting,
        dead=released_terminal.dead,
        started_at=released_terminal.started_at,
        finished_at=released_terminal.finished_at,
        exit_code=released_terminal.exit_code,
        oom_killed=released_terminal.oom_killed,
        state_error=released_terminal.state_error,
    )
    return RunActionPreReleaseMainTerminalObservation.mint(
        activation_event_id=activation_event_id,
        preparation_allocation=allocation,
        activation_revalidation_receipt=activation,
        host_boot_id="123e4567-e89b-42d3-a456-426614174000",
        observed_before_boottime_nanoseconds=80_000_000_000,
        first_complete_inventory_digest=inventory_digest,
        reobserved_volume_evidence=activation.reobserved_volume_evidence,
        reobserved_keeper_evidence=activation.reobserved_keeper_evidence,
        terminal_container_observation=terminal,
        second_complete_inventory_digest=inventory_digest,
        observed_after_boottime_nanoseconds=80_000_000_001,
        observed_runtime_volume_names=(prepared.runtime_volume_authority.volume_name,),
        observed_keeper_container_ids=(prepared.volume_keeper_evidence.container_id,),
        observed_main_container_ids=(activation.spawn_commit.provider_execution_id,),
        control_mount_id=control.mount_id,
        control_device=control.device,
        control_inode=control.inode,
        control_entry_count=0,
        control_directory_topology=RunActionControlDirectoryTopology.EMPTY,
    )


def _termination_graph(reason):
    activation, adoption, successful_terminal, _nonempty_capture = _result_graph()
    timeout = None
    capture = None
    loss = None
    if reason is RunActionProviderTerminationReason.TIMEOUT:
        terminal = successful_terminal
        timeout = _timeout_publication(activation, adoption)
        disposition = RunActionProviderTerminationDisposition.INTERRUPTED
    elif reason is RunActionProviderTerminationReason.OOM:
        terminal = _remint_contract(
            successful_terminal,
            exit_code=137,
            oom_killed=True,
        )
        disposition = RunActionProviderTerminationDisposition.FAILED
    elif reason is RunActionProviderTerminationReason.NONZERO_EXIT:
        terminal = _remint_contract(successful_terminal, exit_code=17)
        disposition = RunActionProviderTerminationDisposition.FAILED
    elif reason is RunActionProviderTerminationReason.EMPTY_RESULT:
        terminal = successful_terminal
        capture = _result_capture_receipt(
            activation.prepared_execution,
            activation,
            terminal,
            b"",
        )
        disposition = RunActionProviderTerminationDisposition.FAILED
    elif reason is RunActionProviderTerminationReason.PRE_RELEASE_MAIN_LOSS:
        adoption = None
        terminal = None
        activation_event_id = content_id(
            "run-action-execution-event",
            {"fixture": "pre-release activation event"},
        )
        loss = _pre_release_loss(activation, activation_event_id)
        disposition = RunActionProviderTerminationDisposition.FAILED
    else:
        adoption = None
        activation_event_id = content_id(
            "run-action-execution-event",
            {"fixture": "pre-release activation event"},
        )
        terminal = _pre_release_terminal(
            activation,
            activation_event_id,
            successful_terminal,
        )
        disposition = RunActionProviderTerminationDisposition.FAILED
    receipt = RunActionProviderTerminationReceipt.mint(
        disposition=disposition,
        reason=reason,
        activation_event_id=(
            activation_event_id
            if adoption is None
            else adoption.workload_release_receipt.activation_event_id
        ),
        workload_release_adoption=adoption,
        terminal_observation=terminal,
        timeout_directive_publication=timeout,
        empty_result_capture_receipt=capture,
        pre_release_main_loss_observation=loss,
    )
    return receipt


@pytest.mark.parametrize("reason", tuple(RunActionProviderTerminationReason))
def test_each_termination_branch_round_trips_with_one_exact_evidence_graph(reason):
    receipt = _termination_graph(reason)

    assert (
        RunActionProviderTerminationReceipt.from_json_bytes(receipt.to_json_bytes())
        == receipt
    )


@pytest.mark.parametrize(
    ("exit_code", "oom_killed"),
    (
        (0, False),
        (23, False),
        (137, True),
        (0, True),
    ),
)
def test_published_timeout_has_precedence_over_every_terminal_outcome(
    exit_code,
    oom_killed,
):
    receipt = _termination_graph(RunActionProviderTerminationReason.TIMEOUT)
    terminal = _remint_contract(
        receipt.terminal_observation,
        exit_code=exit_code,
        oom_killed=oom_killed,
    )

    reminted = _remint(receipt, terminal_observation=terminal)

    assert reminted.reason is RunActionProviderTerminationReason.TIMEOUT


@pytest.mark.parametrize("reason", tuple(RunActionProviderTerminationReason))
def test_reason_fixes_failed_or_interrupted_disposition(reason):
    receipt = _termination_graph(reason)
    wrong = (
        RunActionProviderTerminationDisposition.FAILED
        if receipt.disposition is RunActionProviderTerminationDisposition.INTERRUPTED
        else RunActionProviderTerminationDisposition.INTERRUPTED
    )

    with pytest.raises(
        RunActionTerminationContractError,
        match="disposition differs",
    ):
        _remint(receipt, disposition=wrong)


def test_timeout_requires_descriptor_publication_and_rejects_every_splice():
    receipt = _termination_graph(RunActionProviderTerminationReason.TIMEOUT)
    publication = receipt.timeout_directive_publication
    directive = publication.timeout_directive
    control = (
        receipt.activation_revalidation_receipt.prepared_execution.control_directory
    )

    with pytest.raises(
        RunActionTerminationContractError,
        match="published directive",
    ):
        _remint(receipt, timeout_directive_publication=None)

    wrong_deadline = _remint(
        directive,
        execution_deadline_boottime_nanoseconds=(
            directive.execution_deadline_boottime_nanoseconds + 1
        ),
        observed_before_boottime_nanoseconds=(
            directive.observed_before_boottime_nanoseconds + 1
        ),
        observed_after_boottime_nanoseconds=(
            directive.observed_after_boottime_nanoseconds + 1
        ),
    )
    with pytest.raises(
        RunActionTerminationContractError,
        match="published directive",
    ):
        _remint(
            receipt,
            timeout_directive_publication=_publication_with_directive(
                publication,
                wrong_deadline,
            ),
        )

    with pytest.raises(
        RunActionTerminationContractError,
        match="linked directive",
    ):
        _remint(
            publication,
            release_mount_id=publication.release_mount_id + 1,
        )

    wrong_running = _remint_contract(
        directive.running_container_observation,
        started_at="2026-07-25T01:02:02.123456789Z",
    )
    with pytest.raises(
        RunActionTerminationContractError,
        match="published directive",
    ):
        _remint(
            receipt,
            timeout_directive_publication=_publication_with_directive(
                publication,
                _remint(directive, running_container_observation=wrong_running),
            ),
        )

    with pytest.raises(
        RunActionTerminationContractError,
        match="published directive",
    ):
        _remint(
            receipt,
            timeout_directive_publication=_remint(
                publication,
                release_inode=control.inode + 2,
            ),
        )


def _publication_with_directive(publication, directive):
    payload = directive.to_json_bytes()
    return _remint(
        publication,
        timeout_directive=directive,
        size_bytes=len(payload),
        content_digest=tree_or_blob_digest(payload),
    )


def test_timeout_directive_requires_a_post_deadline_boottime_sandwich():
    receipt = _termination_graph(RunActionProviderTerminationReason.TIMEOUT)
    directive = receipt.timeout_directive_publication.timeout_directive

    with pytest.raises(
        RunActionTerminationContractError,
        match="deadline observation",
    ):
        _remint(
            directive,
            observed_before_boottime_nanoseconds=(
                directive.execution_deadline_boottime_nanoseconds - 1
            ),
        )
    with pytest.raises(
        RunActionTerminationContractError,
        match="deadline observation",
    ):
        _remint(
            directive,
            observed_after_boottime_nanoseconds=(
                directive.observed_before_boottime_nanoseconds - 1
            ),
        )


def test_timeout_directive_may_be_published_after_the_containment_deadline():
    receipt = _termination_graph(RunActionProviderTerminationReason.TIMEOUT)
    publication = receipt.timeout_directive_publication
    directive = publication.timeout_directive
    late_directive = _remint(
        directive,
        observed_before_boottime_nanoseconds=(
            directive.containment_deadline_boottime_nanoseconds + 1
        ),
        observed_after_boottime_nanoseconds=(
            directive.containment_deadline_boottime_nanoseconds + 2
        ),
    )
    late_publication = _publication_with_directive(
        publication,
        late_directive,
    )

    assert run_action_timeout_directive_evidence_matches(
        late_directive,
        receipt.activation_event_id,
        receipt.activation_revalidation_receipt,
        receipt.workload_release_adoption,
    )
    assert run_action_timeout_publication_evidence_matches(
        late_publication,
        receipt.activation_event_id,
        receipt.activation_revalidation_receipt,
        receipt.workload_release_adoption,
    )


def test_running_occurrence_matcher_ignores_fresh_inspection_identity_only():
    receipt = _termination_graph(RunActionProviderTerminationReason.TIMEOUT)
    directive = receipt.timeout_directive_publication.timeout_directive
    released = (
        receipt.workload_release_adoption.workload_release_receipt.resolved_workload_observation.running_container_observation
    )

    assert directive.running_container_observation != released
    assert run_action_running_container_occurrence_matches(
        directive.running_container_observation,
        released,
    )
    assert not run_action_running_container_occurrence_matches(
        _remint_contract(
            directive.running_container_observation,
            init_process_id=directive.running_container_observation.init_process_id + 1,
        ),
        released,
    )
    assert not run_action_running_container_occurrence_matches(object(), released)


def test_timeout_semantic_and_physical_matchers_compose_exactly():
    receipt = _termination_graph(RunActionProviderTerminationReason.TIMEOUT)
    publication = receipt.timeout_directive_publication
    activation = receipt.activation_revalidation_receipt
    adoption = receipt.workload_release_adoption
    activation_event_id = receipt.activation_event_id

    assert run_action_timeout_directive_evidence_matches(
        publication.timeout_directive,
        activation_event_id,
        activation,
        adoption,
    )
    assert run_action_timeout_publication_evidence_matches(
        publication,
        activation_event_id,
        activation,
        adoption,
    )
    wrong_running = _remint_contract(
        publication.timeout_directive.running_container_observation,
        started_at="2026-07-25T01:02:02.123456789Z",
    )
    wrong_directive = _remint(
        publication.timeout_directive,
        running_container_observation=wrong_running,
    )
    wrong_semantic_publication = _publication_with_directive(
        publication,
        wrong_directive,
    )
    assert not run_action_timeout_directive_evidence_matches(
        wrong_directive,
        activation_event_id,
        activation,
        adoption,
    )
    assert not run_action_timeout_publication_evidence_matches(
        wrong_semantic_publication,
        activation_event_id,
        activation,
        adoption,
    )
    physically_spliced = _remint(
        publication,
        release_inode=publication.release_inode + 2,
    )
    assert run_action_timeout_directive_evidence_matches(
        physically_spliced.timeout_directive,
        activation_event_id,
        activation,
        adoption,
    )
    assert not run_action_timeout_publication_evidence_matches(
        physically_spliced,
        activation_event_id,
        activation,
        adoption,
    )


def test_non_timeout_failures_reject_timeout_and_unrelated_capture_evidence():
    oom = _termination_graph(RunActionProviderTerminationReason.OOM)
    timeout = _termination_graph(
        RunActionProviderTerminationReason.TIMEOUT
    ).timeout_directive_publication
    empty = _termination_graph(RunActionProviderTerminationReason.EMPTY_RESULT)

    with pytest.raises(
        RunActionTerminationContractError,
        match="precedence",
    ):
        _remint(oom, timeout_directive_publication=timeout)
    with pytest.raises(
        RunActionTerminationContractError,
        match="terminal evidence",
    ):
        _remint(
            oom,
            empty_result_capture_receipt=empty.empty_result_capture_receipt,
        )


def test_oom_nonzero_and_empty_result_are_mutually_exclusive():
    oom = _termination_graph(RunActionProviderTerminationReason.OOM)
    nonzero = _termination_graph(RunActionProviderTerminationReason.NONZERO_EXIT)
    empty = _termination_graph(RunActionProviderTerminationReason.EMPTY_RESULT)

    with pytest.raises(
        RunActionTerminationContractError,
        match="terminal evidence",
    ):
        _remint(oom, reason=RunActionProviderTerminationReason.NONZERO_EXIT)
    with pytest.raises(
        RunActionTerminationContractError,
        match="terminal evidence",
    ):
        _remint(nonzero, reason=RunActionProviderTerminationReason.OOM)
    with pytest.raises(
        RunActionTerminationContractError,
        match="terminal evidence",
    ):
        _remint(empty, empty_result_capture_receipt=None)
    with pytest.raises(
        RunActionTerminationContractError,
        match="terminal evidence",
    ):
        _remint(
            empty,
            empty_result_capture_receipt=_result_graph()[3],
        )


def test_empty_capture_is_termination_evidence_not_a_provider_result():
    empty = _termination_graph(RunActionProviderTerminationReason.EMPTY_RESULT)

    with pytest.raises(
        RunActionRecoveryError,
        match="lacks exact terminal capture evidence",
    ):
        RunActionProviderResult(
            terminal_observation=empty.terminal_observation,
            result_capture_receipt=empty.empty_result_capture_receipt,
            result_payload=b"",
        )


def test_released_terminal_must_join_the_exact_activation_and_adoption():
    receipt = _termination_graph(RunActionProviderTerminationReason.NONZERO_EXIT)
    foreign_terminal = _remint_contract(
        receipt.terminal_observation,
        started_at="2026-07-25T01:02:02.123456789Z",
    )

    with pytest.raises(
        RunActionTerminationContractError,
        match="terminal occurrence",
    ):
        _remint(receipt, terminal_observation=foreign_terminal)
    with pytest.raises(
        RunActionTerminationContractError,
        match="terminal occurrence",
    ):
        _remint(
            receipt,
            activation_event_id=content_id(
                "run-action-execution-event",
                {"fixture": "another activation event"},
            ),
        )


def test_pre_release_loss_is_a_distinct_stable_absence_branch():
    receipt = _termination_graph(
        RunActionProviderTerminationReason.PRE_RELEASE_MAIN_LOSS
    )
    loss = receipt.pre_release_main_loss_observation
    released = _termination_graph(RunActionProviderTerminationReason.NONZERO_EXIT)

    with pytest.raises(
        RunActionTerminationContractError,
        match="sole termination evidence",
    ):
        _remint(
            receipt,
            workload_release_adoption=released.workload_release_adoption,
        )
    with pytest.raises(
        RunActionTerminationContractError,
        match="sole termination evidence",
    ):
        _remint(
            receipt,
            activation_event_id=content_id(
                "run-action-execution-event",
                {"fixture": "another pre-release activation event"},
            ),
        )
    with pytest.raises(
        RunActionTerminationContractError,
        match="incomplete or spliced",
    ):
        _remint(
            loss,
            second_complete_inventory_digest=tree_or_blob_digest(b"changed inventory"),
        )
    with pytest.raises(
        RunActionTerminationContractError,
        match="incomplete or spliced",
    ):
        _remint(
            loss,
            observed_main_container_ids=(loss.missing_provider_execution_id,),
        )
    with pytest.raises(
        RunActionTerminationContractError,
        match="incomplete or spliced",
    ):
        _remint(
            loss,
            control_directory_topology=RunActionControlDirectoryTopology.RELEASED,
        )
    with pytest.raises(
        RunActionTerminationContractError,
        match="incomplete or spliced",
    ):
        _remint(loss, control_inode=loss.control_inode + 1)


def test_pre_release_loss_allows_mutable_usage_on_the_same_volume_occurrence():
    receipt = _termination_graph(
        RunActionProviderTerminationReason.PRE_RELEASE_MAIN_LOSS
    )
    loss = receipt.pre_release_main_loss_observation
    volume = loss.reobserved_volume_evidence
    changed_volume = _remint_contract(
        volume,
        used_block_count=volume.used_block_count + 1,
        used_size_bytes=(volume.used_size_bytes + volume.allocation_block_size_bytes),
        available_block_count=volume.available_block_count - 1,
        available_size_bytes=(
            volume.available_size_bytes - volume.allocation_block_size_bytes
        ),
    )

    reminted_loss = _remint(loss, reobserved_volume_evidence=changed_volume)

    assert reminted_loss.reobserved_volume_evidence == changed_volume


def test_pre_release_loss_token_ignores_sampling_time_and_mutable_usage():
    receipt = _termination_graph(
        RunActionProviderTerminationReason.PRE_RELEASE_MAIN_LOSS
    )
    loss = receipt.pre_release_main_loss_observation
    volume = loss.reobserved_volume_evidence
    changed_volume = _remint_contract(
        volume,
        used_block_count=volume.used_block_count + 1,
        used_size_bytes=(volume.used_size_bytes + volume.allocation_block_size_bytes),
        available_block_count=volume.available_block_count - 1,
        available_size_bytes=(
            volume.available_size_bytes - volume.allocation_block_size_bytes
        ),
    )
    later_observation = _remint(
        loss,
        observed_before_boottime_nanoseconds=(
            loss.observed_before_boottime_nanoseconds + 100
        ),
        observed_after_boottime_nanoseconds=(
            loss.observed_after_boottime_nanoseconds + 100
        ),
        reobserved_volume_evidence=changed_volume,
    )

    assert later_observation != loss
    assert run_action_pre_release_main_loss_observation_token(
        later_observation
    ) == run_action_pre_release_main_loss_observation_token(loss)


def test_pre_release_loss_token_changes_with_the_physical_occurrence():
    receipt = _termination_graph(
        RunActionProviderTerminationReason.PRE_RELEASE_MAIN_LOSS
    )
    loss = receipt.pre_release_main_loss_observation
    changed_inventory_digest = tree_or_blob_digest(b"another stable inventory")
    changed_inventory = _remint(
        loss,
        first_complete_inventory_digest=changed_inventory_digest,
        second_complete_inventory_digest=changed_inventory_digest,
    )

    assert run_action_pre_release_main_loss_observation_token(
        changed_inventory
    ) != run_action_pre_release_main_loss_observation_token(loss)


def test_pre_release_terminal_is_distinct_from_loss_and_released_terminal():
    receipt = _termination_graph(
        RunActionProviderTerminationReason.PRE_RELEASE_MAIN_TERMINAL
    )
    terminal = receipt.terminal_observation
    loss = _termination_graph(RunActionProviderTerminationReason.PRE_RELEASE_MAIN_LOSS)
    released = _termination_graph(RunActionProviderTerminationReason.NONZERO_EXIT)

    assert type(terminal) is RunActionPreReleaseMainTerminalObservation
    assert provider_termination_matches_durable_activation(
        receipt,
        receipt.activation_event_id,
        terminal.preparation_allocation,
        terminal.activation_revalidation_receipt,
    )
    with pytest.raises(
        RunActionTerminationContractError,
        match="sole termination evidence",
    ):
        _remint(receipt, terminal_observation=released.terminal_observation)
    with pytest.raises(
        RunActionTerminationContractError,
        match="sole termination evidence",
    ):
        _remint(
            receipt,
            pre_release_main_loss_observation=(loss.pre_release_main_loss_observation),
        )
    with pytest.raises(
        RunActionTerminationContractError,
        match="sole termination evidence",
    ):
        _remint(receipt, workload_release_adoption=released.workload_release_adoption)


def test_pre_release_terminal_rejects_inventory_control_and_main_splices():
    receipt = _termination_graph(
        RunActionProviderTerminationReason.PRE_RELEASE_MAIN_TERMINAL
    )
    observation = receipt.terminal_observation

    with pytest.raises(
        RunActionTerminationContractError,
        match="incomplete or spliced",
    ):
        _remint(
            observation,
            second_complete_inventory_digest=tree_or_blob_digest(b"changed inventory"),
        )
    with pytest.raises(
        RunActionTerminationContractError,
        match="incomplete or spliced",
    ):
        _remint(observation, observed_main_container_ids=())
    with pytest.raises(
        RunActionTerminationContractError,
        match="incomplete or spliced",
    ):
        _remint(
            observation,
            control_directory_topology=RunActionControlDirectoryTopology.RELEASED,
        )
    with pytest.raises(
        RunActionTerminationContractError,
        match="incomplete or spliced",
    ):
        _remint(
            observation,
            terminal_container_observation=_remint(
                observation.terminal_container_observation,
                provider_execution_id="e" * 64,
            ),
        )


def test_pre_release_terminal_token_ignores_sampling_time_and_usage_only():
    receipt = _termination_graph(
        RunActionProviderTerminationReason.PRE_RELEASE_MAIN_TERMINAL
    )
    observation = receipt.terminal_observation
    volume = observation.reobserved_volume_evidence
    changed_volume = _remint_contract(
        volume,
        used_block_count=volume.used_block_count + 1,
        used_size_bytes=(volume.used_size_bytes + volume.allocation_block_size_bytes),
        available_block_count=volume.available_block_count - 1,
        available_size_bytes=(
            volume.available_size_bytes - volume.allocation_block_size_bytes
        ),
    )
    later = _remint(
        observation,
        observed_before_boottime_nanoseconds=(
            observation.observed_before_boottime_nanoseconds + 100
        ),
        observed_after_boottime_nanoseconds=(
            observation.observed_after_boottime_nanoseconds + 100
        ),
        reobserved_volume_evidence=changed_volume,
    )
    changed_terminal = _remint(
        observation,
        terminal_container_observation=_remint(
            observation.terminal_container_observation,
            complete_inspection_digest=tree_or_blob_digest(
                b"another terminal snapshot"
            ),
        ),
    )

    assert run_action_pre_release_main_terminal_observation_token(
        later
    ) == run_action_pre_release_main_terminal_observation_token(observation)
    assert run_action_pre_release_main_terminal_observation_token(
        changed_terminal
    ) != run_action_pre_release_main_terminal_observation_token(observation)
