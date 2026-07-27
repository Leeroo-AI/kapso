"""Reservation-only current-frontier action gate tests."""

from __future__ import annotations

import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.launch.checkpoint_contracts import (
    RunCheckpoint,
    RunCheckpointStatus,
    RunCheckpointStop,
)
from kapso.cross_run.launch.derived_state_contracts import RunStateAuthority
from kapso.cross_run.launch.resume_contracts import (
    RunDerivativeFrontier,
    RunReleaseUseMode,
    RunSafetyBoundary,
    RunSafetyState,
)
from kapso.cross_run.launch.run_action_contracts import (
    RunActionBoundaryIdentity,
    RunActionContractError,
    RunActionExecutionLifecycleIdentity,
    RunActionResultInterpreterIdentity,
)
from kapso.cross_run.launch.run_action_credential_broker import (
    RunActionCredentialBrokerBackend,
    RunActionCredentialBrokerRegistry,
    RunActionCredentialIssueResponse,
    RunActionCredentialLeaseStatus,
)
from kapso.cross_run.launch.run_action_gate import (
    RunFrontierActionError,
    RunFrontierActionGate,
    RunFrontierActionKind,
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_ledger import RunActionExecutionEventKind
from kapso.cross_run.launch.run_action_reservation_contracts import (
    RunActionReservation,
)
from kapso.cross_run.launch.run_action_resource_finalization import (
    _issue_run_action_resource_finalization_authority,
)
from kapso.cross_run.launch.run_state_publisher import (
    RunStatePublisher,
    RunStatePublisherError,
)
from kapso.cross_run.launch.workspace_frontier import RunWorkspaceFrontierError
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
)
from test_launch_resume_contracts import (
    _remint_evidence,
    _security_observation,
    _subjects,
)
from test_launch_resolver import resolver_case
from test_run_action_supervisor_contracts import _execution_policy
from test_run_state_publisher import _publish_genesis, publisher_case


def _boundary_identity(
    kind: RunFrontierActionKind,
    workspace_access: RunFrontierWorkspaceAccess | None = None,
) -> RunActionBoundaryIdentity:
    if workspace_access is None:
        workspace_access = (
            RunFrontierWorkspaceAccess.NONE
            if kind is RunFrontierActionKind.EMBEDDING
            else RunFrontierWorkspaceAccess.READ_ONLY
        )
    policy = _execution_policy(
        kind=kind,
        workspace_access=workspace_access,
    )
    return RunActionBoundaryIdentity.mint(
        kind=kind,
        execution_lifecycle_identity=RunActionExecutionLifecycleIdentity.mint(
            kind=kind,
            implementation_id=f"test.{kind.value}.execution",
            implementation_version="test.execution.v1",
            recovery_protocol_version="test.recovery.v1",
            execution_policy_id=policy.docker_execution_policy_id,
        ),
        result_interpreter_identity=RunActionResultInterpreterIdentity.mint(
            kind=kind,
            implementation_id=f"test.{kind.value}.interpreter",
            implementation_version="test.interpreter.v1",
            interpretation_protocol_version="test.interpretation.v1",
            interpretation_policy_id=content_id(
                "test-run-action-interpretation-policy",
                {"kind": kind.value},
            ),
        ),
    )


class _StaticSecurityAuthority:
    def __init__(self, observation: SecurityDenylistObservation) -> None:
        self.observation = observation
        self.calls = []

    def observe_exact_descendant_of(
        self,
        *,
        scope_id,
        scope_contract_id,
        checked_subject_ids,
        required_ancestor,
    ):
        self.calls.append(
            (
                scope_id,
                scope_contract_id,
                checked_subject_ids,
                required_ancestor,
            )
        )
        return self.observation


class _StaticCredentialBroker(RunActionCredentialBrokerBackend):
    def __init__(self):
        super().__init__(
            broker_id="test.credential.broker",
            broker_protocol_version="test.credential.broker.v1",
        )
        self.issue_calls = []
        self.status_calls = []
        self._lease_expiries = {}

    def _valid_until(self, request):
        existing = self._lease_expiries.get(request.credential_lease_request_id)
        if existing is not None:
            return existing
        valid_until = (
            time.time_ns()
            + (request.credential_policy.maximum_lease_seconds - 1) * 1_000_000_000
        )
        self._lease_expiries[request.credential_lease_request_id] = valid_until
        return valid_until

    def issue_or_replay_exact(self, request):
        self.issue_calls.append(request)
        return RunActionCredentialIssueResponse(
            credential_lease_request_id=request.credential_lease_request_id,
            payload=b"credential bytes",
            valid_until_realtime_nanoseconds=self._valid_until(request),
        )

    def observe_exact(self, request):
        self.status_calls.append(request)
        return RunActionCredentialLeaseStatus.mint(
            credential_lease_request_id=request.credential_lease_request_id,
            valid_until_realtime_nanoseconds=self._valid_until(request),
        )


def _credential_broker_registry():
    backend = _StaticCredentialBroker()
    return RunActionCredentialBrokerRegistry((backend,)), backend


class _StaticRunActionResourceFinalizationDriver:
    def __init__(self) -> None:
        self.finalized_operation_ids = []
        self.absence_checked_operation_ids = []
        self.block_finalization = False
        self.block_absence = False

    def finalize_terminal(self, operation_id):
        self.finalized_operation_ids.append(operation_id)
        if self.block_finalization:
            raise RuntimeError("terminal resources remain")

    def require_terminal_absence(self, operation_id):
        self.absence_checked_operation_ids.append(operation_id)
        if self.block_absence:
            raise RuntimeError("terminal resources remain")


def _static_resource_finalization_authority(publisher):
    return _issue_run_action_resource_finalization_authority(
        action_store=publisher._action_store,
        launch_settings=publisher._settings,
        driver=_StaticRunActionResourceFinalizationDriver(),
    )


def _successor_at_boundary(
    case,
    publisher,
    predecessor_receipt,
    boundary,
    *,
    last_stop=None,
    derivative_frontier=None,
):
    predecessor = predecessor_receipt.checkpoint
    pin = case["active"].bootstrap_pin
    derivative_frontier = (
        predecessor.safety_state.derivative_frontier
        if derivative_frontier is None
        else derivative_frontier
    )
    projection = replace(
        case["projection"],
        action_ledger=publisher.action_ledger_snapshot(),
    )
    action_ledger_payload = projection.action_ledger.to_json_bytes()
    evidence = _remint_evidence(
        derivative_frontier.evidence,
        state_authority_digests={
            **derivative_frontier.evidence.state_authority_digests,
            RunStateAuthority.ACTION_LEDGER.value: tree_or_blob_digest(
                action_ledger_payload
            ),
        },
        state_authority_revisions={
            **derivative_frontier.evidence.state_authority_revisions,
            RunStateAuthority.ACTION_LEDGER.value: (
                projection.action_ledger.event_count
            ),
        },
    )
    derivative_frontier = RunDerivativeFrontier.build(
        launch_subject_ids=derivative_frontier.launch_subject_ids,
        evidence=evidence,
        derivatives=derivative_frontier.derivatives,
    )
    release_use = predecessor.safety_state.release_use_observation
    safety = RunSafetyState.build(
        predecessor=predecessor.safety_state,
        bootstrap_pin=pin,
        boundary=boundary,
        derivative_frontier=derivative_frontier,
        security_observation=_security_observation(
            pin,
            _subjects(
                pin,
                release_use,
                derivative_frontier,
                predecessor.safety_state,
            ),
            generation_offset=(
                predecessor.safety_state.security_observation.generation
                - pin.launch_manifest.security_observation.generation
                + 1
            ),
        ),
        release_use_observation=release_use,
        release_use_mode=RunReleaseUseMode.ONLINE_CURRENT,
    )
    bundle = projection.build_bundle(
        bootstrap_pin=pin,
        run_state_layout=predecessor.derived_state_generation.run_state_layout,
        predecessor_checkpoint_head_id=predecessor_receipt.journal_head_id,
        predecessor_checkpoint_id=predecessor.run_checkpoint_id,
        predecessor_evidence_id=(
            predecessor.safety_state.derivative_frontier.evidence.evidence_id
        ),
        target_evidence_id=safety.derivative_frontier.evidence.evidence_id,
        predecessor_bundle=predecessor_receipt.bundle,
        predecessor_strategy_state=predecessor.strategy_state,
    )
    checkpoint = RunCheckpoint.build(
        predecessor=predecessor,
        status=RunCheckpointStatus.ACTIVE,
        last_stop=last_stop,
        completed_iterations=predecessor.completed_iterations,
        cumulative_cost=predecessor.cumulative_cost,
        elapsed_seconds=predecessor.elapsed_seconds,
        cost_by_component=predecessor.cost_by_component,
        feedback_source=predecessor.feedback_source,
        current_feedback=predecessor.current_feedback,
        termination_decision=None,
        strategy_state=predecessor.strategy_state,
        safety_state=safety,
        derived_state_generation=bundle.generation,
    )
    return bundle, checkpoint


def _action_case(
    case,
    boundary=RunSafetyBoundary.IDEATION,
    credential_broker_registry=None,
    resource_finalization_authority_factory=None,
):
    publisher, initial = _publish_genesis(case)
    bundle, checkpoint = _successor_at_boundary(
        case,
        publisher,
        initial,
        boundary,
    )
    receipt = publisher.publish(
        publisher.issue_publication_permit(initial, checkpoint, bundle),
        checkpoint,
        bundle,
    )
    security = _StaticSecurityAuthority(
        receipt.checkpoint.safety_state.security_observation
    )
    resource_finalization_authority = (
        _static_resource_finalization_authority(publisher)
        if resource_finalization_authority_factory is None
        else resource_finalization_authority_factory(publisher)
    )
    if credential_broker_registry is None:
        credential_broker_registry, _credential_backend = _credential_broker_registry()
    gate = RunFrontierActionGate(
        active_workspace=case["active"],
        publisher=publisher,
        security_authority=security,
        credential_broker_registry=credential_broker_registry,
        resource_finalization_authority=resource_finalization_authority,
    )
    return publisher, receipt, security, gate


def _reserve_ideation_agent(
    gate,
    receipt,
    payload=b'{"prompt":"complete"}',
):
    return gate.reserve(
        receipt,
        kind=RunFrontierActionKind.CODING_AGENT,
        boundary=RunSafetyBoundary.IDEATION,
        operation_id="agent_call_0123456789abcdef0123456789abcdef",
        request_payload=payload,
        workspace_access=RunFrontierWorkspaceAccess.READ_ONLY,
        boundary_identity=_boundary_identity(
            RunFrontierActionKind.CODING_AGENT,
            RunFrontierWorkspaceAccess.READ_ONLY,
        ),
    )


def _reserve_implementation_agent(
    gate,
    receipt,
    operation_suffix,
    payload=b'{"implementation":"complete"}',
):
    return gate.reserve(
        receipt,
        kind=RunFrontierActionKind.CODING_AGENT,
        boundary=RunSafetyBoundary.IMPLEMENTATION,
        operation_id=f"implementation_{operation_suffix}_0123456789abcdef",
        request_payload=payload,
        workspace_access=RunFrontierWorkspaceAccess.EDIT_WORKSPACE,
        boundary_identity=_boundary_identity(
            RunFrontierActionKind.CODING_AGENT,
            RunFrontierWorkspaceAccess.EDIT_WORKSPACE,
        ),
    )


def _commit_workspace_edit(case, filename, payload):
    workspace = case["active"].workspace
    (workspace / filename).write_text(payload, encoding="utf-8")
    _run_git(workspace, "add", "--", filename)
    _run_git(
        workspace,
        "-c",
        "user.name=Kapso Test",
        "-c",
        "user.email=kapso-test@example.invalid",
        "commit",
        "-m",
        f"Edit {filename}",
    )
    return _run_git(workspace, "rev-parse", "HEAD").strip()


def _run_git(workspace, *arguments):
    process = subprocess.Popen(
        [
            "git",
            "-C",
            str(workspace),
            *arguments,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate()
    assert process.returncode == 0, stderr
    return stdout


def test_gate_reserves_exact_event_one_and_complete_request(
    publisher_case,
) -> None:
    publisher, frontier, security, gate = _action_case(publisher_case)
    payload = b'{"prompt":"complete untruncated request"}'

    reservation = _reserve_ideation_agent(gate, frontier, payload)

    events = gate._action_store.inspect().events_for(reservation.intent.operation_id)
    assert type(reservation) is RunActionReservation
    assert len(events) == 1
    assert events[0].event_kind is RunActionExecutionEventKind.INTENT_RESERVED
    assert events[0].reservation == reservation
    assert reservation.request_blob.size_bytes == len(payload)
    assert publisher.action_ledger_snapshot().event_count == 1
    assert not security.calls


def test_reservation_rejects_boundary_kind_substitution(
    publisher_case,
) -> None:
    publisher, frontier, _security, gate = _action_case(publisher_case)
    before = publisher.action_ledger_snapshot()

    with pytest.raises(RunFrontierActionError, match="unrecognized enum"):
        gate.reserve(
            frontier,
            kind=RunFrontierActionKind.CODING_AGENT,
            boundary=RunSafetyBoundary.IDEATION,
            operation_id="wrong_adapter_kind_0123456789abcdef",
            request_payload=b'{"prompt":"complete"}',
            workspace_access=RunFrontierWorkspaceAccess.READ_ONLY,
            boundary_identity=_boundary_identity(RunFrontierActionKind.EMBEDDING),
        )

    assert publisher.action_ledger_snapshot() == before


def test_boundary_rejects_cross_kind_lifecycle_or_interpreter() -> None:
    coding_boundary = _boundary_identity(RunFrontierActionKind.CODING_AGENT)
    embedding_boundary = _boundary_identity(RunFrontierActionKind.EMBEDDING)

    with pytest.raises(RunActionContractError, match="components"):
        RunActionBoundaryIdentity.mint(
            kind=RunFrontierActionKind.CODING_AGENT,
            execution_lifecycle_identity=(
                embedding_boundary.execution_lifecycle_identity
            ),
            result_interpreter_identity=(coding_boundary.result_interpreter_identity),
        )
    with pytest.raises(RunActionContractError, match="components"):
        RunActionBoundaryIdentity.mint(
            kind=RunFrontierActionKind.CODING_AGENT,
            execution_lifecycle_identity=(coding_boundary.execution_lifecycle_identity),
            result_interpreter_identity=(
                embedding_boundary.result_interpreter_identity
            ),
        )


def test_reservation_rejects_wrong_checkpoint_boundary(
    publisher_case,
) -> None:
    _publisher, frontier, _security, gate = _action_case(publisher_case)

    with pytest.raises(RunFrontierActionError, match="boundary"):
        gate.reserve(
            frontier,
            kind=RunFrontierActionKind.EVALUATOR,
            boundary=RunSafetyBoundary.EVALUATION,
            operation_id="evaluation_0123456789abcdef",
            request_payload=b'{"evaluation":"full"}',
            workspace_access=RunFrontierWorkspaceAccess.READ_ONLY,
            boundary_identity=_boundary_identity(RunFrontierActionKind.EVALUATOR),
        )


def test_unresolved_reservation_blocks_checkpoint_publication(
    publisher_case,
) -> None:
    publisher, frontier, _security, gate = _action_case(publisher_case)
    _reserve_ideation_agent(gate, frontier)
    successor_bundle, successor_checkpoint = _successor_at_boundary(
        publisher_case,
        publisher,
        frontier,
        RunSafetyBoundary.IMPLEMENTATION,
    )

    with pytest.raises(
        RunStatePublisherError,
        match="unresolved execution",
    ):
        publisher.issue_publication_permit(
            frontier,
            successor_checkpoint,
            successor_bundle,
        )


def test_stopped_checkpoint_cannot_reserve_an_action(
    publisher_case,
) -> None:
    publisher, initial = _publish_genesis(publisher_case)
    bundle, stopped = _successor_at_boundary(
        publisher_case,
        publisher,
        initial,
        RunSafetyBoundary.IDEATION,
        last_stop=RunCheckpointStop.COST_BUDGET,
    )
    receipt = publisher.publish(
        publisher.issue_publication_permit(initial, stopped, bundle),
        stopped,
        bundle,
    )
    gate = RunFrontierActionGate(
        active_workspace=publisher_case["active"],
        publisher=publisher,
        security_authority=_StaticSecurityAuthority(
            receipt.checkpoint.safety_state.security_observation
        ),
        credential_broker_registry=_credential_broker_registry()[0],
        resource_finalization_authority=(
            _static_resource_finalization_authority(publisher)
        ),
    )

    with pytest.raises(RunFrontierActionError, match="stopped or completed"):
        _reserve_ideation_agent(gate, receipt)


def test_completed_checkpoint_is_not_actionable(
    publisher_case,
) -> None:
    _publisher, frontier, _security, gate = _action_case(publisher_case)
    reservation = _reserve_ideation_agent(gate, frontier)
    completed = object.__new__(RunCheckpoint)
    for field_name, value in vars(frontier.checkpoint).items():
        object.__setattr__(completed, field_name, value)
    object.__setattr__(
        completed,
        "status",
        RunCheckpointStatus.COMPLETED,
    )

    with pytest.raises(RunFrontierActionError, match="stopped or completed"):
        gate._require_actionable(completed, reservation.intent)


@pytest.mark.parametrize(
    ("boundary", "kind", "access"),
    (
        (
            RunSafetyBoundary.IDEATION,
            RunFrontierActionKind.CODING_AGENT,
            RunFrontierWorkspaceAccess.EDIT_WORKSPACE,
        ),
        (
            RunSafetyBoundary.IMPLEMENTATION,
            RunFrontierActionKind.CODING_AGENT,
            RunFrontierWorkspaceAccess.READ_ONLY,
        ),
        (
            RunSafetyBoundary.EVALUATION,
            RunFrontierActionKind.CODING_AGENT,
            RunFrontierWorkspaceAccess.EDIT_WORKSPACE,
        ),
        (
            RunSafetyBoundary.IDEATION,
            RunFrontierActionKind.EMBEDDING,
            RunFrontierWorkspaceAccess.READ_ONLY,
        ),
        (
            RunSafetyBoundary.EVALUATION,
            RunFrontierActionKind.EVALUATOR,
            RunFrontierWorkspaceAccess.EDIT_WORKSPACE,
        ),
    ),
)
def test_reservation_enforces_exact_capability_matrix(
    publisher_case,
    boundary,
    kind,
    access,
) -> None:
    _publisher, frontier, _security, gate = _action_case(
        publisher_case,
        boundary,
    )

    with pytest.raises(RunFrontierActionError, match="workspace access"):
        gate.reserve(
            frontier,
            kind=kind,
            boundary=boundary,
            operation_id="forbidden_action_0123456789abcdef",
            request_payload=b'{"request":"complete"}',
            workspace_access=access,
            boundary_identity=_boundary_identity(kind),
        )


def test_embedding_reservation_has_no_workspace_binding(
    publisher_case,
) -> None:
    _publisher, frontier, _security, gate = _action_case(publisher_case)
    reservation = gate.reserve(
        frontier,
        kind=RunFrontierActionKind.EMBEDDING,
        boundary=RunSafetyBoundary.IDEATION,
        operation_id="embedding_0123456789abcdef",
        request_payload=b'{"texts":["complete input"]}',
        workspace_access=RunFrontierWorkspaceAccess.NONE,
        boundary_identity=_boundary_identity(RunFrontierActionKind.EMBEDDING),
    )

    assert reservation.frontier.workspace_before is None


def test_duplicate_intent_and_operation_are_reserved_once(
    publisher_case,
) -> None:
    _publisher, frontier, _security, gate = _action_case(publisher_case)
    payload = b'{"prompt":"complete"}'
    first = _reserve_ideation_agent(gate, frontier, payload)

    with pytest.raises(RunFrontierActionError, match="unresolved durable action"):
        _reserve_ideation_agent(gate, frontier, payload)
    with pytest.raises(RunFrontierActionError, match="unresolved durable action"):
        gate.reserve(
            frontier,
            kind=RunFrontierActionKind.CODING_AGENT,
            boundary=RunSafetyBoundary.IDEATION,
            operation_id=first.intent.operation_id,
            request_payload=b'{"prompt":"another complete request"}',
            workspace_access=RunFrontierWorkspaceAccess.READ_ONLY,
            boundary_identity=_boundary_identity(RunFrontierActionKind.CODING_AGENT),
        )


def test_concurrent_duplicate_reservation_has_one_winner(
    publisher_case,
) -> None:
    _publisher, frontier, _security, gate = _action_case(publisher_case)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(
            executor.submit(_reserve_ideation_agent, gate, frontier)
            for _attempt in range(2)
        )
    reservations = []
    errors = []
    for future in futures:
        if future.exception() is None:
            reservations.append(future.result())
        else:
            errors.append(future.exception())

    assert len(reservations) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], RunFrontierActionError)
    assert (
        "unresolved durable action" in str(errors[0])
        or "predecessor ledger moved" in str(errors[0])
        or "already reserved" in str(errors[0])
        or "live session" in str(errors[0])
    )


def test_reconstructed_gate_observes_unresolved_reservation(
    publisher_case,
) -> None:
    publisher, frontier, security, gate = _action_case(publisher_case)
    _reserve_ideation_agent(gate, frontier)
    reconstructed_publisher = RunStatePublisher(
        publisher_case["active"],
        publisher_case["settings"],
    )
    reconstructed_gate = RunFrontierActionGate(
        active_workspace=publisher_case["active"],
        publisher=reconstructed_publisher,
        security_authority=security,
        credential_broker_registry=_credential_broker_registry()[0],
        resource_finalization_authority=(
            _static_resource_finalization_authority(reconstructed_publisher)
        ),
    )
    reconstructed_frontier = reconstructed_publisher.load_reconciled()

    with pytest.raises(
        RunFrontierActionError,
        match="unresolved durable action",
    ):
        _reserve_ideation_agent(
            reconstructed_gate,
            reconstructed_frontier,
            b'{"prompt":"different complete request"}',
        )


def test_dirty_workspace_cannot_be_reserved(
    publisher_case,
) -> None:
    _publisher, frontier, _security, gate = _action_case(publisher_case)
    (publisher_case["active"].workspace / "uncommitted.py").write_text(
        "DIRTY = True\n",
        encoding="utf-8",
    )

    with pytest.raises(RunWorkspaceFrontierError):
        _reserve_ideation_agent(gate, frontier)


def test_gate_cannot_cross_process_boundary(
    publisher_case,
) -> None:
    _publisher, frontier, _security, gate = _action_case(publisher_case)
    read_descriptor, write_descriptor = os.pipe()
    child_process_id = os.fork()
    if child_process_id == 0:
        os.close(read_descriptor)
        with pytest.raises(RunFrontierActionError, match="process boundary"):
            _reserve_ideation_agent(gate, frontier)
        os.write(write_descriptor, b"invalid")
        os._exit(0)
    os.close(write_descriptor)
    assert os.read(read_descriptor, len(b"invalid")) == b"invalid"
    os.close(read_descriptor)
    waited_process_id, status = os.waitpid(child_process_id, 0)
    assert waited_process_id == child_process_id
    assert os.waitstatus_to_exitcode(status) == 0


def test_direct_gate_lifecycle_is_absent() -> None:
    for name in (
        "issue",
        "hold",
        "allocate_preparation",
        "commit_prepared_execution",
        "claim_activation",
        "commit_activation",
        "record_result",
        "accept_result",
        "invalidate_frontier",
    ):
        assert not hasattr(RunFrontierActionGate, name)
