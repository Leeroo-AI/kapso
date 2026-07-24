"""Current-frontier action permits and shared publication exclusion."""

from __future__ import annotations

import gc
import hashlib
import os
import subprocess
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import replace

import pytest

from kapso.cross_run.canonical import tree_or_blob_digest
from kapso.cross_run.launch.checkpoint_contracts import (
    RunCheckpoint,
    RunCheckpointStatus,
    RunCheckpointStop,
)
from kapso.cross_run.launch.derived_state_contracts import RunStateAuthority
from kapso.cross_run.launch.resume_contracts import (
    RunBranchAdvance,
    RunDerivativeFrontier,
    RunReleaseUseMode,
    RunSafetyBoundary,
    RunSafetyState,
)
from kapso.cross_run.launch.run_action_contracts import (
    RunActionBoundaryIdentity,
    RunActionContractError,
    RunActionExecutionLifecycleIdentity,
    RunActionIntent,
    RunActionResultInterpreterIdentity,
)
from kapso.cross_run.launch.run_action_gate import (
    RunFrontierActionError,
    RunFrontierActionGate,
    RunFrontierActionKind,
    RunFrontierWorkspaceAccess,
)
from kapso.cross_run.launch.run_action_store import (
    RunActionAcceptance,
    RunActionExecutionEvent,
    RunActionExecutionEventKind,
    RunActionReservation,
    RunActionResultDisposition,
    RunActionResultReceipt,
    RunActionSpawnCommit,
    RunActionStoreError,
)
from kapso.cross_run.launch.run_state_publisher import (
    RunStatePublisher,
    RunStatePublisherError,
)
from kapso.cross_run.launch.workspace_frontier import RunWorkspaceFrontierError
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
)
from test_launch_resolver import resolver_case
from test_launch_resume_contracts import (
    _remint_evidence,
    _security_observation,
    _subjects,
)
from test_run_state_publisher import _publish_genesis, publisher_case


def _boundary_identity(
    kind: RunFrontierActionKind,
) -> RunActionBoundaryIdentity:
    return RunActionBoundaryIdentity.mint(
        kind=kind,
        execution_lifecycle_identity=RunActionExecutionLifecycleIdentity.mint(
            kind=kind,
            implementation_id=f"test.{kind.value}.execution",
            implementation_version="test.execution.v1",
            recovery_protocol_version="test.recovery.v1",
            sandbox_policy_id=f"test.{kind.value}.sandbox.v1",
        ),
        result_interpreter_identity=RunActionResultInterpreterIdentity.mint(
            kind=kind,
            implementation_id=f"test.{kind.value}.interpreter",
            implementation_version="test.interpreter.v1",
            interpretation_protocol_version="test.interpretation.v1",
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


def _action_case(case, boundary=RunSafetyBoundary.IDEATION):
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
    gate = RunFrontierActionGate(
        active_workspace=case["active"],
        publisher=publisher,
        security_authority=security,
    )
    return publisher, receipt, security, gate


def _issue_ideation_agent(gate, receipt, payload=b'{"prompt":"complete"}'):
    return gate.issue(
        receipt,
        kind=RunFrontierActionKind.CODING_AGENT,
        boundary=RunSafetyBoundary.IDEATION,
        operation_id="agent_call_0123456789abcdef0123456789abcdef",
        request_payload=payload,
        workspace_access=RunFrontierWorkspaceAccess.READ_ONLY,
        boundary_identity=_boundary_identity(RunFrontierActionKind.CODING_AGENT),
    )


def _issue_implementation_agent(
    gate,
    receipt,
    operation_suffix,
    payload=b'{"implementation":"complete"}',
):
    return gate.issue(
        receipt,
        kind=RunFrontierActionKind.CODING_AGENT,
        boundary=RunSafetyBoundary.IMPLEMENTATION,
        operation_id=f"implementation_{operation_suffix}_0123456789abcdef",
        request_payload=payload,
        workspace_access=RunFrontierWorkspaceAccess.EDIT_WORKSPACE,
        boundary_identity=_boundary_identity(RunFrontierActionKind.CODING_AGENT),
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


def _claim_action(gate, lease, kind=RunFrontierActionKind.CODING_AGENT):
    return gate.claim(
        lease,
        kind=kind,
        provider_execution_id=(f"provider_{lease._reservation.intent.operation_id}"),
        boundary_identity=lease._reservation.intent.boundary_identity,
    )


def _accept_action(gate, lease):
    result = gate.record_result(
        lease,
        result_payload=b'{"provider_result":"complete"}',
    )
    return gate.accept_result(
        lease,
        result_receipt=result,
        disposition=RunActionResultDisposition.SUCCEEDED,
        accepted_result_payload=b'{"accepted_result":"complete"}',
    )


def _complete_action(
    gate,
    lease,
    kind=RunFrontierActionKind.CODING_AGENT,
):
    workspace_descriptor = _claim_action(gate, lease, kind)
    _accept_action(gate, lease)
    return workspace_descriptor


def _workspace_advance_frontier(case, receipt, commit_sha):
    predecessor_safety = receipt.checkpoint.safety_state
    predecessor_frontier = predecessor_safety.derivative_frontier
    predecessor_evidence = predecessor_frontier.evidence
    branch = case["settings"].workspace_git_branch
    predecessor_commit_sha = predecessor_evidence.branch_heads[branch]
    terminal = tuple(
        advance
        for advance in predecessor_evidence.branch_advances
        if advance.branch == branch and advance.commit_sha == predecessor_commit_sha
    )
    predecessor_advance_id = (
        None
        if predecessor_evidence.branch_origin_heads[branch] == predecessor_commit_sha
        else terminal[0].branch_advance_id
    )
    advance = RunBranchAdvance.build(
        branch=branch,
        predecessor_commit_sha=predecessor_commit_sha,
        commit_sha=commit_sha,
        predecessor_branch_advance_id=predecessor_advance_id,
        authorization_safety_state_id=predecessor_safety.safety_state_id,
    )
    evidence = _remint_evidence(
        predecessor_evidence,
        branch_advances=tuple(
            sorted(
                (*predecessor_evidence.branch_advances, advance),
                key=lambda item: item.branch_advance_id,
            )
        ),
        branch_heads={
            **predecessor_evidence.branch_heads,
            branch: commit_sha,
        },
    )
    return RunDerivativeFrontier.build(
        launch_subject_ids=predecessor_frontier.launch_subject_ids,
        evidence=evidence,
        derivatives=predecessor_frontier.derivatives,
    )


def _complete_unpublished_workspace_edit(case):
    publisher, receipt, _security, gate = _action_case(
        case,
        RunSafetyBoundary.IMPLEMENTATION,
    )
    stale_bundle, stale_checkpoint = _successor_at_boundary(
        case,
        publisher,
        receipt,
        RunSafetyBoundary.EVALUATION,
    )
    payload = b'{"implementation":"complete"}'
    permit = _issue_implementation_agent(
        gate,
        receipt,
        "gc_lifetime",
        payload,
    )
    with gate.hold(permit, payload) as lease:
        _claim_action(gate, lease)
        _commit_workspace_edit(
            case,
            "gc-lifetime-result.txt",
            "complete result\n",
        )
        _accept_action(gate, lease)
    return stale_bundle, stale_checkpoint


def _install_raw_commit(workspace, payload):
    object_bytes = f"commit {len(payload)}\0".encode("ascii") + payload
    object_id = hashlib.sha1(
        object_bytes,
        usedforsecurity=False,
    ).hexdigest()
    object_directory = workspace / ".git" / "objects" / object_id[:2]
    object_directory.mkdir(mode=0o700, exist_ok=True)
    object_path = object_directory / object_id[2:]
    object_path.write_bytes(zlib.compress(object_bytes))
    object_path.chmod(0o400)
    return object_id


def _rewrite_git_index(workspace, mutation):
    index_path = workspace / ".git" / "index"
    payload = index_path.read_bytes()
    entry_count = struct.unpack_from("!L", payload, 8)[0]
    entries = []
    position = 12
    for _entry_number in range(entry_count):
        terminator = payload.index(b"\0", position + 62, len(payload) - 20)
        entry_end = position + ((terminator + 1 - position + 7) // 8) * 8
        entries.append(bytearray(payload[position:entry_end]))
        position = entry_end
    extensions = payload[position:-20]
    if mutation == "order":
        entries[0], entries[1] = entries[1], entries[0]
    else:
        mode = struct.unpack_from("!L", entries[0], 24)[0]
        struct.pack_into("!L", entries[0], 24, mode | 0x80000000)
    body = payload[:12] + b"".join(entries) + extensions
    checksum = hashlib.sha1(body, usedforsecurity=False).digest()
    index_path.write_bytes(body + checksum)


def test_action_gate_holds_current_frontier_and_claims_once(
    publisher_case,
) -> None:
    publisher, receipt, security, gate = _action_case(publisher_case)
    payload = b'{"prompt":"complete"}'
    permit = _issue_ideation_agent(gate, receipt, payload)

    with gate.hold(permit, payload) as lease:
        workspace_descriptor = _claim_action(gate, lease)
        workspace = os.fstat(workspace_descriptor)
        assert (
            workspace.st_dev,
            workspace.st_ino,
        ) == permit.workspace_frontier.workspace_identity
        assert lease.run_checkpoint_id == receipt.run_checkpoint_id
        assert lease.safety_state_id == receipt.checkpoint.safety_state.safety_state_id
        assert publisher.require_current(receipt) == receipt.checkpoint
        with pytest.raises(RunFrontierActionError, match="claimed"):
            gate.claim(
                lease,
                kind=RunFrontierActionKind.CODING_AGENT,
                provider_execution_id="duplicate_execution_0123456789abcdef",
                boundary_identity=lease._reservation.intent.boundary_identity,
            )
        _accept_action(gate, lease)

    ledger = publisher.action_ledger_snapshot()
    assert ledger.event_count == 4
    assert ledger.operation_tails[0].tail_kind is (
        RunActionExecutionEventKind.RESULT_ACCEPTED
    )
    assert security.calls == [
        (
            receipt.checkpoint.safety_state.security_observation.scope_id,
            receipt.checkpoint.safety_state.security_observation.scope_contract_id,
            receipt.checkpoint.safety_state.security_observation.checked_subject_ids,
            receipt.checkpoint.safety_state.security_observation,
        )
    ]
    with pytest.raises(RunFrontierActionError, match="consumed"):
        with gate.hold(permit, payload):
            raise AssertionError("consumed permit entered")


def test_action_claim_rejects_adapter_substitution_before_security_use(
    publisher_case,
) -> None:
    _publisher, receipt, security, gate = _action_case(publisher_case)
    payload = b'{"prompt":"adapter-bound"}'
    permit = _issue_ideation_agent(gate, receipt, payload)
    substituted = RunActionBoundaryIdentity.mint(
        kind=RunFrontierActionKind.CODING_AGENT,
        execution_lifecycle_identity=RunActionExecutionLifecycleIdentity.mint(
            kind=RunFrontierActionKind.CODING_AGENT,
            implementation_id="test.coding_agent.execution",
            implementation_version="test.execution.v2",
            recovery_protocol_version="test.recovery.v1",
            sandbox_policy_id="test.coding_agent.sandbox.v1",
        ),
        result_interpreter_identity=(
            permit.intent.boundary_identity.result_interpreter_identity
        ),
    )

    with gate.hold(permit, payload) as lease:
        with pytest.raises(RunFrontierActionError, match="claim boundary"):
            gate.claim(
                lease,
                kind=RunFrontierActionKind.CODING_AGENT,
                provider_execution_id="substituted_adapter_0123456789abcdef",
                boundary_identity=substituted,
            )
        assert not security.calls
        _complete_action(gate, lease)


def test_action_issue_rejects_boundary_kind_substitution(
    publisher_case,
) -> None:
    publisher, receipt, _security, gate = _action_case(publisher_case)
    before = publisher.action_ledger_snapshot()

    with pytest.raises(RunFrontierActionError, match="unrecognized enum"):
        gate.issue(
            receipt,
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


@pytest.mark.parametrize("mutation", ("clone", "request"))
def test_action_permit_rejects_clone_and_complete_request_change(
    publisher_case,
    mutation,
) -> None:
    _publisher, receipt, _security, gate = _action_case(publisher_case)
    payload = b'{"prompt":"complete"}'
    permit = _issue_ideation_agent(gate, receipt, payload)
    attempted_permit = replace(permit) if mutation == "clone" else permit
    attempted_payload = payload if mutation == "clone" else b'{"prompt":"changed"}'

    with pytest.raises(
        RunFrontierActionError,
        match="cloned|another request",
    ):
        with gate.hold(attempted_permit, attempted_payload):
            raise AssertionError("invalid action permit entered")


def test_action_gate_requires_exact_current_security_observation(
    publisher_case,
) -> None:
    _publisher, receipt, security, gate = _action_case(publisher_case)
    current = receipt.checkpoint.safety_state.security_observation
    security.observation = _security_observation(
        publisher_case["active"].bootstrap_pin,
        current.checked_subject_ids,
        generation_offset=(
            current.generation
            - publisher_case[
                "active"
            ].bootstrap_pin.launch_manifest.security_observation.generation
            + 1
        ),
    )
    payload = b'{"prompt":"complete"}'
    permit = _issue_ideation_agent(gate, receipt, payload)

    with pytest.raises(RunFrontierActionError, match="must be refreshed"):
        with gate.hold(permit, payload) as lease:
            _claim_action(gate, lease)

    security.observation = current
    with pytest.raises(RunFrontierActionError, match="consumed"):
        with gate.hold(permit, payload):
            raise AssertionError("failed security permit was replayed")


def test_action_gate_rejects_wrong_checkpoint_boundary(
    publisher_case,
) -> None:
    _publisher, receipt, _security, gate = _action_case(publisher_case)

    with pytest.raises(RunFrontierActionError, match="boundary"):
        gate.issue(
            receipt,
            kind=RunFrontierActionKind.EVALUATOR,
            boundary=RunSafetyBoundary.EVALUATION,
            operation_id="evaluation_0123456789abcdef",
            request_payload=b'{"evaluation":"full"}',
            workspace_access=RunFrontierWorkspaceAccess.READ_ONLY,
            boundary_identity=_boundary_identity(RunFrontierActionKind.EVALUATOR),
        )


def test_action_shared_lease_blocks_run_state_publication(
    publisher_case,
) -> None:
    publisher, receipt, _security, gate = _action_case(publisher_case)
    payload = b'{"prompt":"complete"}'
    action_permit = _issue_ideation_agent(gate, receipt, payload)

    with ThreadPoolExecutor(max_workers=1) as executor:
        with gate.hold(action_permit, payload) as lease:
            _complete_action(gate, lease)
            successor_bundle, successor_checkpoint = _successor_at_boundary(
                publisher_case,
                publisher,
                receipt,
                RunSafetyBoundary.IMPLEMENTATION,
            )

            def publish_successor():
                publication_permit = publisher.issue_publication_permit(
                    receipt,
                    successor_checkpoint,
                    successor_bundle,
                )
                return publisher.publish(
                    publication_permit,
                    successor_checkpoint,
                    successor_bundle,
                )

            future = executor.submit(
                publish_successor,
            )
            with pytest.raises(TimeoutError):
                future.result(timeout=0.05)
        successor_receipt = future.result(timeout=5)

    assert successor_receipt.checkpoint == successor_checkpoint
    with pytest.raises(RunFrontierActionError, match="consumed"):
        with gate.hold(action_permit, payload):
            raise AssertionError("completed action permit was replayed")


def test_publication_requires_exact_unchanged_terminal_workspace(
    publisher_case,
) -> None:
    publisher, receipt, _security, gate = _action_case(publisher_case)
    payload = b'{"prompt":"complete"}'
    permit = _issue_ideation_agent(gate, receipt, payload)
    with gate.hold(permit, payload) as lease:
        _complete_action(gate, lease)
    (publisher_case["active"].workspace / ".git" / "COMMIT_EDITMSG").write_text(
        "changed after terminal action\n",
        encoding="utf-8",
    )
    successor_bundle, successor_checkpoint = _successor_at_boundary(
        publisher_case,
        publisher,
        receipt,
        RunSafetyBoundary.IMPLEMENTATION,
    )

    with pytest.raises(
        RunStatePublisherError,
        match="terminal workspace changed",
    ):
        publisher.issue_publication_permit(
            receipt,
            successor_checkpoint,
            successor_bundle,
        )


def test_multiple_read_only_actions_form_one_workspace_chain(
    publisher_case,
) -> None:
    publisher, receipt, _security, gate = _action_case(publisher_case)
    first_payload = b'{"prompt":"first complete request"}'
    first = _issue_ideation_agent(gate, receipt, first_payload)
    with gate.hold(first, first_payload) as lease:
        _complete_action(gate, lease)
    second_payload = b'{"prompt":"second complete request"}'
    second = gate.issue(
        receipt,
        kind=RunFrontierActionKind.CODING_AGENT,
        boundary=RunSafetyBoundary.IDEATION,
        operation_id="second_read_action_0123456789abcdef",
        request_payload=second_payload,
        workspace_access=RunFrontierWorkspaceAccess.READ_ONLY,
        boundary_identity=_boundary_identity(RunFrontierActionKind.CODING_AGENT),
    )
    with gate.hold(second, second_payload) as lease:
        _complete_action(gate, lease)
    successor_bundle, successor_checkpoint = _successor_at_boundary(
        publisher_case,
        publisher,
        receipt,
        RunSafetyBoundary.IMPLEMENTATION,
    )

    successor = publisher.publish(
        publisher.issue_publication_permit(
            receipt,
            successor_checkpoint,
            successor_bundle,
        ),
        successor_checkpoint,
        successor_bundle,
    )

    assert successor.projection.action_ledger.event_count == 8


def test_unresolved_durable_action_blocks_candidate_publication(
    publisher_case,
) -> None:
    publisher, receipt, _security, gate = _action_case(publisher_case)
    _issue_ideation_agent(gate, receipt)
    successor_bundle, successor_checkpoint = _successor_at_boundary(
        publisher_case,
        publisher,
        receipt,
        RunSafetyBoundary.IMPLEMENTATION,
    )

    with pytest.raises(
        RunStatePublisherError,
        match="unresolved execution",
    ):
        publisher.issue_publication_permit(
            receipt,
            successor_checkpoint,
            successor_bundle,
        )


def test_stopped_checkpoint_cannot_issue_an_action(
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
    )

    with pytest.raises(RunFrontierActionError, match="stopped or completed"):
        _issue_ideation_agent(gate, receipt)


def test_completed_checkpoint_is_not_actionable(
    publisher_case,
) -> None:
    _publisher, receipt, _security, gate = _action_case(publisher_case)
    permit = _issue_ideation_agent(gate, receipt)
    completed = object.__new__(RunCheckpoint)
    for field_name, value in vars(receipt.checkpoint).items():
        object.__setattr__(completed, field_name, value)
    object.__setattr__(
        completed,
        "status",
        RunCheckpointStatus.COMPLETED,
    )

    with pytest.raises(RunFrontierActionError, match="stopped or completed"):
        gate._require_actionable(completed, permit.intent)


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
def test_action_gate_enforces_exact_capability_matrix(
    publisher_case,
    boundary,
    kind,
    access,
) -> None:
    _publisher, receipt, _security, gate = _action_case(
        publisher_case,
        boundary,
    )

    with pytest.raises(RunFrontierActionError, match="workspace access"):
        gate.issue(
            receipt,
            kind=kind,
            boundary=boundary,
            operation_id="forbidden_action_0123456789abcdef",
            request_payload=b'{"request":"complete"}',
            workspace_access=access,
            boundary_identity=_boundary_identity(kind),
        )


def test_embedding_action_receives_no_workspace_capability(
    publisher_case,
) -> None:
    _publisher, receipt, _security, gate = _action_case(publisher_case)
    payload = b'{"texts":["complete input"]}'
    permit = gate.issue(
        receipt,
        kind=RunFrontierActionKind.EMBEDDING,
        boundary=RunSafetyBoundary.IDEATION,
        operation_id="embedding_0123456789abcdef",
        request_payload=payload,
        workspace_access=RunFrontierWorkspaceAccess.NONE,
        boundary_identity=_boundary_identity(RunFrontierActionKind.EMBEDDING),
    )

    assert permit.workspace_frontier is None
    with gate.hold(permit, payload) as lease:
        assert (
            _complete_action(
                gate,
                lease,
                RunFrontierActionKind.EMBEDDING,
            )
            is None
        )


def test_duplicate_action_intent_and_operation_are_reserved_once(
    publisher_case,
) -> None:
    _publisher, receipt, _security, gate = _action_case(publisher_case)
    payload = b'{"prompt":"complete"}'
    first = _issue_ideation_agent(gate, receipt, payload)

    with pytest.raises(RunFrontierActionError, match="unresolved durable action"):
        _issue_ideation_agent(gate, receipt, payload)
    with pytest.raises(RunFrontierActionError, match="unresolved durable action"):
        gate.issue(
            receipt,
            kind=RunFrontierActionKind.CODING_AGENT,
            boundary=RunSafetyBoundary.IDEATION,
            operation_id=first.intent.operation_id,
            request_payload=b'{"prompt":"another complete request"}',
            workspace_access=RunFrontierWorkspaceAccess.READ_ONLY,
            boundary_identity=_boundary_identity(RunFrontierActionKind.CODING_AGENT),
        )

    with gate.hold(first, payload) as lease:
        _complete_action(gate, lease)


def test_concurrent_duplicate_action_issue_has_one_winner(
    publisher_case,
) -> None:
    _publisher, receipt, _security, gate = _action_case(publisher_case)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(
            executor.submit(_issue_ideation_agent, gate, receipt)
            for _attempt in range(2)
        )
    permits = []
    errors = []
    for future in futures:
        if future.exception() is None:
            permits.append(future.result())
        else:
            errors.append(future.exception())

    assert len(permits) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], RunFrontierActionError)
    assert (
        "unresolved durable action" in str(errors[0])
        or "predecessor ledger moved" in str(errors[0])
        or "already reserved" in str(errors[0])
        or "live session" in str(errors[0])
    )


def test_reconstructed_gate_observes_durable_unresolved_reservation(
    publisher_case,
) -> None:
    publisher, receipt, security, gate = _action_case(publisher_case)
    _issue_ideation_agent(gate, receipt)
    reconstructed_publisher = RunStatePublisher(
        publisher_case["active"],
        publisher_case["settings"],
    )
    reconstructed_gate = RunFrontierActionGate(
        active_workspace=publisher_case["active"],
        publisher=reconstructed_publisher,
        security_authority=security,
    )
    reconstructed_receipt = reconstructed_publisher.load_reconciled()

    with pytest.raises(
        RunFrontierActionError,
        match="unresolved durable action",
    ):
        _issue_ideation_agent(
            reconstructed_gate,
            reconstructed_receipt,
            b'{"prompt":"different complete request"}',
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "source",
        "index",
        "branch",
        "assume_unchanged",
        "replace_ref",
        "alternates",
        "shallow",
        "unreachable_object",
        "missing_object",
    ),
)
def test_workspace_mutation_after_issue_invalidates_action(
    publisher_case,
    mutation,
) -> None:
    _publisher, receipt, _security, gate = _action_case(publisher_case)
    payload = b'{"prompt":"complete"}'
    permit = _issue_ideation_agent(gate, receipt, payload)
    workspace = publisher_case["active"].workspace
    if mutation == "source":
        (workspace / "uncommitted.txt").write_text(
            "dirty",
            encoding="utf-8",
        )
    elif mutation == "index":
        index = workspace / ".git" / "index"
        payload_bytes = bytearray(index.read_bytes())
        payload_bytes[-1] ^= 1
        index.write_bytes(payload_bytes)
    elif mutation == "branch":
        (
            workspace
            / ".git"
            / "refs"
            / "heads"
            / publisher_case["settings"].workspace_git_branch
        ).write_text("0" * 40 + "\n", encoding="ascii")
    elif mutation == "assume_unchanged":
        _run_git(
            workspace,
            "update-index",
            "--assume-unchanged",
            "EXPERT_REPO.md",
        )
    elif mutation == "replace_ref":
        replacement = workspace / ".git" / "refs" / "replace"
        replacement.mkdir()
        (replacement / permit.workspace_frontier.commit_sha).write_text(
            permit.workspace_frontier.commit_sha + "\n",
            encoding="ascii",
        )
    elif mutation == "alternates":
        (workspace / ".git" / "objects" / "info" / "alternates").write_text(
            "/untrusted/object/store\n",
            encoding="utf-8",
        )
    elif mutation == "shallow":
        (workspace / ".git" / "shallow").write_text(
            permit.workspace_frontier.commit_sha + "\n",
            encoding="ascii",
        )
    elif mutation == "unreachable_object":
        message = workspace / ".git" / "COMMIT_EDITMSG"
        message.write_text("unreachable object\n", encoding="utf-8")
        _run_git(
            workspace,
            "hash-object",
            "-w",
            ".git/COMMIT_EDITMSG",
        )
    else:
        blob_id = _run_git(
            workspace,
            "rev-parse",
            "HEAD:EXPERT_REPO.md",
        ).strip()
        (workspace / ".git" / "objects" / blob_id[:2] / blob_id[2:]).unlink()

    with pytest.raises(
        RunWorkspaceFrontierError,
        match=(
            "commit tree|checksum|checkpoint|unsupported control|"
            "unsupported references|not loose and self-contained|"
            "unreachable|missing or wrong-kind|object directory is invalid"
            "|unsupported flags"
        ),
    ):
        with gate.hold(permit, payload):
            raise AssertionError("dirty workspace authorized an action")


def test_admitted_git_metadata_is_bound_to_read_only_frontier(
    publisher_case,
) -> None:
    _publisher, receipt, _security, gate = _action_case(publisher_case)
    payload = b'{"prompt":"complete"}'
    permit = _issue_ideation_agent(gate, receipt, payload)
    (publisher_case["active"].workspace / ".git" / "COMMIT_EDITMSG").write_text(
        "changed metadata\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RunFrontierActionError,
        match="workspace frontier changed",
    ):
        with gate.hold(permit, payload):
            raise AssertionError("changed Git metadata authorized an action")


@pytest.mark.parametrize("mutation", ("order", "mode"))
def test_git_index_requires_canonical_entry_order_and_mode(
    publisher_case,
    mutation,
) -> None:
    _publisher, receipt, _security, gate = _action_case(publisher_case)
    payload = b'{"prompt":"complete"}'
    permit = _issue_ideation_agent(gate, receipt, payload)
    _rewrite_git_index(
        publisher_case["active"].workspace,
        mutation,
    )

    with pytest.raises(
        RunWorkspaceFrontierError,
        match="path order|entry is invalid",
    ):
        with gate.hold(permit, payload):
            raise AssertionError("noncanonical Git index authorized an action")


@pytest.mark.parametrize(
    "headers",
    (
        (
            b"author Kapso Test <kapso-test@example.invalid> 1 +0000\n"
            b"parent {parent}\n"
            b"committer Kapso Test <kapso-test@example.invalid> 1 +0000\n"
        ),
        (
            b"parent {parent}\n"
            b"committer Kapso Test <kapso-test@example.invalid> 1 +0000\n"
        ),
        (
            b"parent {parent}\n"
            b"author Kapso Test <kapso-test@example.invalid> 1 +0000\n"
            b"author Kapso Test <kapso-test@example.invalid> 1 +0000\n"
            b"committer Kapso Test <kapso-test@example.invalid> 1 +0000\n"
        ),
    ),
)
def test_workspace_edit_rejects_malformed_commit_parent_grammar(
    publisher_case,
    headers,
) -> None:
    _publisher, receipt, _security, gate = _action_case(
        publisher_case,
        RunSafetyBoundary.IMPLEMENTATION,
    )
    payload = b'{"implementation":"complete"}'
    permit = _issue_implementation_agent(
        gate,
        receipt,
        "malformed_commit",
        payload,
    )
    workspace = publisher_case["active"].workspace
    before = permit.workspace_frontier
    commit_payload = (
        f"tree {before.git_tree_sha}\n".encode("ascii")
        + headers.replace(
            b"{parent}",
            before.commit_sha.encode("ascii"),
        )
        + b"\nmalformed\n"
    )

    with pytest.raises(
        RunWorkspaceFrontierError,
        match="commit",
    ):
        with gate.hold(permit, payload) as lease:
            _claim_action(gate, lease)
            commit_sha = _install_raw_commit(workspace, commit_payload)
            (
                workspace
                / ".git"
                / "refs"
                / "heads"
                / publisher_case["settings"].workspace_git_branch
            ).write_text(commit_sha + "\n", encoding="ascii")
            _accept_action(gate, lease)


def test_workspace_cannot_change_after_durable_acceptance(
    publisher_case,
) -> None:
    _publisher, receipt, _security, gate = _action_case(
        publisher_case,
        RunSafetyBoundary.IMPLEMENTATION,
    )
    payload = b'{"implementation":"complete"}'
    permit = _issue_implementation_agent(
        gate,
        receipt,
        "post_accept_mutation",
        payload,
    )
    workspace = publisher_case["active"].workspace

    with pytest.raises(
        RunWorkspaceFrontierError,
        match="commit tree",
    ):
        with gate.hold(permit, payload) as lease:
            _claim_action(gate, lease)
            _commit_workspace_edit(
                publisher_case,
                "accepted-result.txt",
                "accepted result\n",
            )
            _accept_action(gate, lease)
            (workspace / "accepted-result.txt").write_text(
                "changed after acceptance\n",
                encoding="utf-8",
            )


def test_failed_workspace_edit_accepts_only_the_unchanged_frontier(
    publisher_case,
) -> None:
    _publisher, receipt, _security, gate = _action_case(
        publisher_case,
        RunSafetyBoundary.IMPLEMENTATION,
    )
    payload = b'{"implementation":"provider failed"}'
    permit = _issue_implementation_agent(
        gate,
        receipt,
        "failed_unchanged",
        payload,
    )

    with gate.hold(permit, payload) as lease:
        _claim_action(gate, lease)
        result = gate.record_result(
            lease,
            result_payload=b'{"provider_result":"failed"}',
        )
        acceptance = gate.accept_result(
            lease,
            result_receipt=result,
            disposition=RunActionResultDisposition.FAILED,
            accepted_result_payload=b'{"accepted_result":"failed"}',
        )

    assert acceptance.workspace_after == lease._reservation.frontier.workspace_before


def test_failed_workspace_edit_rejects_a_committed_successor(
    publisher_case,
) -> None:
    _publisher, receipt, _security, gate = _action_case(
        publisher_case,
        RunSafetyBoundary.IMPLEMENTATION,
    )
    payload = b'{"implementation":"provider failed after edit"}'
    permit = _issue_implementation_agent(
        gate,
        receipt,
        "failed_changed",
        payload,
    )

    with pytest.raises(
        RunActionStoreError,
        match="failed editing action",
    ):
        with gate.hold(permit, payload) as lease:
            _claim_action(gate, lease)
            _commit_workspace_edit(
                publisher_case,
                "failed-edit.txt",
                "must not be accepted\n",
            )
            result = gate.record_result(
                lease,
                result_payload=b'{"provider_result":"failed"}',
            )
            gate.accept_result(
                lease,
                result_receipt=result,
                disposition=RunActionResultDisposition.FAILED,
                accepted_result_payload=b'{"accepted_result":"failed"}',
            )


def test_action_gate_rejects_a_reminted_terminal_from_another_safety_boundary(
    publisher_case,
) -> None:
    _publisher, frontier, _security, gate = _action_case(publisher_case)
    payload = b'{"prompt":"complete before frontier remint"}'
    permit = _issue_ideation_agent(gate, frontier, payload)
    with gate.hold(permit, payload) as lease:
        _claim_action(gate, lease)
        _accept_action(gate, lease)

    original_events = gate._action_store.inspect().events_for(
        permit.intent.operation_id
    )
    original_reservation = original_events[0].reservation
    reminted_intent = RunActionIntent.from_request(
        kind=original_reservation.intent.kind,
        boundary=RunSafetyBoundary.EVALUATION,
        operation_id=original_reservation.intent.operation_id,
        request_payload=payload,
        workspace_access=original_reservation.intent.workspace_access,
        boundary_identity=original_reservation.intent.boundary_identity,
    )
    reminted_reservation = RunActionReservation.build(
        intent=reminted_intent,
        frontier=original_reservation.frontier,
        predecessor_ledger=permit._predecessor_ledger,
    )
    original_spawn = original_events[1].spawn_commit
    reminted_spawn = RunActionSpawnCommit.mint(
        reservation_id=reminted_reservation.reservation_id,
        provider_execution_id=original_spawn.provider_execution_id,
        invocation_nonce=original_spawn.invocation_nonce,
        security_observation_id=original_spawn.security_observation_id,
        boundary_identity=original_spawn.boundary_identity,
    )
    original_result = original_events[2].result_receipt
    reminted_result = RunActionResultReceipt.mint(
        spawn_commit_id=reminted_spawn.spawn_commit_id,
        provider_execution_id=original_result.provider_execution_id,
        result_blob=original_result.result_blob,
    )
    original_acceptance = original_events[3].acceptance
    reminted_acceptance = RunActionAcceptance.mint(
        result_receipt_id=reminted_result.result_receipt_id,
        disposition=original_acceptance.disposition,
        accepted_result_blob=original_acceptance.accepted_result_blob,
        workspace_after=original_acceptance.workspace_after,
    )
    event_payloads = (
        (RunActionExecutionEventKind.INTENT_RESERVED, None, None, None),
        (RunActionExecutionEventKind.SPAWN_COMMITTED, reminted_spawn, None, None),
        (RunActionExecutionEventKind.RESULT_RECEIVED, None, reminted_result, None),
        (
            RunActionExecutionEventKind.RESULT_ACCEPTED,
            None,
            None,
            reminted_acceptance,
        ),
    )
    predecessor_event_id = None
    reminted_events = []
    for event_number, (
        event_kind,
        spawn_commit,
        result_receipt,
        acceptance,
    ) in enumerate(event_payloads, start=1):
        event = RunActionExecutionEvent.mint(
            event_number=event_number,
            predecessor_event_id=predecessor_event_id,
            event_kind=event_kind,
            reservation=reminted_reservation,
            spawn_commit=spawn_commit,
            result_receipt=result_receipt,
            acceptance=acceptance,
            terminal_reason=None,
            workspace_after=None,
        )
        reminted_events.append(event)
        predecessor_event_id = event.event_id
    operation_digest = tree_or_blob_digest(
        permit.intent.operation_id.encode("utf-8")
    ).removeprefix("sha256:")
    store_path = (
        publisher_case["active"].run_root
        / publisher_case["settings"].run_action_store_path
    )
    for event in reminted_events:
        event_path = store_path / (
            f"operation-{operation_digest}-event-{event.event_number:04d}.json"
        )
        event_path.chmod(0o600)
        event_path.write_bytes(event.to_json_bytes())
        event_path.chmod(0o400)

    with pytest.raises(
        RunFrontierActionError,
        match="another frontier",
    ):
        gate.issue(
            frontier,
            kind=RunFrontierActionKind.CODING_AGENT,
            boundary=RunSafetyBoundary.IDEATION,
            operation_id="after_remint_0123456789abcdef0123456789abcdef",
            request_payload=b'{"prompt":"must not run"}',
            workspace_access=RunFrontierWorkspaceAccess.READ_ONLY,
            boundary_identity=_boundary_identity(RunFrontierActionKind.CODING_AGENT),
        )


def test_durable_edit_evidence_survives_gate_and_publisher_collection(
    publisher_case,
) -> None:
    stale_bundle, stale_checkpoint = _complete_unpublished_workspace_edit(
        publisher_case
    )
    gc.collect()
    publisher = RunStatePublisher(
        publisher_case["active"],
        publisher_case["settings"],
    )
    receipt = publisher.load_reconciled()

    with pytest.raises(
        RunStatePublisherError,
        match="action ledger",
    ):
        publisher.issue_publication_permit(
            receipt,
            stale_checkpoint,
            stale_bundle,
        )


def test_workspace_edit_is_exclusive_and_requires_accounting_successor(
    publisher_case,
) -> None:
    publisher, receipt, security, gate = _action_case(
        publisher_case,
        RunSafetyBoundary.IMPLEMENTATION,
    )
    payload = b'{"implementation":"complete"}'
    stale_bundle, stale_checkpoint = _successor_at_boundary(
        publisher_case,
        publisher,
        receipt,
        RunSafetyBoundary.EVALUATION,
    )
    stale_publication = publisher.issue_publication_permit(
        receipt,
        stale_checkpoint,
        stale_bundle,
    )
    first = _issue_implementation_agent(gate, receipt, "first", payload)
    with pytest.raises(
        RunFrontierActionError,
        match="unresolved durable action",
    ):
        _issue_implementation_agent(gate, receipt, "second", payload)
    with gate.hold(first, payload) as lease:
        _claim_action(gate, lease)
        commit_sha = _commit_workspace_edit(
            publisher_case,
            "implementation-result.txt",
            "complete result\n",
        )
        _accept_action(gate, lease)

    with pytest.raises(
        RunStatePublisherError,
        match="action ledger",
    ):
        publisher.publish(
            stale_publication,
            stale_checkpoint,
            stale_bundle,
        )
    with pytest.raises(
        RunFrontierActionError,
        match="awaits reconciliation",
    ):
        gate.issue(
            receipt,
            kind=RunFrontierActionKind.CODING_AGENT,
            boundary=RunSafetyBoundary.IMPLEMENTATION,
            operation_id="post_edit_0123456789abcdef",
            request_payload=payload,
            workspace_access=RunFrontierWorkspaceAccess.EDIT_WORKSPACE,
            boundary_identity=_boundary_identity(RunFrontierActionKind.CODING_AGENT),
        )
    alternate_publisher = RunStatePublisher(
        publisher_case["active"],
        publisher_case["settings"],
    )
    alternate_receipt = alternate_publisher.load_reconciled()

    frontier = _workspace_advance_frontier(
        publisher_case,
        receipt,
        commit_sha,
    )
    bundle, checkpoint = _successor_at_boundary(
        publisher_case,
        publisher,
        receipt,
        RunSafetyBoundary.EVALUATION,
        derivative_frontier=frontier,
    )
    with pytest.raises(
        RunStatePublisherError,
        match="action ledger",
    ):
        alternate_publisher.issue_publication_permit(
            alternate_receipt,
            stale_checkpoint,
            stale_bundle,
        )
    successor = publisher.publish(
        publisher.issue_publication_permit(
            receipt,
            checkpoint,
            bundle,
        ),
        checkpoint,
        bundle,
    )
    security.observation = successor.checkpoint.safety_state.security_observation
    evaluation_permit = gate.issue(
        successor,
        kind=RunFrontierActionKind.EVALUATOR,
        boundary=RunSafetyBoundary.EVALUATION,
        operation_id="evaluation_after_edit_0123456789abcdef",
        request_payload=b'{"evaluation":"complete"}',
        workspace_access=RunFrontierWorkspaceAccess.READ_ONLY,
        boundary_identity=_boundary_identity(RunFrontierActionKind.EVALUATOR),
    )

    assert evaluation_permit.workspace_frontier.commit_sha == commit_sha
