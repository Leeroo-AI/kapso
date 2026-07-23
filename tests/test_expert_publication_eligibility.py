from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import test_expert_release_matrix_reservation as reservation_fixture_module
from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.contracts import (
    ExpertPromotionState,
    ExpertValidationStage,
)
from kapso.cross_run.expert.promotion_authority import (
    ExpertPublicationEligibilityCoordinator,
    ExpertPublicationEligibilityError,
    publication_eligibility_security_subject_ids,
)
from kapso.cross_run.expert.promotion_decision_contracts import (
    ExpertReleaseMatrixDecisionOutcome,
)
from kapso.cross_run.expert.promotion_stage import (
    ExpertReleaseMatrixStageCoordinator,
)
from kapso.cross_run.expert.task_evaluation_contracts import (
    TaskEvaluationLegKind,
)
from kapso.cross_run.expert.task_evaluation_authority_contracts import (
    TaskEvaluationCurrentReleaseObservation,
)
from kapso.cross_run.expert.validation_store import (
    ExpertValidationCompareAndSwapError,
    ExpertValidationStore,
    ExpertValidationStoreError,
)
from kapso.cross_run.security_authority_contracts import (
    SecurityDenylistObservation,
)
from test_expert_promotion_decision import _settings
from test_expert_promotion_evidence import (
    _SemanticProvider,
    _bootstrap_prepared_with_store,
    _complete_execution,
    _execution_runtime,
)
from test_expert_promotion_stage import _completed_runtime
from test_expert_task_evaluation_execution import (
    _parent_prepared_with_additional_case,
)


class _CurrentAuthority:
    def __init__(self, observations, calls, provider):
        self.observations = observations
        self.calls = calls
        self.provider = provider
        self.position = 0

    def observe_task_evaluation_current(self, scope_id):
        self.calls.append("current")
        observation = self.observations[self.position]
        self.position += 1
        assert observation.scope_id == scope_id
        self.provider.release_id = observation.release_id
        return observation

    def reset(self, observations):
        self.observations = observations
        self.position = 0


class _DenylistAuthority:
    def __init__(self, calls, *, denied=False, callback=None):
        self.calls = calls
        self.denied = denied
        self.callback = callback
        self.checked_subject_ids = None

    def observe_exact(
        self,
        *,
        scope_id,
        scope_contract_id,
        checked_subject_ids,
    ):
        self.calls.append("denylist")
        self.checked_subject_ids = checked_subject_ids
        denied_subject_ids = (checked_subject_ids[0],) if self.denied else ()
        observation = SecurityDenylistObservation.mint(
            scope_id=scope_id,
            scope_contract_id=scope_contract_id,
            scope_repository_binding_hash=tree_or_blob_digest(b"scope binding"),
            snapshot_id=content_id(
                "security-denylist-snapshot",
                {"generation": 7},
            ),
            generation=7,
            publication_id=content_id(
                "github-publication",
                {"security_denylist_generation": 7},
            ),
            repository_full_name="Leeroo-AI/kapso-security",
            repository_node_id="security_repo_node",
            pointer_digest=tree_or_blob_digest(b"security CURRENT"),
            authority_commit_sha="b" * 40,
            release_attestation_ref="attestations/security-denylist",
            checked_subject_ids=checked_subject_ids,
            denied_subject_ids=denied_subject_ids,
        )
        if self.callback is not None:
            self.callback()
        return observation


def _moved_current(observation, *, release_id=None, head_commit_sha="c" * 40):
    if release_id is None:
        release_id = observation.release_id
    present = release_id is not None
    return TaskEvaluationCurrentReleaseObservation.mint(
        scope_id=observation.scope_id,
        release_id=release_id,
        publication_id=(
            content_id("github-publication", {"release_id": release_id})
            if present
            else None
        ),
        repository_full_name=observation.repository_full_name,
        repository_node_id=observation.repository_node_id,
        default_branch_head_commit_sha=head_commit_sha,
        current_pointer_digest=(
            tree_or_blob_digest(f"CURRENT:{release_id}".encode("utf-8"))
            if present
            else None
        ),
        validation_closure_ids=(
            (content_id("expert-validation-transition", {"release": release_id}),)
            if present
            else ()
        ),
    )


def _publish_matrix(
    tmp_path,
    monkeypatch,
    *,
    bootstrap,
    settings,
    negative_effect=False,
):
    monkeypatch.setattr(
        reservation_fixture_module,
        "_quality_only_validation_settings",
        lambda: settings,
    )
    if bootstrap:
        validation_store, snapshot, prepared = _bootstrap_prepared_with_store(
            tmp_path,
            monkeypatch,
        )
    else:
        validation_store, snapshot, prepared = _parent_prepared_with_additional_case(
            tmp_path,
            monkeypatch,
        )
    if negative_effect:
        values = {
            case.evaluation_case_id: {
                TaskEvaluationLegKind.CANDIDATE: 0.0,
                TaskEvaluationLegKind.PARENT_CONTROL: 1.0,
            }
            for case in prepared.plan_join.request.cases
        }
        reservation, execution_store, registry, authority = _execution_runtime(
            validation_store,
            snapshot,
            prepared,
            lambda trusted_root, provider_key: _SemanticProvider(
                trusted_root,
                provider_key,
                values,
            ),
        )
        completed = _complete_execution(
            prepared=prepared,
            reservation_snapshot=reservation,
            execution_store=execution_store,
            registry=registry,
            authority_coordinator=authority,
        )
    else:
        reservation, execution_store, completed = _completed_runtime(
            validation_store,
            snapshot,
            prepared,
        )
    committed = ExpertReleaseMatrixStageCoordinator(
        validation_store=validation_store,
        execution_store=execution_store,
    ).publish_completed(
        completed_execution=completed,
        reservation_snapshot=reservation,
        prepared_request=prepared,
    )
    return SimpleNamespace(
        validation_store=validation_store,
        matrix_commit=committed,
        prepared=prepared,
    )


@pytest.fixture(scope="module")
def terminal_cases(tmp_path_factory):
    with pytest.MonkeyPatch.context() as monkeypatch:
        parent_approved = _publish_matrix(
            tmp_path_factory.mktemp("publication-parent-approved"),
            monkeypatch,
            bootstrap=False,
            settings=_settings(minimum_replicates=1, minimum_pairs=2),
        )
    with pytest.MonkeyPatch.context() as monkeypatch:
        bootstrap_approved = _publish_matrix(
            tmp_path_factory.mktemp("publication-bootstrap-approved"),
            monkeypatch,
            bootstrap=True,
            settings=_settings(minimum_replicates=1, minimum_pairs=1),
        )
    with pytest.MonkeyPatch.context() as monkeypatch:
        retained = _publish_matrix(
            tmp_path_factory.mktemp("publication-retained"),
            monkeypatch,
            bootstrap=False,
            settings=_settings(minimum_replicates=2, minimum_pairs=2),
        )
    with pytest.MonkeyPatch.context() as monkeypatch:
        failed = _publish_matrix(
            tmp_path_factory.mktemp("publication-failed"),
            monkeypatch,
            bootstrap=False,
            settings=_settings(minimum_replicates=1, minimum_pairs=2),
            negative_effect=True,
        )
    with pytest.MonkeyPatch.context() as monkeypatch:
        bootstrap_current_mismatch = _publish_matrix(
            tmp_path_factory.mktemp("publication-bootstrap-current-mismatch"),
            monkeypatch,
            bootstrap=True,
            settings=_settings(minimum_replicates=1, minimum_pairs=1),
        )
    with pytest.MonkeyPatch.context() as monkeypatch:
        adversarial = _publish_matrix(
            tmp_path_factory.mktemp("publication-adversarial"),
            monkeypatch,
            bootstrap=False,
            settings=_settings(minimum_replicates=1, minimum_pairs=2),
        )
    return SimpleNamespace(
        parent_approved=parent_approved,
        bootstrap_approved=bootstrap_approved,
        retained=retained,
        failed=failed,
        bootstrap_current_mismatch=bootstrap_current_mismatch,
        adversarial=adversarial,
    )


def _coordinator(case, current_observations=None, *, denylist=None):
    calls = []
    prepared = case.prepared
    current = case.validation_store.reducer.current_release_provider
    current_sequence = _CurrentAuthority(
        (
            (prepared.current_release_observation,) * 2
            if current_observations is None
            else current_observations
        ),
        calls,
        current,
    )
    current.observe_task_evaluation_current = (
        current_sequence.observe_task_evaluation_current
    )
    adapters = case.validation_store.reducer.task_adapter_provider
    original_resolve_exact = adapters.resolve_exact

    def resolve_exact(**request):
        calls.append(
            (
                "adapter",
                (
                    request["task_adapter_manifest_id"],
                    request["verification_receipt_id"],
                ),
            )
        )
        return original_resolve_exact(**request)

    adapters.resolve_exact = resolve_exact
    security = _DenylistAuthority(calls) if denylist is None else denylist
    coordinator = ExpertPublicationEligibilityCoordinator(
        validation_store=case.validation_store,
        current_release_authority=current,
        task_adapter_authority=adapters,
        security_denylist_authority=security,
    )
    return SimpleNamespace(
        coordinator=coordinator,
        calls=calls,
        current=current,
        current_sequence=current_sequence,
        adapters=adapters,
        denylist=security,
        original_resolve_exact=original_resolve_exact,
    )


@pytest.mark.parametrize(
    ("case_name", "expected_outcome", "expected_state"),
    (
        (
            "failed",
            ExpertReleaseMatrixDecisionOutcome.FAILED,
            ExpertPromotionState.FAILED,
        ),
        (
            "retained",
            ExpertReleaseMatrixDecisionOutcome.PARETO_RETAINED,
            ExpertPromotionState.PARETO_RETAINED,
        ),
    ),
)
def test_local_terminal_outcomes_preserve_matrix_prefix_without_external_calls(
    terminal_cases,
    case_name,
    expected_outcome,
    expected_state,
):
    case = getattr(terminal_cases, case_name)
    external_calls = []
    current = case.validation_store.reducer.current_release_provider
    adapters = case.validation_store.reducer.task_adapter_provider

    def unexpected_current(_scope_id):
        external_calls.append("current")
        raise AssertionError("local outcome must not consult CURRENT")

    def unexpected_adapter(**_request):
        external_calls.append("adapter")
        raise AssertionError("local outcome must not re-resolve adapters")

    current.observe_task_evaluation_current = unexpected_current
    adapters.resolve_exact = unexpected_adapter
    denylist = _DenylistAuthority(external_calls)
    coordinator = ExpertPublicationEligibilityCoordinator(
        validation_store=case.validation_store,
        current_release_authority=current,
        task_adapter_authority=adapters,
        security_denylist_authority=denylist,
    )
    matrix = case.matrix_commit

    committed = coordinator.publish(
        candidate_id=matrix.snapshot.state.candidate_id,
        release_matrix_stage_result_id=matrix.stage_result.stage_result_record_id,
    )

    assert committed.replayed is False
    assert committed.stage_result.outcome is expected_outcome
    assert committed.snapshot.state.promotion_state is expected_state
    assert committed.snapshot.state.next_stage is None
    assert (
        committed.snapshot.transition.accepted_stage_result_record_ids
        == matrix.snapshot.transition.accepted_stage_result_record_ids
    )
    assert committed.snapshot.state.terminal_evidence_ids == (
        committed.stage_result.promotion_decision.promotion_decision_id,
    )
    assert external_calls == []


@pytest.mark.parametrize("case_name", ("parent_approved", "bootstrap_approved"))
def test_approved_parent_and_bootstrap_recheck_exact_external_authority(
    terminal_cases,
    case_name,
):
    case = getattr(terminal_cases, case_name)
    authority = _coordinator(case)
    matrix = case.matrix_commit
    replay, input_snapshot = (
        case.validation_store.reopen_or_replay_publication_eligibility(
            candidate_id=matrix.snapshot.state.candidate_id,
            release_matrix_stage_result_id=(matrix.stage_result.stage_result_record_id),
        )
    )
    assert replay is None
    assert input_snapshot is not None

    committed = authority.coordinator.publish(
        candidate_id=matrix.snapshot.state.candidate_id,
        release_matrix_stage_result_id=matrix.stage_result.stage_result_record_id,
    )

    assert committed.replayed is False
    assert committed.stage_result.outcome is ExpertReleaseMatrixDecisionOutcome.APPROVED
    assert committed.snapshot.state.promotion_state is ExpertPromotionState.APPROVED
    assert committed.snapshot.state.next_stage is None
    assert (
        committed.snapshot.state.accepted_stage_results[-1].stage
        is ExpertValidationStage.PUBLICATION_ELIGIBILITY
    )
    assert authority.calls[0] == "current"
    assert authority.calls[-1] == "current"
    assert authority.calls.count("current") == 2
    assert authority.calls.count("denylist") == 1
    observed_adapter_keys = {
        call[1] for call in authority.calls if isinstance(call, tuple)
    }
    assert observed_adapter_keys == {
        (
            authority.task_adapter_manifest.task_adapter_manifest_id,
            authority.verification_receipt.verification_receipt_id,
        )
        for authority in (
            matrix.stage_result.release_matrix_report.evaluation_plan.adapter_authorities
        )
    }
    fence = committed.stage_result.publication_authority_fence
    assert fence is not None
    assert fence.current_release_observation == (
        case.prepared.current_release_observation
    )
    assert fence.security_subject_ids == authority.denylist.checked_subject_ids
    with pytest.raises(
        ExpertPublicationEligibilityError,
        match="do not share one authority",
    ):
        publication_eligibility_security_subject_ids(
            input_snapshot=input_snapshot,
            stored_candidate=case.validation_store.reducer.candidate_store.read(
                committed.stage_result.candidate_id
            ),
            decision=committed.stage_result.promotion_decision,
            current_release_observation=fence.current_release_observation,
            task_adapter_trust_observations=(
                fence.task_adapter_trust_observations[:-1]
            ),
        )

    if case_name == "parent_approved":
        authority.current.observe_task_evaluation_current = lambda _scope_id: (
            pytest.fail("durable replay must not observe CURRENT")
        )
        authority.adapters.resolve_exact = lambda **_request: pytest.fail(
            "durable replay must not resolve adapters"
        )
        authority.denylist.observe_exact = lambda **_request: pytest.fail(
            "durable replay must not observe the denylist"
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            replays = tuple(
                executor.map(
                    lambda _position: authority.coordinator.publish(
                        candidate_id=matrix.snapshot.state.candidate_id,
                        release_matrix_stage_result_id=(
                            matrix.stage_result.stage_result_record_id
                        ),
                    ),
                    range(2),
                )
            )
        assert all(replay.replayed for replay in replays)
        assert all(
            replay.stage_result.to_json_bytes()
            == committed.stage_result.to_json_bytes()
            for replay in replays
        )

        reopened_store = ExpertValidationStore(
            case.validation_store.root,
            case.validation_store.state_root,
            case.validation_store.settings,
            case.validation_store.reducer,
        )
        restarted = ExpertPublicationEligibilityCoordinator(
            validation_store=reopened_store,
            current_release_authority=authority.current,
            task_adapter_authority=authority.adapters,
            security_denylist_authority=authority.denylist,
        ).publish(
            candidate_id=matrix.snapshot.state.candidate_id,
            release_matrix_stage_result_id=matrix.stage_result.stage_result_record_id,
        )
        assert restarted.replayed is True
        assert (
            restarted.stage_result.to_json_bytes()
            == committed.stage_result.to_json_bytes()
        )
        assert restarted.snapshot == committed.snapshot


def test_bootstrap_current_appearance_commits_generalized_authority_invalidation(
    terminal_cases,
):
    case = terminal_cases.bootstrap_current_mismatch
    expected = case.prepared.current_release_observation
    appeared_release_id = content_id("expert-base-release", {"appeared": True})
    authority = _coordinator(
        case,
        (_moved_current(expected, release_id=appeared_release_id),),
    )
    matrix = case.matrix_commit

    invalidated = authority.coordinator.publish(
        candidate_id=matrix.snapshot.state.candidate_id,
        release_matrix_stage_result_id=matrix.stage_result.stage_result_record_id,
    )

    assert invalidated.replayed is False
    assert invalidated.snapshot.state.promotion_state is ExpertPromotionState.FAILED
    assert invalidated.snapshot.state.next_stage is None
    assert invalidated.snapshot.state.reason == (
        "validation_current_release_authority_changed"
    )
    assert (
        invalidated.snapshot.transition.accepted_stage_result_record_ids
        == matrix.snapshot.transition.accepted_stage_result_record_ids
    )
    assert authority.calls == ["current"]


def test_approved_path_fails_closed_for_substitution_denial_and_local_head_race(
    terminal_cases,
):
    case = terminal_cases.adversarial
    matrix = case.matrix_commit
    authority = _coordinator(case)
    expected = case.prepared.current_release_observation
    candidate_id = matrix.snapshot.state.candidate_id
    matrix_result_id = matrix.stage_result.stage_result_record_id

    with pytest.raises(ExpertValidationStoreError, match="sealed"):
        case.validation_store.publish_publication_eligibility(matrix.stage_result)

    authority.calls.clear()
    authority.current_sequence.reset(
        (
            expected,
            _moved_current(expected, head_commit_sha="d" * 40),
        )
    )
    with pytest.raises(ExpertPublicationEligibilityError, match="CURRENT changed"):
        authority.coordinator.publish(
            candidate_id=candidate_id,
            release_matrix_stage_result_id=matrix_result_id,
        )
    assert case.validation_store.snapshot(candidate_id) == matrix.snapshot

    authority.calls.clear()
    authority.current_sequence.reset((expected, expected))
    tracking_resolve_exact = authority.adapters.resolve_exact

    def substituted_adapter(**request):
        authority.calls.append(
            (
                "adapter",
                (
                    request["task_adapter_manifest_id"],
                    request["verification_receipt_id"],
                ),
            )
        )
        return object()

    authority.adapters.resolve_exact = substituted_adapter
    with pytest.raises(ExpertPublicationEligibilityError, match="adapter differs"):
        authority.coordinator.publish(
            candidate_id=candidate_id,
            release_matrix_stage_result_id=matrix_result_id,
        )
    assert case.validation_store.snapshot(candidate_id) == matrix.snapshot
    authority.adapters.resolve_exact = tracking_resolve_exact

    authority.calls.clear()
    authority.current_sequence.reset((expected, expected))
    authority.denylist.denied = True
    with pytest.raises(ExpertPublicationEligibilityError, match="denylist"):
        authority.coordinator.publish(
            candidate_id=candidate_id,
            release_matrix_stage_result_id=matrix_result_id,
        )
    assert case.validation_store.snapshot(candidate_id) == matrix.snapshot

    authority.calls.clear()
    authority.current_sequence.reset((expected, expected))
    authority.denylist.denied = False

    def advance_local_head():
        authority.current.release_id = content_id(
            "expert-base-release",
            {"changed_during_publication": True},
        )
        case.validation_store.publish_current_release_authority_invalidation(
            candidate_id=candidate_id,
            expected_validation_state_id=matrix.snapshot.state.validation_state_id,
        )

    authority.denylist.callback = advance_local_head
    with pytest.raises(
        (ExpertPublicationEligibilityError, ExpertValidationCompareAndSwapError),
    ):
        authority.coordinator.publish(
            candidate_id=candidate_id,
            release_matrix_stage_result_id=matrix_result_id,
        )
    final_snapshot = case.validation_store.snapshot(candidate_id)
    assert final_snapshot is not None
    assert final_snapshot.state.promotion_state is ExpertPromotionState.FAILED
    assert (
        final_snapshot.transition.accepted_stage_result_record_ids
        == matrix.snapshot.transition.accepted_stage_result_record_ids
    )
