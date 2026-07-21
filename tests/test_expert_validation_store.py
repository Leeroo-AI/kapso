from dataclasses import replace
from pathlib import Path

import pytest

from kapso.cross_run.expert.validation import (
    ExpertCandidateEligibilityEvaluator,
    ExpertEvaluatorRunBuilder,
    ExpertValidationError,
)
from kapso.cross_run.expert.validation_store import (
    ExpertEvaluatorResultRecord,
    ExpertValidationCompareAndSwapError,
    ExpertValidationStore,
    ExpertValidationStoreError,
)
from kapso.cross_run.contracts import (
    ExpertEvaluatorOutcome,
    ExpertPromotionState,
    ExpertValidationStage,
)
from test_expert_candidate_store import candidate_store
from test_expert_candidates import bootstrap_candidate_closure
from test_expert_validation import (
    _content_id,
    _eligibility_evaluator,
    _task_adapter,
    _validation_reducer,
    _validation_settings,
)
from test_expert_source_replay import _CandidateReader
from test_expert_source_replay_request import _request_fixture


def _validation_store(tmp_path, settings, reducer):
    state_root = tmp_path / "validation-state"
    if not state_root.exists():
        state_root.mkdir(mode=0o700)
    return ExpertValidationStore(
        (state_root / "validation").resolve(),
        state_root.resolve(),
        settings,
        reducer,
    )


def _candidate_and_eligibility(tmp_path, current_release_id=None):
    candidates = candidate_store(tmp_path)
    stored = candidates.persist(bootstrap_candidate_closure())
    adapter = _task_adapter(stored.closure)
    settings = _validation_settings()
    eligibility = _eligibility_evaluator(
        settings,
        candidates,
        adapter,
        current_release_id,
    ).decide(
        candidate_id=stored.closure.manifest.candidate_id,
    )
    return candidates, stored, adapter, settings, eligibility


def _result(settings, attempt, stage, outcome):
    return ExpertEvaluatorRunBuilder(settings).build(
        attempt=attempt,
        stage=stage,
        exact_additional_input_ids=(),
        output_payloads={"result.json": b'{"completed":true}'},
        measurements={},
        costs={},
        duration_seconds=1.0,
        outcome=outcome,
        signature="test-signature",
    )


def test_start_round_trip_and_lost_response_replay_are_exact(tmp_path):
    candidates, stored, adapter, settings, eligibility = _candidate_and_eligibility(
        tmp_path
    )
    reducer = _validation_reducer(
        settings,
        adapter,
        candidate_store=candidates,
    )
    store = _validation_store(tmp_path, settings, reducer)

    committed = store.publish_start(
        expected_transition_id=None,
        eligibility=eligibility,
    )
    reopened = _validation_store(tmp_path, settings, reducer)
    current = reopened.current(stored.closure.manifest.candidate_id)
    replay = reopened.publish_start(
        expected_transition_id=None,
        eligibility=eligibility,
    )

    assert committed.replayed is False
    assert current is not None
    assert current.latest_attempt == committed.snapshot.latest_attempt
    assert current.state == committed.snapshot.state
    assert replay.replayed is True
    assert replay.snapshot.transition == committed.snapshot.transition


def test_passed_results_reopen_as_the_exact_ordered_reducer_prefix(tmp_path):
    candidates, stored, adapter, settings, eligibility = _candidate_and_eligibility(
        tmp_path
    )
    reducer = _validation_reducer(
        settings,
        adapter,
        candidate_store=candidates,
    )
    store = _validation_store(tmp_path, settings, reducer)
    started = store.publish_start(
        expected_transition_id=None,
        eligibility=eligibility,
    ).snapshot
    assert started.latest_attempt is not None
    first = _result(
        settings,
        started.latest_attempt,
        ExpertValidationStage.CONTRACT_SCHEMA,
        ExpertEvaluatorOutcome.PASSED,
    )
    after_first = store.publish_result(
        candidate_id=started.state.candidate_id,
        expected_transition_id=started.transition.transition_id,
        result=first,
    ).snapshot
    reopened = _validation_store(tmp_path, settings, reducer)
    recovered = reopened.snapshot(started.state.candidate_id)
    assert recovered is not None
    assert recovered.accepted_results == (first,)

    second = _result(
        settings,
        started.latest_attempt,
        ExpertValidationStage.IDENTITY_SECRETS_LICENSE_DEPENDENCY,
        ExpertEvaluatorOutcome.PASSED,
    )
    after_second = reopened.publish_result(
        candidate_id=started.state.candidate_id,
        expected_transition_id=after_first.transition.transition_id,
        result=second,
    ).snapshot

    assert after_second.accepted_results == (first, second)
    assert len(after_second.state.accepted_evaluator_evidence) == 2


def test_failed_ineligible_retry_preserves_historical_attempt_across_reopen(
    tmp_path,
):
    candidates, stored, adapter, settings, eligible = _candidate_and_eligibility(
        tmp_path
    )
    store = _validation_store(
        tmp_path,
        settings,
        _validation_reducer(settings, adapter, candidate_store=candidates),
    )
    started = store.publish_start(
        expected_transition_id=None,
        eligibility=eligible,
    ).snapshot
    assert started.latest_attempt is not None
    failed = store.publish_result(
        candidate_id=started.state.candidate_id,
        expected_transition_id=started.transition.transition_id,
        result=_result(
            settings,
            started.latest_attempt,
            ExpertValidationStage.CONTRACT_SCHEMA,
            ExpertEvaluatorOutcome.CANDIDATE_FAILED,
        ),
    ).snapshot
    current_release_id = _content_id("temporarily-current-release")
    evolved_settings = replace(
        settings,
        state_path=".kapso/cross-run/evolved-expert-validation",
    )
    candidates = candidate_store(tmp_path)
    stale = _eligibility_evaluator(
        evolved_settings,
        candidates,
        adapter,
        current_release_id,
    ).decide(
        candidate_id=stored.closure.manifest.candidate_id,
    )
    stale_store = _validation_store(
        tmp_path,
        evolved_settings,
        _validation_reducer(
            evolved_settings,
            adapter,
            candidate_store=candidates,
            current_release_id=current_release_id,
        ),
    )
    ineligible = stale_store.publish_start(
        expected_transition_id=failed.transition.transition_id,
        eligibility=stale,
    ).snapshot
    assert ineligible.state.promotion_state is ExpertPromotionState.INELIGIBLE
    assert ineligible.latest_attempt == started.latest_attempt
    assert (
        _validation_store(
            tmp_path,
            evolved_settings,
            _validation_reducer(
                evolved_settings,
                adapter,
                candidate_store=candidates,
                current_release_id=current_release_id,
            ),
        ).snapshot(ineligible.state.candidate_id)
        == ineligible
    )

    retry_store = _validation_store(
        tmp_path,
        settings,
        _validation_reducer(settings, adapter, candidate_store=candidates),
    )
    retry = retry_store.publish_start(
        expected_transition_id=ineligible.transition.transition_id,
        eligibility=eligible,
    ).snapshot

    assert retry.latest_attempt is not None
    assert retry.latest_attempt.attempt_number == 2
    assert (
        retry.latest_attempt.predecessor_attempt_id
        == started.latest_attempt.validation_attempt_id
    )


def test_stale_result_cannot_fork_or_rewind_the_candidate_head(tmp_path):
    candidates, stored, adapter, settings, eligibility = _candidate_and_eligibility(
        tmp_path
    )
    store = _validation_store(
        tmp_path,
        settings,
        _validation_reducer(settings, adapter, candidate_store=candidates),
    )
    started = store.publish_start(
        expected_transition_id=None,
        eligibility=eligibility,
    ).snapshot
    assert started.latest_attempt is not None
    passed = _result(
        settings,
        started.latest_attempt,
        ExpertValidationStage.CONTRACT_SCHEMA,
        ExpertEvaluatorOutcome.PASSED,
    )
    advanced = store.publish_result(
        candidate_id=started.state.candidate_id,
        expected_transition_id=started.transition.transition_id,
        result=passed,
    ).snapshot
    competing = _result(
        settings,
        started.latest_attempt,
        ExpertValidationStage.CONTRACT_SCHEMA,
        ExpertEvaluatorOutcome.INCONCLUSIVE,
    )

    with pytest.raises(ExpertValidationCompareAndSwapError, match="head changed"):
        store.publish_result(
            candidate_id=started.state.candidate_id,
            expected_transition_id=started.transition.transition_id,
            result=competing,
        )

    assert store.snapshot(started.state.candidate_id).transition == (
        advanced.transition
    )


def test_orphaned_objects_retry_and_missing_referenced_object_fails_loud(
    tmp_path,
    monkeypatch,
):
    candidates, stored, adapter, settings, eligibility = _candidate_and_eligibility(
        tmp_path
    )
    reducer = _validation_reducer(
        settings,
        adapter,
        candidate_store=candidates,
    )
    store = _validation_store(tmp_path, settings, reducer)

    def interrupt_journal_write(journal):
        raise OSError("simulated journal interruption")

    monkeypatch.setattr(store, "_write_journal_unlocked", interrupt_journal_write)
    with pytest.raises(OSError, match="interruption"):
        store.publish_start(
            expected_transition_id=None,
            eligibility=eligibility,
        )

    recovered = _validation_store(tmp_path, settings, reducer)
    committed = recovered.publish_start(
        expected_transition_id=None,
        eligibility=eligibility,
    ).snapshot
    state_path = recovered._object_path(
        committed.state.validation_state_id,
        create_namespace=False,
    )
    Path(state_path).unlink()

    with pytest.raises(ExpertValidationStoreError, match="regular file"):
        recovered.snapshot(committed.state.candidate_id)


def test_result_record_identity_includes_the_selected_signature_envelope(tmp_path):
    candidates, stored, adapter, settings, eligibility = _candidate_and_eligibility(
        tmp_path
    )
    started = _validation_reducer(
        settings,
        adapter,
        candidate_store=candidates,
    ).start_from_predecessor(
        eligibility=eligibility,
        predecessor=None,
    )
    assert started.attempt is not None
    result = _result(
        settings,
        started.attempt,
        ExpertValidationStage.CONTRACT_SCHEMA,
        ExpertEvaluatorOutcome.PASSED,
    )
    original = ExpertEvaluatorResultRecord.mint(
        evaluator_run=result.evaluator_run,
        attestation_envelope=result.attestation_envelope,
    )
    rotated = ExpertEvaluatorResultRecord.mint(
        evaluator_run=result.evaluator_run,
        attestation_envelope=replace(
            result.attestation_envelope,
            signature="rotated-signature",
        ),
    )

    assert original.evaluator_run == rotated.evaluator_run
    assert original.attestation_envelope.attestation == (
        rotated.attestation_envelope.attestation
    )
    assert original.evaluator_result_record_id != rotated.evaluator_result_record_id


def test_parent_authority_change_terminates_attempt_and_requires_successor_candidate(
    tmp_path,
):
    fixture = _request_fixture(tmp_path)
    store = fixture.validation_store
    started = store.snapshot(fixture.attempt.candidate_id)
    assert started is not None
    with pytest.raises(ExpertValidationError, match="has not changed"):
        store.publish_parent_authority_invalidation(
            candidate_id=started.state.candidate_id,
            expected_validation_state_id=started.state.validation_state_id,
        )
    fixture.current_release_provider.release_id = _content_id(
        "successor-parent-release"
    )

    invalidated_result = store.publish_parent_authority_invalidation(
        candidate_id=started.state.candidate_id,
        expected_validation_state_id=started.state.validation_state_id,
    )
    invalidated = invalidated_result.snapshot
    replayed = store.publish_parent_authority_invalidation(
        candidate_id=started.state.candidate_id,
        expected_validation_state_id=started.state.validation_state_id,
    )

    assert invalidated_result.replayed is False
    assert replayed.replayed is True
    assert replayed.snapshot == invalidated
    assert invalidated.state.promotion_state is ExpertPromotionState.FAILED
    assert invalidated.state.validation_attempt_id == (
        started.latest_attempt.validation_attempt_id
    )
    assert invalidated.transition.transition_authority_invalidation_id == (
        invalidated.state.transition_evidence_id
    )
    stale_eligibility = ExpertCandidateEligibilityEvaluator(
        fixture.settings,
        _CandidateReader(fixture.stored),
        fixture.adapter_provider,
        fixture.current_release_provider,
    ).decide(candidate_id=started.state.candidate_id)
    reenrolled = store.publish_start(
        expected_transition_id=invalidated.transition.transition_id,
        eligibility=stale_eligibility,
    ).snapshot
    replayed_after_reenrollment = store.publish_parent_authority_invalidation(
        candidate_id=started.state.candidate_id,
        expected_validation_state_id=started.state.validation_state_id,
    )

    assert reenrolled.state.promotion_state is ExpertPromotionState.INELIGIBLE
    assert reenrolled.latest_attempt == started.latest_attempt
    assert replayed_after_reenrollment.replayed is True
    assert replayed_after_reenrollment.snapshot == invalidated


def test_store_rejects_a_reducer_with_different_configuration(tmp_path):
    fixture = _request_fixture(tmp_path)
    mismatched_settings = replace(
        fixture.settings,
        state_path="mismatched-validation-state",
    )
    mismatched_root = tmp_path / "mismatched-store"
    mismatched_root.mkdir(mode=0o700)

    with pytest.raises(
        ExpertValidationStoreError,
        match="reducer differs from store configuration",
    ):
        ExpertValidationStore(
            (mismatched_root / "validation").resolve(),
            mismatched_root.resolve(),
            mismatched_settings,
            fixture.validation_store.reducer,
        )
