"""Deterministic M3 RunBundle to M4 evidence projection."""

from __future__ import annotations

from dataclasses import replace

import pytest

from kapso.cross_run.capture.pipeline import RunCaptureContext, RunCapturePipeline
from kapso.cross_run.catalog.projector import (
    BundleProjectionError,
    RunBundleProjector,
)
from kapso.cross_run.catalog.service import CrossRunCatalog
from kapso.cross_run.contracts import (
    ArtifactEnvironment,
    ComparisonStatus,
    CompletionState,
    EpisodeEvaluationStatus,
    EvaluationFingerprint,
    ExpertScopeContract,
    ExecutionStatus,
    InterventionStructure,
    ObjectiveDirection,
    TaskContextBinding,
)
from kapso.execution.search_strategies.generic.ideation import new_identifier
from test_cross_run_contracts import build_records, digest
from cross_run_capture_fixtures import make_capture_fixture
from test_ideation_domain import (
    analyzed_candidate,
    coding_agent_call,
    eligible_analysis,
    generated_idea,
    planned_batch,
    selection,
)


def _publish_bundle(fixture):
    pipeline = RunCapturePipeline(
        RunCaptureContext(fixture.request),
        fixture.settings,
    )
    stored = pipeline.capture_if_due(CompletionState.STOPPED, force=True)
    assert stored is not None
    return pipeline, stored


def _projector(fixture):
    return RunBundleProjector(fixture.settings.capture.score_comparison_tolerance)


@pytest.mark.parametrize("tolerance", (0.0, -1.0, float("nan"), True))
def test_projector_requires_explicit_positive_finite_score_tolerance(tolerance):
    with pytest.raises(ValueError, match="finite and positive"):
        RunBundleProjector(tolerance)


def _add_abandoned_prior_idea(fixture) -> str:
    archive = fixture.strategy.idea_archive
    batch_id = new_identifier("batch")
    idea_id = new_identifier("idea")
    parent = archive.state.batches[0].resolved_parents[0]
    batch = replace(
        planned_batch(),
        batch_id=batch_id,
        campaign_id=fixture.request.campaign_id,
        iteration_index=1,
        context_hash="9" * 64,
        evidence_snapshot=replace(
            planned_batch().evidence_snapshot,
            campaign_id=fixture.request.campaign_id,
        ),
        resolved_parents=(parent,),
    )
    idea = replace(
        generated_idea(),
        idea_id=idea_id,
        origin_batch_id=batch_id,
        resolved_parent=parent,
        assumptions=(),
        evidence_refs=("evaluation:frontier",),
    )
    archive.create_batch(batch, expected_revision=archive.revision)
    archive.add_ideas(
        batch_id,
        (idea,),
        generation_calls=(coding_agent_call(),),
        expected_revision=archive.revision,
    )
    archive.abandon_batch(
        batch_id,
        "The capture budget ended before analysis.",
        expected_revision=archive.revision,
    )
    fixture.save_checkpoint("running")
    return idea_id


def _add_measured_child_experiment(
    fixture,
    score: float,
    evaluator_id: str | None = None,
):
    archive = fixture.strategy.idea_archive
    parent_node = fixture.strategy.node_history[0]
    parent_commit = parent_node.evaluation_attempts[0].commit_sha
    parent = replace(
        generated_idea().resolved_parent,
        node_id=parent_node.node_id,
        branch_name=parent_node.branch_name,
        git_ref=parent_commit,
        materialized_ref=parent_commit,
        diff_base_ref=parent_commit,
        feedback_base_ref=parent_commit,
    )
    batch_id = new_identifier("batch")
    idea_id = new_identifier("idea")
    batch_template = planned_batch()
    batch = replace(
        batch_template,
        batch_id=batch_id,
        campaign_id=fixture.request.campaign_id,
        iteration_index=1,
        context_hash="8" * 64,
        evidence_snapshot=replace(
            batch_template.evidence_snapshot,
            campaign_id=fixture.request.campaign_id,
        ),
        resolved_parents=(parent,),
    )
    idea = replace(
        generated_idea(),
        idea_id=idea_id,
        origin_batch_id=batch_id,
        proposal="Add a parent-bound calibration layer and measure it.",
        resolved_parent=parent,
        parent_experiment_node_ids=(parent_node.node_id,),
    )
    archive.create_batch(batch, expected_revision=archive.revision)
    archive.add_ideas(
        batch_id,
        (idea,),
        generation_calls=(coding_agent_call(),),
        expected_revision=archive.revision,
    )
    analysis = replace(eligible_analysis(), idea_id=idea_id)
    archive.record_analyses(
        batch_id,
        (replace(analyzed_candidate(analysis), descriptor=idea.descriptor),),
        expected_revision=archive.revision,
    )
    decision = selection()
    archive.record_selection(
        batch_id,
        replace(
            decision,
            selected_idea_id=idea_id,
            dispositions=tuple(
                replace(disposition, idea_id=idea_id)
                for disposition in decision.dispositions
            ),
        ),
        selection_call=coding_agent_call(),
        expected_revision=archive.revision,
    )
    archive.link_experiment(
        idea_id,
        1,
        batch_id,
        expected_revision=archive.revision,
    )

    repository = fixture.strategy.workspace.repo
    branch_name = "generic_exp_1"
    repository.create_head(branch_name, parent_commit)
    repository.git.checkout(branch_name)
    (fixture.workspace / "solution.py").write_text("VALUE = 3\n", encoding="utf-8")
    repository.git.add(["solution.py"])
    repository.git.commit("-m", "measured child candidate")
    candidate_commit = repository.head.commit.hexsha
    repository.git.checkout("main")
    child = replace(
        parent_node,
        node_id=1,
        parent_node_id=0,
        execution_revision=0,
        idea_id=idea_id,
        selection_batch_id=batch_id,
        solution=idea.proposal,
        branch_name=branch_name,
        parent_branch_name=parent_node.branch_name,
        implementation_base_ref=parent_commit,
        diff_base_ref=parent_commit,
        feedback_base_ref=parent_commit,
        score=score,
        started_at="2026-07-20T12:30:00+00:00",
    )
    child.evaluation_attempts = [
        replace(
            parent_node.evaluation_attempts[0],
            commit_sha=candidate_commit,
            evaluator_id=(
                parent_node.evaluation_attempts[0].evaluator_id
                if evaluator_id is None
                else evaluator_id
            ),
            score=score,
            metrics={"quality": score},
        )
    ]
    child.metrics = {"quality": score}
    child.primary_metric = "quality"
    fixture.store.add_experiment(child)
    fixture.strategy.node_history = [parent_node, child]
    fixture.strategy.iteration_count = 2
    fixture.strategy.record_finalized_idea_outcome(child)
    fixture.save_checkpoint("running")
    return child


def test_posttrain_bundle_projects_every_idea_once_and_never_invents_an_effect(
    tmp_path,
):
    fixture = make_capture_fixture(tmp_path)
    prior_idea_id = _add_abandoned_prior_idea(fixture)
    (fixture.workspace / "run.txt").write_text(
        "sanitized observation\n",
        encoding="utf-8",
    )
    fixture.request = replace(fixture.request, run_log_paths=("run.txt",))
    _, stored = _publish_bundle(fixture)

    projector = _projector(fixture)
    projected = projector.project(stored)

    assert len(projected.episodes) == 1
    assert len(projected.prior_ideas) == 1
    assert projected.sanitation_report.status == "admitted"
    assert projected.episodes[0].sanitation_report_id == (
        projected.sanitation_report.report_id
    )
    assert projected.prior_ideas[0].source["idea_id"] == prior_idea_id
    assert projected.prior_ideas[0].assumptions == ()
    assert projected.prior_ideas[0].source_rationale == (
        "The capture budget ended before analysis."
    )
    assert projected.prior_ideas[0].source_evidence_refs == ("evaluation:frontier",)
    attempt = projected.episodes[0].attempts[0]
    assert attempt.execution_revision == 0
    assert attempt.score_of_record_fingerprint_id is not None
    assert attempt.comparison_status is ComparisonStatus.NOT_COMPARABLE
    assert attempt.source_parent_effect is None
    assert attempt.intervention_structure is InterventionStructure.UNDETERMINED
    assert attempt.intervention_ref is not None
    assert len(projected.episodes[0].safe_observation_refs) == 1
    assert projected.episodes[0].safe_observation_refs[0].relative_path in (
        projected.source_bundle.run_log_refs
    )
    assert {
        *(episode.source["idea_id"] for episode in projected.episodes),
        *(prior.source["idea_id"] for prior in projected.prior_ideas),
    } == {idea.idea_id for idea in fixture.strategy.idea_archive.state.ideas}
    assert projector.project(stored) == projected


def test_relbench_context_uses_the_same_domain_neutral_projection(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    posttrain_context = fixture.request.task_context_binding
    context = TaskContextBinding.mint(
        scope_contract_id=posttrain_context.scope_contract_id,
        scope_id=posttrain_context.scope_id,
        task_family_id="relational_tabular_prediction",
        task_adapter_id="relbench",
        capability_tags=("relational.training",),
        input_contract_fingerprint=digest("relbench-input"),
        target_contract_fingerprint=digest("relbench-target"),
        starting_artifact_refs=("artifact/relational-table",),
        method_fingerprint=digest("relbench-method"),
        toolchain_fingerprint=digest("relbench-toolchain"),
        dependency_runtime_fingerprint=digest("relbench-runtime"),
        budget_hardware_envelope={"accelerator": "cpu", "hours": 1},
        transfer_dimensions={
            "dataset_family": "relational",
            "runtime_family": "pytorch",
        },
    )
    environment = ArtifactEnvironment.mint(
        kapso_commit=fixture.request.artifact_environment.kapso_commit,
        expert_base_release_id=(
            fixture.request.artifact_environment.expert_base_release_id
        ),
        task_adapter_hash=digest("relbench-adapter"),
        dependency_lock_hash=(
            fixture.request.artifact_environment.dependency_lock_hash
        ),
    )
    posttrain_fingerprint = fixture.request.evaluation_fingerprints[0]
    fingerprint = EvaluationFingerprint.mint(
        benchmark_id="relbench",
        dataset_version="rel-stack-v1",
        split_version="entity-split-v1",
        evaluator_fingerprint=posttrain_fingerprint.evaluator_fingerprint,
        metric_name="quality",
        objective_direction=ObjectiveDirection.MAXIMIZE,
        fidelity="full",
        fraction=1.0,
        seed_or_replicate_ids=("seed-1",),
        aggregation_protocol="arithmetic-mean",
        judge_version=None,
    )
    fixture.request = replace(
        fixture.request,
        task_context_binding=context,
        artifact_environment=environment,
        evaluation_fingerprints=(fingerprint,),
    )
    _, stored = _publish_bundle(fixture)

    projected = _projector(fixture).project(stored)

    assert projected.source_bundle.task_context_binding.task_family_id == (
        "relational_tabular_prediction"
    )
    assert projected.episodes[0].task_context_binding.task_adapter_id == "relbench"
    assert projected.episodes[0].attempts[0].evaluation_fingerprints == (fingerprint,)


def test_relative_effect_requires_the_same_full_fingerprint_on_a_measured_parent(
    tmp_path,
):
    fixture = make_capture_fixture(tmp_path)
    child = _add_measured_child_experiment(fixture, 0.8)
    _, stored = _publish_bundle(fixture)

    projected = _projector(fixture).project(stored)
    child_episode = next(
        episode for episode in projected.episodes if episode.source["node_id"] == "1"
    )
    parent_episode = next(
        episode for episode in projected.episodes if episode.source["node_id"] == "0"
    )
    attempt = child_episode.attempts[0]

    assert child_episode.parent_episode_ref == parent_episode.episode_id
    assert attempt.comparison_status is ComparisonStatus.COMPARABLE
    assert attempt.source_parent_effect is not None
    assert attempt.source_parent_effect.candidate_value == child.score
    assert attempt.source_parent_effect.source_parent_value == 0.5
    assert attempt.source_parent_effect.raw_delta == pytest.approx(0.3)
    assert attempt.source_parent_effect.normalized_delta == pytest.approx(0.3)
    assert attempt.source_parent_effect.uncertainty is None


def test_measured_parent_under_another_full_fingerprint_has_no_relative_effect(
    tmp_path,
):
    fixture = make_capture_fixture(tmp_path)
    original_fingerprint = fixture.request.evaluation_fingerprints[0]
    fingerprint_values = original_fingerprint.to_dict()
    fingerprint_values.pop("evaluation_fingerprint_id")
    fingerprint_values["evaluator_fingerprint"] = digest("child-only-evaluator")
    child_fingerprint = EvaluationFingerprint.mint(**fingerprint_values)
    _add_measured_child_experiment(
        fixture,
        0.8,
        child_fingerprint.evaluator_fingerprint.removeprefix("sha256:"),
    )
    fixture.request = replace(
        fixture.request,
        evaluation_fingerprints=tuple(
            sorted(
                (original_fingerprint, child_fingerprint),
                key=lambda item: item.evaluation_fingerprint_id,
            )
        ),
    )
    _, stored = _publish_bundle(fixture)

    projected = _projector(fixture).project(stored)
    child_attempt = next(
        episode for episode in projected.episodes if episode.source["node_id"] == "1"
    ).attempts[0]

    assert child_attempt.comparison_status is ComparisonStatus.NOT_COMPARABLE
    assert child_attempt.source_parent_effect is None


def test_successor_capture_replaces_projection_and_supersedes_prior_payloads(
    tmp_path,
):
    fixture = make_capture_fixture(tmp_path)
    pipeline, first_bundle = _publish_bundle(fixture)
    first = _projector(fixture).project(first_bundle)
    prior_idea_id = _add_abandoned_prior_idea(fixture)

    successor_bundle = pipeline.capture_if_due(
        CompletionState.STOPPED,
        force=True,
    )
    assert successor_bundle is not None
    successor = _projector(fixture).project(successor_bundle, previous=first)

    assert successor.source_bundle.supersedes_bundle_id == first.source_bundle.bundle_id
    assert (
        successor.episodes[0].supersedes_projection_id == first.episodes[0].episode_id
    )
    assert successor.episodes[0].attempts == first.episodes[0].attempts
    assert successor.prior_ideas[0].source["idea_id"] == prior_idea_id
    assert successor.prior_ideas[0].supersedes_projection_id is None


def test_catalog_accepts_successor_manifest_reusing_historical_derivation_events(
    tmp_path,
):
    run_root = tmp_path / "run"
    run_root.mkdir()
    fixture = make_capture_fixture(run_root)
    pipeline, first_bundle = _publish_bundle(fixture)
    projector = _projector(fixture)
    first = projector.project(first_bundle)
    _add_abandoned_prior_idea(fixture)
    successor_bundle = pipeline.capture_if_due(
        CompletionState.STOPPED,
        force=True,
    )
    assert successor_bundle is not None
    successor = projector.project(successor_bundle, previous=first)
    scope_contract = next(
        record for record in build_records() if isinstance(record, ExpertScopeContract)
    )
    catalog = CrossRunCatalog(
        tmp_path / "catalog",
        scope_contract,
        fixture.settings.catalog,
    )

    first_generation = catalog.publish_projection(
        catalog.store.read_current(),
        first,
    ).generation
    successor_generation = catalog.publish_projection(
        first_generation,
        successor,
    ).generation

    assert successor_generation.generation_number == 2
    assert set(first.projection_manifest.derivation_object_ids).issubset(
        successor.projection_manifest.derivation_object_ids
    )
    assert set(successor_generation.bundle_frontier.values()) == {
        successor.source_bundle.bundle_id
    }


def test_recovery_revisions_remain_one_episode_with_every_historical_evaluator(
    tmp_path,
):
    fixture = make_capture_fixture(tmp_path)
    original = fixture.strategy.node_history[0]
    original_commit = original.evaluation_attempts[0].commit_sha
    repository = fixture.strategy.workspace.repo
    repository.git.checkout(original.branch_name)
    (fixture.workspace / "solution.py").write_text("VALUE = 2\n", encoding="utf-8")
    repository.git.add(["solution.py"])
    repository.git.commit("-m", "recover candidate under a successor evaluator")
    recovered_commit = repository.head.commit.hexsha
    repository.git.checkout("main")
    original_fingerprint = fixture.request.evaluation_fingerprints[0]
    fingerprint_values = original_fingerprint.to_dict()
    fingerprint_values.pop("evaluation_fingerprint_id")
    fingerprint_values["evaluator_fingerprint"] = digest("successor-evaluator")
    successor_fingerprint = EvaluationFingerprint.mint(**fingerprint_values)
    recovered = replace(original, execution_revision=1, score=0.9)
    recovered.implementation_base_ref = original_commit
    recovered.evaluation_attempts = [
        replace(
            original.evaluation_attempts[0],
            commit_sha=recovered_commit,
            evaluator_id=successor_fingerprint.evaluator_fingerprint.removeprefix(
                "sha256:"
            ),
            score=0.9,
            metrics={"quality": 0.9},
        )
    ]
    recovered.metrics = {"quality": 0.9}
    fixture.store.add_experiment(recovered)
    fixture.strategy.node_history = [recovered]
    fixture.save_checkpoint("running")
    fixture.request = replace(
        fixture.request,
        evaluation_fingerprints=tuple(
            sorted(
                (original_fingerprint, successor_fingerprint),
                key=lambda item: item.evaluation_fingerprint_id,
            )
        ),
    )
    _, stored = _publish_bundle(fixture)

    episode = _projector(fixture).project(stored).episodes[0]

    assert tuple(attempt.execution_revision for attempt in episode.attempts) == (0, 1)
    assert episode.terminal_attempt_revision == 1
    assert episode.attempts[0].evaluation_fingerprints == (original_fingerprint,)
    assert episode.attempts[1].evaluation_fingerprints == (successor_fingerprint,)
    assert episode.attempts[0].score_of_record_fingerprint_id == (
        original_fingerprint.evaluation_fingerprint_id
    )
    assert episode.attempts[1].score_of_record_fingerprint_id == (
        successor_fingerprint.evaluation_fingerprint_id
    )


def test_projector_uses_the_same_configured_score_predicate_as_capture(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    original = fixture.strategy.node_history[0]
    tolerance = fixture.settings.capture.score_comparison_tolerance
    recovered = replace(
        original,
        execution_revision=1,
        score=original.score + tolerance / 2.0,
    )
    recovered.implementation_base_ref = original.evaluation_attempts[0].commit_sha
    fixture.store.add_experiment(recovered)
    fixture.strategy.node_history = [recovered]
    fixture.save_checkpoint("running")
    _, stored = _publish_bundle(fixture)

    attempt = _projector(fixture).project(stored).episodes[0].attempts[1]

    assert attempt.score_of_record_fingerprint_id == (
        fixture.request.evaluation_fingerprints[0].evaluation_fingerprint_id
    )
    assert attempt.measurements["quality"] == original.score


@pytest.mark.parametrize(
    ("recoverable", "expected_status"),
    (
        (True, ExecutionStatus.INTERRUPTED),
        (False, ExecutionStatus.FAILED_TECHNICAL),
    ),
)
def test_failed_revision_is_retained_without_fake_evaluation(
    tmp_path,
    recoverable,
    expected_status,
):
    fixture = make_capture_fixture(tmp_path)
    original = fixture.strategy.node_history[0]
    original_commit = original.evaluation_attempts[0].commit_sha
    interrupted = replace(
        original,
        execution_revision=1,
        score=None,
        had_error=True,
        recoverable_error=recoverable,
        error_message="worker preempted",
        evaluation_attempts=[],
    )
    interrupted.implementation_base_ref = original_commit
    interrupted.metrics = {}
    interrupted.primary_metric = None
    fixture.store.add_experiment(interrupted)
    fixture.strategy.node_history = [interrupted]
    fixture.save_checkpoint("running")
    _, stored = _publish_bundle(fixture)

    attempts = _projector(fixture).project(stored).episodes[0].attempts

    assert tuple(attempt.execution_revision for attempt in attempts) == (0, 1)
    assert attempts[1].execution_status is expected_status
    assert attempts[1].evaluation_status is EpisodeEvaluationStatus.NOT_RUN
    assert attempts[1].evaluation_fingerprints == ()
    assert attempts[1].score_of_record_fingerprint_id is None
    assert attempts[1].source_parent_effect is None
    assert attempts[1].comparison_status is ComparisonStatus.INCONCLUSIVE


@pytest.mark.parametrize(
    ("evaluation_valid", "retain_measurement", "expected_status"),
    (
        (False, True, EpisodeEvaluationStatus.INVALID),
        (True, False, EpisodeEvaluationStatus.PARTIAL),
    ),
)
def test_non_valid_evaluation_remains_explicit_and_cannot_be_score_of_record(
    tmp_path,
    evaluation_valid,
    retain_measurement,
    expected_status,
):
    fixture = make_capture_fixture(tmp_path)
    original = fixture.strategy.node_history[0]
    original_commit = original.evaluation_attempts[0].commit_sha
    revision = replace(
        original,
        execution_revision=1,
        score=None,
        evaluation_valid=evaluation_valid,
        evaluation_attempts=(
            list(original.evaluation_attempts) if retain_measurement else []
        ),
    )
    revision.implementation_base_ref = original_commit
    revision.metrics = {"quality": 0.5} if retain_measurement else {}
    revision.primary_metric = "quality" if retain_measurement else None
    fixture.store.add_experiment(revision)
    fixture.strategy.node_history = [revision]
    fixture.save_checkpoint("running")
    _, stored = _publish_bundle(fixture)

    attempt = _projector(fixture).project(stored).episodes[0].attempts[1]

    assert attempt.evaluation_status is expected_status
    assert len(attempt.evaluation_fingerprints) == int(retain_measurement)
    assert attempt.score_of_record_fingerprint_id is None
    assert attempt.comparison_status is ComparisonStatus.INCONCLUSIVE
    assert attempt.source_parent_effect is None


def test_projector_rechecks_every_sanitized_object_checksum(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    _, stored = _publish_bundle(fixture)

    class CorruptReader:
        manifest = stored.manifest

        @staticmethod
        def read_ref(relative_path):
            payload = stored.read_ref(relative_path)
            if relative_path == stored.manifest.idea_archive_ref:
                return payload + b"corruption"
            return payload

    with pytest.raises(BundleProjectionError, match="checksum mismatch"):
        _projector(fixture).project(CorruptReader())


def test_projection_rejects_missing_or_wrong_supersession_frontier(tmp_path):
    fixture = make_capture_fixture(tmp_path)
    pipeline, first_bundle = _publish_bundle(fixture)
    first = _projector(fixture).project(first_bundle)
    _add_abandoned_prior_idea(fixture)
    successor_bundle = pipeline.capture_if_due(
        CompletionState.STOPPED,
        force=True,
    )
    assert successor_bundle is not None

    with pytest.raises(BundleProjectionError, match="prior frontier"):
        _projector(fixture).project(successor_bundle)
    other_root = tmp_path / "other-run"
    other_root.mkdir()
    wrong_fixture = make_capture_fixture(other_root)
    wrong_fixture.request = replace(
        wrong_fixture.request,
        configuration_fingerprint=digest("other-run-config"),
    )
    _, wrong_bundle = _publish_bundle(wrong_fixture)
    wrong_previous = _projector(wrong_fixture).project(wrong_bundle)
    with pytest.raises(BundleProjectionError, match="does not supersede"):
        _projector(fixture).project(successor_bundle, previous=wrong_previous)
