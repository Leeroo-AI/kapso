from __future__ import annotations

from dataclasses import replace

import pytest

from kapso.core.embedding_contracts import EmbeddingRecord, complete_input_hash
from kapso.cross_run.canonical import content_id, tree_or_blob_digest
from kapso.cross_run.capture.revision_projection import ExecutionRevisionProjection
from kapso.cross_run.contracts import (
    EpisodeEvaluationStatus,
    ExecutionStatus,
    TaskContextBinding,
)
from kapso.cross_run.launch.derived_state_contracts import (
    RunDerivedStateGeneration,
    RunStateAuthority,
    RunStateLayout,
    RunStatePayloadTransition,
)
from kapso.cross_run.launch.derived_state_bundle import RunDerivedStateBundle
from kapso.cross_run.launch.checkpoint_contracts import (
    RunStrategyKind,
    RunStrategyState,
)
from kapso.cross_run.launch.run_state_projection import (
    ReconciledRunStateProjection,
    RunStateProjectionError,
)
from kapso.execution.memories.experiment_memory.projection import (
    build_experiment_history_genesis,
    project_records,
)
from kapso.execution.evaluation_integrity import AGENT_GENERATED
from kapso.execution.fidelity import EvaluationAttempt
from kapso.execution.search_strategies.node import SearchNode
from kapso.execution.search_strategies.generic.ideation.archive import IdeaArchive
from kapso.execution.search_strategies.generic.ideation.archive_projection import (
    build_archive_genesis,
    project_outcome,
)
from kapso.execution.search_strategies.generic.ideation.types import (
    EvaluationStatus,
    IdeaOutcome,
    IdeationCrossRunIdentity,
    ImplementationStatus,
    ParentPlanKind,
)
from test_ideation_domain import (
    BATCH_ID,
    IDEA_ID,
    analyzed_candidate,
    coding_agent_call,
    eligible_analysis,
    generated_idea,
    planned_batch,
    selection,
)
from test_prior_knowledge_gate import citable_access_materialization
from test_launch_resolver import resolver_case
from test_launch_resume_contracts import _bootstrap_pin

_RUN_ID = "run_test"
_CAMPAIGN_ID = "campaign_test"
_TIMESTAMP = "2026-07-24T00:00:00Z"
_EMBEDDING_PROVIDER = "openai"
_EMBEDDING_MODEL = "test-embedding"
_EMBEDDING_DIMENSIONS = 1
_EMBEDDING_CANONICALIZER_VERSION = "kapso.embedding_input.v1"
_TREE_BASE_COMMIT = "b" * 40
_EMBEDDING_SPACE_ID = content_id(
    "embedding-space",
    {
        "provider": _EMBEDDING_PROVIDER,
        "model": _EMBEDDING_MODEL,
        "dimensions": _EMBEDDING_DIMENSIONS,
        "canonicalizer_version": _EMBEDDING_CANONICALIZER_VERSION,
    },
)


def _identifier(namespace: str, value: str) -> str:
    return content_id(namespace, {"value": value})


def _embedding(solution: str, value: float) -> EmbeddingRecord:
    return EmbeddingRecord(
        provider=_EMBEDDING_PROVIDER,
        model=_EMBEDDING_MODEL,
        dimensions=_EMBEDDING_DIMENSIONS,
        canonicalizer_version=_EMBEDDING_CANONICALIZER_VERSION,
        input_hash=complete_input_hash(solution),
        vector=tuple(value for _ in range(_EMBEDDING_DIMENSIONS)),
    )


def _layout(strategy_kind: str) -> RunStateLayout:
    paths = {
        RunStateAuthority.EXPERIMENT_HISTORY: ".kapso/experiment_history.json",
        RunStateAuthority.EXECUTION_JOURNAL: ".kapso/execution_events.jsonl",
    }
    if strategy_kind == "generic":
        paths[RunStateAuthority.IDEA_ARCHIVE] = ".kapso/idea_archive.json"
    return RunStateLayout.build(
        strategy_kind=strategy_kind,
        authority_paths=paths,
    )


def _evaluation_integrity() -> dict:
    return {
        "provenance": AGENT_GENERATED,
        "manifest": {},
        "fingerprint": None,
    }


def _strategy_state(
    strategy_kind: str,
    *,
    campaign_id: str = _CAMPAIGN_ID,
    archive=None,
    nodes: tuple[SearchNode, ...] = (),
    history_ids: tuple[int, ...] = (),
) -> RunStrategyState:
    if strategy_kind == "generic":
        state = {
            "idea_archive_snapshot": archive.to_dict(),
            "node_history": [node.to_dict() for node in nodes],
            "iteration_count": len(nodes),
            "previous_errors": [],
            "evaluation_integrity": _evaluation_integrity(),
            "scores_evaluator_id": "",
            "evaluator_transition": None,
        }
        kind = RunStrategyKind.GENERIC
    else:
        tree_nodes = []
        for node in nodes:
            projected = node.to_dict()
            projected.update(
                {
                    "parent_id": node.parent_node_id,
                    "children_ids": [
                        candidate.node_id
                        for candidate in nodes
                        if candidate.parent_node_id == node.node_id
                    ],
                    "is_terminated": False,
                    "is_root": node.parent_node_id is None,
                    "node_event_history": [],
                    "ideation_repo_memory_sections_consulted": [],
                }
            )
            tree_nodes.append(projected)
        state = {
            "nodes": tree_nodes,
            "node_history_ids": list(history_ids),
            "experimentation_count": 0,
            "previous_errors": [],
            "evaluation_integrity": _evaluation_integrity(),
        }
        kind = RunStrategyKind.BENCHMARK_TREE_SEARCH
    return RunStrategyState.build(
        strategy_kind=kind,
        campaign_id=campaign_id,
        state=state,
    )


def _projection(
    strategy_kind: str,
    *,
    run_id: str = _RUN_ID,
    campaign_id: str = _CAMPAIGN_ID,
    embedding_space_id: str = _EMBEDDING_SPACE_ID,
    embedding_provider: str = _EMBEDDING_PROVIDER,
    embedding_model: str = _EMBEDDING_MODEL,
    embedding_dimensions: int = _EMBEDDING_DIMENSIONS,
    embedding_canonicalizer_version: str = _EMBEDDING_CANONICALIZER_VERSION,
) -> ReconciledRunStateProjection:
    generic = strategy_kind == "generic"
    history = build_experiment_history_genesis(
        run_id=run_id,
        campaign_id=campaign_id,
        embedding_space_id=embedding_space_id,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        embedding_canonicalizer_version=embedding_canonicalizer_version,
        objective_direction="maximize",
        require_idea_links=generic,
    )
    journal = ExecutionRevisionProjection(
        run_id=run_id,
        campaign_id=campaign_id,
        require_contiguous_node_ids=generic,
    )
    archive = (
        build_archive_genesis(
            campaign_id=campaign_id,
            created_at=_TIMESTAMP,
        )
        if generic
        else None
    )
    return ReconciledRunStateProjection(
        strategy_state=_strategy_state(
            strategy_kind,
            campaign_id=campaign_id,
            archive=archive,
        ),
        experiment_history=history,
        execution_journal=journal,
        idea_archive=archive,
    )


@pytest.fixture
def bootstrap_pin(resolver_case, tmp_path):
    return _bootstrap_pin(resolver_case, tmp_path)


def _resolved_projection(strategy_kind, bootstrap_pin, resolver_case):
    embeddings = bootstrap_pin.launch_manifest.experiment_embedding_space
    receipt = bootstrap_pin.installation_receipt
    return _projection(
        strategy_kind,
        run_id=receipt.run_id,
        campaign_id=receipt.campaign_id,
        embedding_space_id=embeddings.embedding_space_id,
        embedding_provider=embeddings.provider,
        embedding_model=embeddings.model,
        embedding_dimensions=embeddings.dimensions,
        embedding_canonicalizer_version=embeddings.canonicalizer_version,
    )


def _tree_artifact_refs(
    record,
    *,
    candidate_commit: str | None = None,
    parent_branch: str = "main",
    base_commit: str = _TREE_BASE_COMMIT,
) -> dict[str, str]:
    refs = {
        "branch": record.branch_name,
        "parent_branch": parent_branch,
        "implementation_base": base_commit,
        "diff_base": base_commit,
        "feedback_base": base_commit,
    }
    if candidate_commit is not None:
        refs.update(
            {
                "candidate_commit": candidate_commit,
                "candidate_ref": (
                    f"refs/kapso/execution-revisions/{_RUN_ID}/"
                    f"node-{record.node_id}/revision-{record.execution_revision}"
                ),
                "implementation_base_commit": base_commit,
                "diff_base_commit": base_commit,
                "feedback_base_commit": base_commit,
            }
        )
    for position, attempt in enumerate(record.evaluation_attempts):
        refs[f"evaluation_commit_{position}"] = attempt.commit_sha
    return refs


def _tree_projection_with_nodes(
    base: ReconciledRunStateProjection,
    nodes: tuple[SearchNode, ...],
    embedding_value_by_node: dict[int, float],
) -> ReconciledRunStateProjection:
    nodes = tuple(
        replace(
            node,
            parent_branch_name="main",
            implementation_base_ref=_TREE_BASE_COMMIT,
            diff_base_ref=_TREE_BASE_COMMIT,
            feedback_base_ref=_TREE_BASE_COMMIT,
        )
        for node in nodes
    )
    history = project_records(
        predecessor=base.experiment_history,
        nodes=nodes,
        embeddings_by_node_revision={
            (node.node_id, node.execution_revision): _embedding(
                node.solution,
                embedding_value_by_node[node.node_id],
            )
            for node in nodes
        },
    )
    journal = base.execution_journal
    for position, record in enumerate(history.records):
        journal, _ = journal.append_projection(
            node_id=record.node_id,
            execution_revision=record.execution_revision,
            idea_id=None,
            selection_batch_id=None,
            parent_node_id=record.parent_node_id,
            started_at=record.timestamp,
            recorded_at=f"2026-07-24T00:0{position + 1}:00Z",
            execution_status=ExecutionStatus.COMPLETED,
            evaluation_status=EpisodeEvaluationStatus.PARTIAL,
            evaluator_fingerprint_ids=(),
            measurements={},
            feedback=record.feedback,
            technical_difficulties=record.technical_difficulties,
            artifact_refs=_tree_artifact_refs(record),
            projection=record.to_dict(),
        )
    return ReconciledRunStateProjection(
        strategy_state=_strategy_state(
            "benchmark_tree_search",
            campaign_id=base.experiment_history.campaign_id,
            nodes=nodes,
            history_ids=tuple(node.node_id for node in nodes),
        ),
        experiment_history=history,
        execution_journal=journal,
        idea_archive=None,
    )


def _linked_generic_archive(tmp_path):
    archive = IdeaArchive(tmp_path / "idea_archive.json", "campaign-alpha")
    materialization = citable_access_materialization()
    task_context = TaskContextBinding.from_dict(
        materialization.prior_knowledge_snapshot.selected_records[0]["payload"][
            "task_context_binding"
        ]
    )
    identity = IdeationCrossRunIdentity(
        launch_manifest_id=content_id(
            "launch-manifest",
            {"projection": "test"},
        ),
        scope_contract_id=task_context.scope_contract_id,
        knowledge_snapshot_id=(
            materialization.prior_knowledge_snapshot.source_snapshot_id
        ),
        expert_base_release_id=content_id(
            "expert-base-release",
            {"projection": "test"},
        ),
        embedding_space_id=_EMBEDDING_SPACE_ID,
        task_context_binding=task_context,
    )
    idea = generated_idea()
    idea = replace(
        idea,
        resolved_parent=replace(idea.resolved_parent, branch_name="main"),
    )
    archive.create_batch(
        replace(
            planned_batch(),
            cross_run_identity=identity,
            prior_knowledge=materialization,
        ),
        expected_revision=0,
    )
    archive.add_ideas(
        BATCH_ID,
        (idea,),
        generation_calls=(coding_agent_call(),),
        expected_revision=1,
    )
    archive.record_analyses(
        BATCH_ID,
        (analyzed_candidate(eligible_analysis()),),
        expected_revision=2,
    )
    archive.record_selection(
        BATCH_ID,
        selection(),
        selection_call=coding_agent_call(),
        expected_revision=3,
    )
    archive.link_experiment(IDEA_ID, 0, BATCH_ID, expected_revision=4)
    return archive.state


def test_reconciled_genesis_builds_and_decodes_exact_bundle(
    bootstrap_pin,
    resolver_case,
) -> None:
    strategy_kind = "generic"
    projection = _resolved_projection(
        strategy_kind,
        bootstrap_pin,
        resolver_case,
    )
    bundle = projection.build_bundle(
        bootstrap_pin=bootstrap_pin,
        run_state_layout=_layout(strategy_kind),
        predecessor_checkpoint_head_id=_identifier(
            "run-checkpoint-head",
            "initial-head",
        ),
        predecessor_checkpoint_id=None,
        predecessor_evidence_id=None,
        target_evidence_id=_identifier(
            "run-derivative-evidence",
            "target-evidence",
        ),
        predecessor_bundle=None,
        predecessor_strategy_state=None,
    )

    restored = ReconciledRunStateProjection.from_bundle(
        bundle,
        strategy_state=projection.strategy_state,
        bootstrap_pin=bootstrap_pin,
    )

    assert restored == projection
    assert bundle == type(bundle).from_bytes(bundle.to_bytes())
    assert set(bundle.payload_by_relative_path()) == {
        binding.relative_path for binding in bundle.generation.run_state_layout.bindings
    }


def test_successor_bundle_names_exact_predecessor_payload_frontier(
    bootstrap_pin,
    resolver_case,
) -> None:
    projection = _resolved_projection(
        "generic",
        bootstrap_pin,
        resolver_case,
    )
    layout = _layout("generic")
    predecessor_evidence_id = _identifier(
        "run-derivative-evidence",
        "first-evidence",
    )
    first = projection.build_bundle(
        bootstrap_pin=bootstrap_pin,
        run_state_layout=layout,
        predecessor_checkpoint_head_id=_identifier(
            "run-checkpoint-head",
            "initial-head",
        ),
        predecessor_checkpoint_id=None,
        predecessor_evidence_id=None,
        target_evidence_id=predecessor_evidence_id,
        predecessor_bundle=None,
        predecessor_strategy_state=None,
    )

    second = projection.build_bundle(
        bootstrap_pin=bootstrap_pin,
        run_state_layout=layout,
        predecessor_checkpoint_head_id=_identifier(
            "run-checkpoint-head",
            "current-head",
        ),
        predecessor_checkpoint_id=_identifier(
            "run-checkpoint",
            "current-checkpoint",
        ),
        predecessor_evidence_id=predecessor_evidence_id,
        target_evidence_id=_identifier(
            "run-derivative-evidence",
            "next-evidence",
        ),
        predecessor_bundle=first,
        predecessor_strategy_state=projection.strategy_state,
    )

    assert tuple(
        (
            transition.predecessor_digest,
            transition.predecessor_revision,
            transition.predecessor_size_bytes,
        )
        for transition in second.generation.payload_transitions
    ) == tuple(
        (
            transition.target_digest,
            transition.target_revision,
            transition.target_size_bytes,
        )
        for transition in first.generation.payload_transitions
    )


def test_successor_projection_rejects_rewritten_old_event_and_embedding() -> None:
    base = _projection("benchmark_tree_search")
    first_node = SearchNode(
        node_id=0,
        solution="first candidate",
        branch_name="candidate-0",
        feedback="measured",
        started_at=_TIMESTAMP,
    )
    predecessor_projection = _tree_projection_with_nodes(
        base,
        (first_node,),
        {0: 0.1},
    )
    second_node = SearchNode(
        node_id=1,
        solution="second candidate",
        branch_name="candidate-1",
        feedback="measured",
        started_at=_TIMESTAMP,
    )
    rewritten_projection = _tree_projection_with_nodes(
        base,
        (first_node, second_node),
        {0: 0.9, 1: 0.2},
    )

    with pytest.raises(RunStateProjectionError, match="rewrote predecessor"):
        rewritten_projection.require_predecessor(predecessor_projection)


def test_projection_rejects_cross_identity_and_strategy_authority_mix() -> None:
    generic = _projection("generic")
    wrong_journal = ExecutionRevisionProjection(
        run_id="run_other",
        campaign_id=_CAMPAIGN_ID,
        require_contiguous_node_ids=True,
    )

    with pytest.raises(RunStateProjectionError, match="identities differ"):
        ReconciledRunStateProjection(
            strategy_state=generic.strategy_state,
            experiment_history=generic.experiment_history,
            execution_journal=wrong_journal,
            idea_archive=generic.idea_archive,
        )
    with pytest.raises(RunStateProjectionError, match="cannot contain"):
        ReconciledRunStateProjection(
            strategy_state=_strategy_state(
                "benchmark_tree_search",
            ),
            experiment_history=generic.experiment_history,
            execution_journal=generic.execution_journal,
            idea_archive=generic.idea_archive,
        )


def test_bundle_builder_rejects_same_campaign_from_another_run(
    bootstrap_pin,
    resolver_case,
) -> None:
    current = _resolved_projection(
        "generic",
        bootstrap_pin,
        resolver_case,
    )
    projection = ReconciledRunStateProjection(
        strategy_state=current.strategy_state,
        experiment_history=replace(
            current.experiment_history,
            run_id="run_other",
        ),
        execution_journal=ExecutionRevisionProjection(
            run_id="run_other",
            campaign_id=current.experiment_history.campaign_id,
            require_contiguous_node_ids=True,
        ),
        idea_archive=current.idea_archive,
    )

    with pytest.raises(RunStateProjectionError, match="another installed run"):
        projection.build_bundle(
            bootstrap_pin=bootstrap_pin,
            run_state_layout=_layout("generic"),
            predecessor_checkpoint_head_id=_identifier(
                "run-checkpoint-head",
                "initial-head",
            ),
            predecessor_checkpoint_id=None,
            predecessor_evidence_id=None,
            target_evidence_id=_identifier(
                "run-derivative-evidence",
                "target-evidence",
            ),
            predecessor_bundle=None,
            predecessor_strategy_state=None,
        )


def test_bundle_builder_rejects_mismatched_predecessor_evidence(
    bootstrap_pin,
    resolver_case,
) -> None:
    projection = _resolved_projection(
        "generic",
        bootstrap_pin,
        resolver_case,
    )
    layout = _layout("generic")
    first = projection.build_bundle(
        bootstrap_pin=bootstrap_pin,
        run_state_layout=layout,
        predecessor_checkpoint_head_id=_identifier(
            "run-checkpoint-head",
            "initial-head",
        ),
        predecessor_checkpoint_id=None,
        predecessor_evidence_id=None,
        target_evidence_id=_identifier(
            "run-derivative-evidence",
            "first-evidence",
        ),
        predecessor_bundle=None,
        predecessor_strategy_state=None,
    )

    with pytest.raises(RunStateProjectionError, match="another authority frontier"):
        projection.build_bundle(
            bootstrap_pin=bootstrap_pin,
            run_state_layout=layout,
            predecessor_checkpoint_head_id=_identifier(
                "run-checkpoint-head",
                "current-head",
            ),
            predecessor_checkpoint_id=_identifier(
                "run-checkpoint",
                "current-checkpoint",
            ),
            predecessor_evidence_id=_identifier(
                "run-derivative-evidence",
                "wrong-evidence",
            ),
            target_evidence_id=_identifier(
                "run-derivative-evidence",
                "next-evidence",
            ),
            predecessor_bundle=first,
            predecessor_strategy_state=projection.strategy_state,
        )


def test_nonempty_history_requires_exact_journal_semantics() -> None:
    attempt = EvaluationAttempt(
        commit_sha="a" * 40,
        evaluator_id="evaluator_test",
        fidelity="fast",
        fraction=0.5,
        seed=1,
        score=0.5,
    )
    node = SearchNode(
        node_id=0,
        execution_revision=0,
        idea_id=None,
        selection_batch_id=None,
        parent_node_id=None,
        solution="tree candidate",
        branch_name="candidate-0",
        parent_branch_name="main",
        implementation_base_ref=_TREE_BASE_COMMIT,
        diff_base_ref=_TREE_BASE_COMMIT,
        feedback_base_ref=_TREE_BASE_COMMIT,
        feedback="measured",
        score=0.5,
        evaluation_valid=True,
        evaluation_attempts=(attempt,),
        started_at=_TIMESTAMP,
        build_fidelity="full",
        eval_fidelity="fast",
    )
    history = project_records(
        predecessor=build_experiment_history_genesis(
            run_id=_RUN_ID,
            campaign_id=_CAMPAIGN_ID,
            embedding_space_id=_EMBEDDING_SPACE_ID,
            embedding_provider=_EMBEDDING_PROVIDER,
            embedding_model=_EMBEDDING_MODEL,
            embedding_dimensions=_EMBEDDING_DIMENSIONS,
            embedding_canonicalizer_version=(_EMBEDDING_CANONICALIZER_VERSION),
            objective_direction="maximize",
            require_idea_links=False,
        ),
        nodes=(node,),
        embeddings_by_node_revision={(0, 0): _embedding(node.solution, 0.1)},
    )
    record = history.records[0]
    journal, _ = ExecutionRevisionProjection(
        run_id=_RUN_ID,
        campaign_id=_CAMPAIGN_ID,
        require_contiguous_node_ids=False,
    ).append_projection(
        node_id=record.node_id,
        execution_revision=record.execution_revision,
        idea_id=record.idea_id,
        selection_batch_id=record.selection_batch_id,
        parent_node_id=record.parent_node_id,
        started_at=record.timestamp,
        recorded_at="2026-07-24T00:01:00Z",
        execution_status=ExecutionStatus.COMPLETED,
        evaluation_status=EpisodeEvaluationStatus.VALID,
        evaluator_fingerprint_ids=("evaluator_test",),
        measurements={"raw_score": 0.5},
        feedback=record.feedback,
        technical_difficulties=record.technical_difficulties,
        artifact_refs=_tree_artifact_refs(
            record,
            candidate_commit=attempt.commit_sha,
        ),
        projection=record.to_dict(),
    )

    reconciled = ReconciledRunStateProjection(
        strategy_state=_strategy_state(
            "benchmark_tree_search",
            nodes=(node,),
            history_ids=(0,),
        ),
        experiment_history=history,
        execution_journal=journal,
        idea_archive=None,
    )

    assert reconciled.experiment_history.records == (record,)

    branch_only_journal, _ = ExecutionRevisionProjection(
        run_id=_RUN_ID,
        campaign_id=_CAMPAIGN_ID,
        require_contiguous_node_ids=False,
    ).append_projection(
        node_id=record.node_id,
        execution_revision=record.execution_revision,
        idea_id=record.idea_id,
        selection_batch_id=record.selection_batch_id,
        parent_node_id=record.parent_node_id,
        started_at=record.timestamp,
        recorded_at="2026-07-24T00:01:00Z",
        execution_status=ExecutionStatus.COMPLETED,
        evaluation_status=EpisodeEvaluationStatus.VALID,
        evaluator_fingerprint_ids=("evaluator_test",),
        measurements={"raw_score": 0.5},
        feedback=record.feedback,
        technical_difficulties=record.technical_difficulties,
        artifact_refs={
            "branch": record.branch_name,
            "candidate_commit": attempt.commit_sha,
            "candidate_ref": (
                f"refs/kapso/execution-revisions/{_RUN_ID}/node-0/revision-0"
            ),
            "evaluation_commit_0": attempt.commit_sha,
        },
        projection=record.to_dict(),
    )
    with pytest.raises(RunStateProjectionError, match="base commit"):
        ReconciledRunStateProjection(
            strategy_state=_strategy_state(
                "benchmark_tree_search",
                nodes=(node,),
                history_ids=(0,),
            ),
            experiment_history=history,
            execution_journal=branch_only_journal,
            idea_archive=None,
        )

    with pytest.raises(RunStateProjectionError, match="nodes differ"):
        ReconciledRunStateProjection(
            strategy_state=_strategy_state(
                "benchmark_tree_search",
                nodes=(replace(node, solution="substituted candidate"),),
                history_ids=(0,),
            ),
            experiment_history=history,
            execution_journal=journal,
            idea_archive=None,
        )

    wrong_journal, _ = ExecutionRevisionProjection(
        run_id=_RUN_ID,
        campaign_id=_CAMPAIGN_ID,
        require_contiguous_node_ids=False,
    ).append_projection(
        node_id=record.node_id,
        execution_revision=record.execution_revision,
        idea_id=record.idea_id,
        selection_batch_id=record.selection_batch_id,
        parent_node_id=record.parent_node_id,
        started_at=record.timestamp,
        recorded_at="2026-07-24T00:01:00Z",
        execution_status=ExecutionStatus.COMPLETED,
        evaluation_status=EpisodeEvaluationStatus.VALID,
        evaluator_fingerprint_ids=("evaluator_test",),
        measurements={"raw_score": 0.5},
        feedback=record.feedback,
        technical_difficulties=record.technical_difficulties,
        artifact_refs={
            **_tree_artifact_refs(
                record,
                candidate_commit=attempt.commit_sha,
            ),
            "branch": "another-branch",
        },
        projection=record.to_dict(),
    )
    with pytest.raises(RunStateProjectionError, match="semantics differ"):
        ReconciledRunStateProjection(
            strategy_state=_strategy_state(
                "benchmark_tree_search",
                nodes=(node,),
                history_ids=(0,),
            ),
            experiment_history=history,
            execution_journal=wrong_journal,
            idea_archive=None,
        )

    wrong_commit_journal, _ = ExecutionRevisionProjection(
        run_id=_RUN_ID,
        campaign_id=_CAMPAIGN_ID,
        require_contiguous_node_ids=False,
    ).append_projection(
        node_id=record.node_id,
        execution_revision=record.execution_revision,
        idea_id=record.idea_id,
        selection_batch_id=record.selection_batch_id,
        parent_node_id=record.parent_node_id,
        started_at=record.timestamp,
        recorded_at="2026-07-24T00:01:00Z",
        execution_status=ExecutionStatus.COMPLETED,
        evaluation_status=EpisodeEvaluationStatus.VALID,
        evaluator_fingerprint_ids=("evaluator_test",),
        measurements={"raw_score": 0.5},
        feedback=record.feedback,
        technical_difficulties=record.technical_difficulties,
        artifact_refs={
            **_tree_artifact_refs(
                record,
                candidate_commit="c" * 40,
            ),
            "evaluation_commit_0": "c" * 40,
        },
        projection=record.to_dict(),
    )
    with pytest.raises(RunStateProjectionError, match="semantics differ"):
        ReconciledRunStateProjection(
            strategy_state=_strategy_state(
                "benchmark_tree_search",
                nodes=(node,),
                history_ids=(0,),
            ),
            experiment_history=history,
            execution_journal=wrong_commit_journal,
            idea_archive=None,
        )

    immutable_base = "a" * 40
    lineage_node = replace(
        node,
        parent_branch_name="main",
        implementation_base_ref=immutable_base,
        diff_base_ref=immutable_base,
        feedback_base_ref=immutable_base,
    )
    wrong_lineage_journal, _ = ExecutionRevisionProjection(
        run_id=_RUN_ID,
        campaign_id=_CAMPAIGN_ID,
        require_contiguous_node_ids=False,
    ).append_projection(
        node_id=record.node_id,
        execution_revision=record.execution_revision,
        idea_id=record.idea_id,
        selection_batch_id=record.selection_batch_id,
        parent_node_id=record.parent_node_id,
        started_at=record.timestamp,
        recorded_at="2026-07-24T00:01:00Z",
        execution_status=ExecutionStatus.COMPLETED,
        evaluation_status=EpisodeEvaluationStatus.VALID,
        evaluator_fingerprint_ids=("evaluator_test",),
        measurements={"raw_score": 0.5},
        feedback=record.feedback,
        technical_difficulties=record.technical_difficulties,
        artifact_refs={
            "branch": record.branch_name,
            "parent_branch": "evil",
            "implementation_base": "evil",
            "diff_base": "evil",
            "feedback_base": "evil",
            "candidate_commit": attempt.commit_sha,
            "candidate_ref": (
                f"refs/kapso/execution-revisions/{_RUN_ID}/" "node-0/revision-0"
            ),
            "implementation_base_commit": immutable_base,
            "diff_base_commit": immutable_base,
            "feedback_base_commit": immutable_base,
            "evaluation_commit_0": attempt.commit_sha,
        },
        projection=record.to_dict(),
    )
    with pytest.raises(RunStateProjectionError, match="artifact refs differ"):
        ReconciledRunStateProjection(
            strategy_state=_strategy_state(
                "benchmark_tree_search",
                nodes=(lineage_node,),
                history_ids=(0,),
            ),
            experiment_history=history,
            execution_journal=wrong_lineage_journal,
            idea_archive=None,
        )

    record.metrics["post_validation_tamper"] = 1.0
    with pytest.raises(RunStateProjectionError, match="differs"):
        reconciled._canonical_copy()


def test_generic_projection_rejects_numeric_delta_for_unmeasured_baseline(
    tmp_path,
) -> None:
    linked_archive = _linked_generic_archive(tmp_path)
    outcome = IdeaOutcome(
        evaluation_status=EvaluationStatus.INCONCLUSIVE,
        implementation_status=ImplementationStatus.COMPLETED,
        normalized_delta=None,
        validation_tier="full",
        actual_cost=1.0,
        actual_duration=30.0,
    )
    archive = project_outcome(
        linked_archive,
        IDEA_ID,
        outcome,
        updated_at="2030-01-01T00:00:00Z",
    )
    idea = next(item for item in archive.ideas if item.idea_id == IDEA_ID)
    attempt = EvaluationAttempt(
        commit_sha="a" * 40,
        evaluator_id="evaluator_test",
        fidelity="full",
        fraction=1.0,
        seed=1,
        score=0.1,
    )
    node = SearchNode(
        node_id=0,
        idea_id=IDEA_ID,
        selection_batch_id=BATCH_ID,
        solution=idea.proposal,
        branch_name="candidate-0",
        parent_branch_name=idea.resolved_parent.branch_name,
        implementation_base_ref=idea.resolved_parent.git_ref,
        diff_base_ref=idea.resolved_parent.diff_base_ref,
        feedback_base_ref=idea.resolved_parent.feedback_base_ref,
        feedback="measured",
        score=0.1,
        evaluation_attempts=(attempt,),
        started_at=_TIMESTAMP,
        duration_seconds=30.0,
        cost_usd=1.0,
    )
    history = project_records(
        predecessor=build_experiment_history_genesis(
            run_id=_RUN_ID,
            campaign_id="campaign-alpha",
            embedding_space_id=_EMBEDDING_SPACE_ID,
            embedding_provider=_EMBEDDING_PROVIDER,
            embedding_model=_EMBEDDING_MODEL,
            embedding_dimensions=_EMBEDDING_DIMENSIONS,
            embedding_canonicalizer_version=(_EMBEDDING_CANONICALIZER_VERSION),
            objective_direction="maximize",
            require_idea_links=True,
        ),
        nodes=(node,),
        embeddings_by_node_revision={(0, 0): _embedding(node.solution, 0.1)},
    )
    record = history.records[0]
    candidate_ref = f"refs/kapso/execution-revisions/{_RUN_ID}/node-0/revision-0"
    base_commit = idea.resolved_parent.git_ref
    journal, _ = ExecutionRevisionProjection(
        run_id=_RUN_ID,
        campaign_id="campaign-alpha",
        require_contiguous_node_ids=True,
    ).append_projection(
        node_id=0,
        execution_revision=0,
        idea_id=IDEA_ID,
        selection_batch_id=BATCH_ID,
        parent_node_id=None,
        started_at=record.timestamp,
        recorded_at="2026-07-24T00:01:00Z",
        execution_status=ExecutionStatus.COMPLETED,
        evaluation_status=EpisodeEvaluationStatus.VALID,
        evaluator_fingerprint_ids=("evaluator_test",),
        measurements={"raw_score": 0.1},
        feedback=record.feedback,
        technical_difficulties="",
        artifact_refs={
            "branch": record.branch_name,
            "parent_branch": idea.resolved_parent.branch_name,
            "implementation_base": idea.resolved_parent.git_ref,
            "diff_base": idea.resolved_parent.diff_base_ref,
            "feedback_base": idea.resolved_parent.feedback_base_ref,
            "candidate_commit": attempt.commit_sha,
            "candidate_ref": candidate_ref,
            "implementation_base_commit": base_commit,
            "diff_base_commit": base_commit,
            "feedback_base_commit": base_commit,
            "evaluation_commit_0": attempt.commit_sha,
        },
        projection=record.to_dict(),
    )
    ReconciledRunStateProjection(
        strategy_state=_strategy_state(
            "generic",
            campaign_id="campaign-alpha",
            archive=archive,
            nodes=(node,),
        ),
        experiment_history=history,
        execution_journal=journal,
        idea_archive=archive,
    )
    wrong_archive = project_outcome(
        linked_archive,
        IDEA_ID,
        replace(
            outcome,
            evaluation_status=EvaluationStatus.VALID,
            normalized_delta=100.0,
        ),
        updated_at="2030-01-01T00:00:00Z",
    )

    with pytest.raises(
        RunStateProjectionError,
        match="archive outcome semantics differ",
    ):
        ReconciledRunStateProjection(
            strategy_state=_strategy_state(
                "generic",
                campaign_id="campaign-alpha",
                archive=wrong_archive,
                nodes=(node,),
            ),
            experiment_history=history,
            execution_journal=journal,
            idea_archive=wrong_archive,
        )


def test_generic_projection_rejects_parent_commit_spliced_across_events(
    tmp_path,
) -> None:
    parent_commit = "a" * 40
    unrelated_commit = "c" * 40
    child_commit = "d" * 40
    _linked_generic_archive(tmp_path)
    archive_store = IdeaArchive(
        tmp_path / "idea_archive.json",
        "campaign-alpha",
    )
    parent_outcome = IdeaOutcome(
        evaluation_status=EvaluationStatus.INCONCLUSIVE,
        implementation_status=ImplementationStatus.COMPLETED,
        normalized_delta=None,
        validation_tier="full",
        actual_cost=1.0,
        actual_duration=30.0,
    )
    archive_store.record_outcome(
        IDEA_ID,
        parent_outcome,
        expected_revision=archive_store.revision,
    )

    child_batch_id = "batch_" + "d" * 32
    child_idea_id = "idea_" + "e" * 32
    parent_plan = replace(
        generated_idea().parent_plan,
        kind=ParentPlanKind.BEST_VALID,
        experiment_node_id=0,
    )
    resolved_parent = replace(
        generated_idea().resolved_parent,
        node_id=0,
        branch_name="candidate-0",
        git_ref=unrelated_commit,
        materialized_ref=unrelated_commit,
        diff_base_ref=unrelated_commit,
        feedback_base_ref=unrelated_commit,
    )
    batch_template = planned_batch()
    directive = replace(
        batch_template.directive,
        operator_briefs=(
            replace(
                batch_template.directive.operator_briefs[0],
                parent_plan=parent_plan,
            ),
        ),
        allowed_parent_plan_kinds=(ParentPlanKind.BEST_VALID,),
    )
    first_batch = archive_store.state.batches[0]
    child_batch = replace(
        batch_template,
        batch_id=child_batch_id,
        iteration_index=1,
        planning_archive_revision=archive_store.revision,
        directive=directive,
        resolved_parents=(resolved_parent,),
        cross_run_identity=first_batch.cross_run_identity,
        prior_knowledge=first_batch.prior_knowledge,
    )
    child_idea = replace(
        generated_idea(child_idea_id),
        origin_batch_id=child_batch_id,
        parent_plan=parent_plan,
        resolved_parent=resolved_parent,
        parent_experiment_node_ids=(0,),
    )
    archive_store.create_batch(
        child_batch,
        expected_revision=archive_store.revision,
    )
    archive_store.add_ideas(
        child_batch_id,
        (child_idea,),
        generation_calls=(coding_agent_call(),),
        expected_revision=archive_store.revision,
    )
    child_analysis = analyzed_candidate(eligible_analysis())
    archive_store.record_analyses(
        child_batch_id,
        (
            replace(
                child_analysis,
                analysis=replace(
                    child_analysis.analysis,
                    idea_id=child_idea_id,
                ),
            ),
        ),
        expected_revision=archive_store.revision,
    )
    child_selection = selection()
    archive_store.record_selection(
        child_batch_id,
        replace(
            child_selection,
            selected_idea_id=child_idea_id,
            dispositions=(
                replace(
                    child_selection.dispositions[0],
                    idea_id=child_idea_id,
                ),
            ),
        ),
        selection_call=coding_agent_call(),
        expected_revision=archive_store.revision,
    )
    archive_store.link_experiment(
        child_idea_id,
        1,
        child_batch_id,
        expected_revision=archive_store.revision,
    )
    archive = archive_store.state
    parent_idea = next(idea for idea in archive.ideas if idea.idea_id == IDEA_ID)
    child_idea = next(idea for idea in archive.ideas if idea.idea_id == child_idea_id)

    parent_attempt = EvaluationAttempt(
        commit_sha=parent_commit,
        evaluator_id="evaluator_test",
        fidelity="full",
        fraction=1.0,
        seed=1,
        score=0.5,
    )
    parent_node = SearchNode(
        node_id=0,
        idea_id=IDEA_ID,
        selection_batch_id=BATCH_ID,
        solution=parent_idea.proposal,
        branch_name="candidate-0",
        parent_branch_name=parent_idea.resolved_parent.branch_name,
        implementation_base_ref=parent_idea.resolved_parent.git_ref,
        diff_base_ref=parent_idea.resolved_parent.diff_base_ref,
        feedback_base_ref=parent_idea.resolved_parent.feedback_base_ref,
        feedback="measured",
        score=0.5,
        evaluation_attempts=(parent_attempt,),
        started_at=_TIMESTAMP,
        duration_seconds=30.0,
        cost_usd=1.0,
    )
    child_node = SearchNode(
        node_id=1,
        parent_node_id=0,
        idea_id=child_idea_id,
        selection_batch_id=child_batch_id,
        solution=child_idea.proposal,
        branch_name="candidate-1",
        parent_branch_name=child_idea.resolved_parent.branch_name,
        implementation_base_ref=child_idea.resolved_parent.git_ref,
        diff_base_ref=child_idea.resolved_parent.diff_base_ref,
        feedback_base_ref=child_idea.resolved_parent.feedback_base_ref,
        feedback="implementation interrupted",
        evaluation_valid=False,
        started_at="2026-07-24T00:01:00Z",
        had_error=True,
        recoverable_error=True,
        error_message="interrupted",
        duration_seconds=5.0,
        cost_usd=0.2,
    )
    history = project_records(
        predecessor=build_experiment_history_genesis(
            run_id=_RUN_ID,
            campaign_id="campaign-alpha",
            embedding_space_id=_EMBEDDING_SPACE_ID,
            embedding_provider=_EMBEDDING_PROVIDER,
            embedding_model=_EMBEDDING_MODEL,
            embedding_dimensions=_EMBEDDING_DIMENSIONS,
            embedding_canonicalizer_version=_EMBEDDING_CANONICALIZER_VERSION,
            objective_direction="maximize",
            require_idea_links=True,
        ),
        nodes=(parent_node, child_node),
        embeddings_by_node_revision={
            (0, 0): _embedding(parent_node.solution, 0.1),
            (1, 0): _embedding(child_node.solution, 0.2),
        },
    )
    journal = ExecutionRevisionProjection(
        run_id=_RUN_ID,
        campaign_id="campaign-alpha",
        require_contiguous_node_ids=True,
    )
    for position, record in enumerate(history.records):
        is_parent = record.node_id == 0
        idea = parent_idea if is_parent else child_idea
        candidate_commit = parent_commit if is_parent else child_commit
        base_commit = (
            parent_idea.resolved_parent.git_ref if is_parent else unrelated_commit
        )
        journal, _ = journal.append_projection(
            node_id=record.node_id,
            execution_revision=record.execution_revision,
            idea_id=record.idea_id,
            selection_batch_id=record.selection_batch_id,
            parent_node_id=record.parent_node_id,
            started_at=record.timestamp,
            recorded_at=f"2026-07-24T00:0{position + 2}:00Z",
            execution_status=(
                ExecutionStatus.COMPLETED if is_parent else ExecutionStatus.INTERRUPTED
            ),
            evaluation_status=(
                EpisodeEvaluationStatus.VALID
                if is_parent
                else EpisodeEvaluationStatus.NOT_RUN
            ),
            evaluator_fingerprint_ids=(("evaluator_test",) if is_parent else ()),
            measurements={"raw_score": 0.5} if is_parent else {},
            feedback=record.feedback,
            technical_difficulties=record.technical_difficulties,
            artifact_refs={
                "branch": record.branch_name,
                "parent_branch": idea.resolved_parent.branch_name,
                "implementation_base": idea.resolved_parent.git_ref,
                "diff_base": idea.resolved_parent.diff_base_ref,
                "feedback_base": idea.resolved_parent.feedback_base_ref,
                "candidate_commit": candidate_commit,
                "candidate_ref": (
                    f"refs/kapso/execution-revisions/{_RUN_ID}/"
                    f"node-{record.node_id}/revision-0"
                ),
                "implementation_base_commit": base_commit,
                "diff_base_commit": base_commit,
                "feedback_base_commit": base_commit,
                **({"evaluation_commit_0": parent_commit} if is_parent else {}),
            },
            projection=record.to_dict(),
        )

    with pytest.raises(RunStateProjectionError, match="prior valid scored candidate"):
        ReconciledRunStateProjection(
            strategy_state=_strategy_state(
                "generic",
                campaign_id="campaign-alpha",
                archive=archive,
                nodes=(parent_node, child_node),
            ),
            experiment_history=history,
            execution_journal=journal,
            idea_archive=archive,
        )
    mismatched_child = replace(
        child_idea,
        parent_plan=replace(
            child_idea.parent_plan,
            kind=ParentPlanKind.BASELINE,
            experiment_node_id=None,
        ),
    )
    mismatched_archive = replace(
        archive,
        ideas=tuple(
            mismatched_child if idea.idea_id == child_idea_id else idea
            for idea in archive.ideas
        ),
    )
    with pytest.raises(RunStateProjectionError, match="parent plan differs"):
        ReconciledRunStateProjection(
            strategy_state=_strategy_state(
                "generic",
                campaign_id="campaign-alpha",
                archive=mismatched_archive,
                nodes=(parent_node, child_node),
            ),
            experiment_history=history,
            execution_journal=journal,
            idea_archive=mismatched_archive,
        )


def test_recovery_successor_binds_implementation_base_to_prior_candidate(
    tmp_path,
) -> None:
    linked_archive = _linked_generic_archive(tmp_path)
    idea = next(item for item in linked_archive.ideas if item.idea_id == IDEA_ID)
    failed_node = SearchNode(
        node_id=0,
        idea_id=IDEA_ID,
        selection_batch_id=BATCH_ID,
        solution=idea.proposal,
        branch_name="candidate-0",
        parent_branch_name=idea.resolved_parent.branch_name,
        implementation_base_ref=idea.resolved_parent.git_ref,
        diff_base_ref=idea.resolved_parent.diff_base_ref,
        feedback_base_ref=idea.resolved_parent.feedback_base_ref,
        feedback="implementation interrupted",
        evaluation_valid=False,
        started_at=_TIMESTAMP,
        had_error=True,
        recoverable_error=True,
        error_message="interrupted",
        duration_seconds=5.0,
        cost_usd=0.2,
    )
    history_genesis = build_experiment_history_genesis(
        run_id=_RUN_ID,
        campaign_id="campaign-alpha",
        embedding_space_id=_EMBEDDING_SPACE_ID,
        embedding_provider=_EMBEDDING_PROVIDER,
        embedding_model=_EMBEDDING_MODEL,
        embedding_dimensions=_EMBEDDING_DIMENSIONS,
        embedding_canonicalizer_version=_EMBEDDING_CANONICALIZER_VERSION,
        objective_direction="maximize",
        require_idea_links=True,
    )
    failed_history = project_records(
        predecessor=history_genesis,
        nodes=(failed_node,),
        embeddings_by_node_revision={(0, 0): _embedding(failed_node.solution, 0.1)},
    )
    failed_record = failed_history.records[0]
    first_candidate = "a" * 40
    parent_base_commit = idea.resolved_parent.git_ref
    failed_journal, _ = ExecutionRevisionProjection(
        run_id=_RUN_ID,
        campaign_id="campaign-alpha",
        require_contiguous_node_ids=True,
    ).append_projection(
        node_id=0,
        execution_revision=0,
        idea_id=IDEA_ID,
        selection_batch_id=BATCH_ID,
        parent_node_id=None,
        started_at=failed_record.timestamp,
        recorded_at="2026-07-24T00:01:00Z",
        execution_status=ExecutionStatus.INTERRUPTED,
        evaluation_status=EpisodeEvaluationStatus.NOT_RUN,
        evaluator_fingerprint_ids=(),
        measurements={},
        feedback=failed_record.feedback,
        technical_difficulties="",
        artifact_refs={
            "branch": failed_record.branch_name,
            "parent_branch": idea.resolved_parent.branch_name,
            "implementation_base": idea.resolved_parent.git_ref,
            "diff_base": idea.resolved_parent.diff_base_ref,
            "feedback_base": idea.resolved_parent.feedback_base_ref,
            "candidate_commit": first_candidate,
            "candidate_ref": (
                f"refs/kapso/execution-revisions/{_RUN_ID}/" "node-0/revision-0"
            ),
            "implementation_base_commit": parent_base_commit,
            "diff_base_commit": parent_base_commit,
            "feedback_base_commit": parent_base_commit,
        },
        projection=failed_record.to_dict(),
    )
    predecessor = ReconciledRunStateProjection(
        strategy_state=_strategy_state(
            "generic",
            campaign_id="campaign-alpha",
            archive=linked_archive,
            nodes=(failed_node,),
        ),
        experiment_history=failed_history,
        execution_journal=failed_journal,
        idea_archive=linked_archive,
    )
    recovered_attempt = EvaluationAttempt(
        commit_sha="c" * 40,
        evaluator_id="evaluator_test",
        fidelity="full",
        fraction=1.0,
        seed=1,
        score=0.2,
    )
    recovered_node = replace(
        failed_node,
        execution_revision=1,
        implementation_base_ref=first_candidate,
        feedback="recovery completed",
        score=0.2,
        evaluation_valid=True,
        evaluation_attempts=(recovered_attempt,),
        had_error=False,
        recoverable_error=False,
        error_message="",
        duration_seconds=30.0,
        cost_usd=1.0,
    )
    with pytest.raises(ValueError, match="resources moved backwards"):
        project_records(
            predecessor=failed_history,
            nodes=(
                replace(
                    recovered_node,
                    duration_seconds=1.0,
                    cost_usd=0.1,
                ),
            ),
            embeddings_by_node_revision={},
        )
    recovered_history = project_records(
        predecessor=failed_history,
        nodes=(recovered_node,),
        embeddings_by_node_revision={},
    )
    recovered_record = recovered_history.records[0]
    recovered_journal, _ = failed_journal.append_projection(
        node_id=0,
        execution_revision=1,
        idea_id=IDEA_ID,
        selection_batch_id=BATCH_ID,
        parent_node_id=None,
        started_at=recovered_record.timestamp,
        recorded_at="2026-07-24T00:02:00Z",
        execution_status=ExecutionStatus.COMPLETED,
        evaluation_status=EpisodeEvaluationStatus.VALID,
        evaluator_fingerprint_ids=("evaluator_test",),
        measurements={"raw_score": 0.2},
        feedback=recovered_record.feedback,
        technical_difficulties="",
        artifact_refs={
            "branch": recovered_record.branch_name,
            "parent_branch": idea.resolved_parent.branch_name,
            "implementation_base": first_candidate,
            "diff_base": idea.resolved_parent.diff_base_ref,
            "feedback_base": idea.resolved_parent.feedback_base_ref,
            "candidate_commit": recovered_attempt.commit_sha,
            "candidate_ref": (
                f"refs/kapso/execution-revisions/{_RUN_ID}/" "node-0/revision-1"
            ),
            "implementation_base_commit": first_candidate,
            "diff_base_commit": parent_base_commit,
            "feedback_base_commit": parent_base_commit,
            "evaluation_commit_0": recovered_attempt.commit_sha,
        },
        projection=recovered_record.to_dict(),
    )
    recovered_archive = project_outcome(
        linked_archive,
        IDEA_ID,
        IdeaOutcome(
            evaluation_status=EvaluationStatus.INCONCLUSIVE,
            implementation_status=ImplementationStatus.COMPLETED,
            normalized_delta=None,
            validation_tier="full",
            actual_cost=1.0,
            actual_duration=30.0,
        ),
        updated_at="2030-01-01T00:00:00Z",
    )
    successor = ReconciledRunStateProjection(
        strategy_state=_strategy_state(
            "generic",
            campaign_id="campaign-alpha",
            archive=recovered_archive,
            nodes=(recovered_node,),
        ),
        experiment_history=recovered_history,
        execution_journal=recovered_journal,
        idea_archive=recovered_archive,
    )

    successor.require_predecessor(predecessor)
    rolled_back_strategy = _strategy_state(
        "generic",
        campaign_id="campaign-alpha",
        archive=recovered_archive,
        nodes=(
            replace(
                recovered_node,
                duration_seconds=1.0,
                cost_usd=0.1,
            ),
        ),
    )
    with pytest.raises(
        ValueError,
        match="node history",
    ):
        rolled_back_strategy.require_predecessor(
            predecessor.strategy_state,
        )


def test_tree_projection_preserves_arbitrary_strategy_history_id() -> None:
    nodes = tuple(
        SearchNode(
            node_id=node_id,
            parent_node_id=None,
            solution=f"tree candidate {node_id}",
            branch_name=f"candidate-{node_id}",
            parent_branch_name="main",
            implementation_base_ref=_TREE_BASE_COMMIT,
            diff_base_ref=_TREE_BASE_COMMIT,
            feedback_base_ref=_TREE_BASE_COMMIT,
            feedback="measured",
            score=None,
            evaluation_valid=True,
            started_at=_TIMESTAMP,
            build_fidelity="full",
            eval_fidelity="fast",
        )
        for node_id in range(6)
    )
    history = project_records(
        predecessor=build_experiment_history_genesis(
            run_id=_RUN_ID,
            campaign_id=_CAMPAIGN_ID,
            embedding_space_id=_EMBEDDING_SPACE_ID,
            embedding_provider=_EMBEDDING_PROVIDER,
            embedding_model=_EMBEDDING_MODEL,
            embedding_dimensions=_EMBEDDING_DIMENSIONS,
            embedding_canonicalizer_version=(_EMBEDDING_CANONICALIZER_VERSION),
            objective_direction="maximize",
            require_idea_links=False,
        ),
        nodes=(nodes[5],),
        embeddings_by_node_revision={(5, 0): _embedding(nodes[5].solution, 0.5)},
    )
    record = history.records[0]
    journal, _ = ExecutionRevisionProjection(
        run_id=_RUN_ID,
        campaign_id=_CAMPAIGN_ID,
        require_contiguous_node_ids=False,
    ).append_projection(
        node_id=record.node_id,
        execution_revision=record.execution_revision,
        idea_id=None,
        selection_batch_id=None,
        parent_node_id=None,
        started_at=record.timestamp,
        recorded_at="2026-07-24T00:01:00Z",
        execution_status=ExecutionStatus.COMPLETED,
        evaluation_status=EpisodeEvaluationStatus.PARTIAL,
        evaluator_fingerprint_ids=(),
        measurements={},
        feedback=record.feedback,
        technical_difficulties=record.technical_difficulties,
        artifact_refs=_tree_artifact_refs(record),
        projection=record.to_dict(),
    )

    reconciled = ReconciledRunStateProjection(
        strategy_state=_strategy_state(
            "benchmark_tree_search",
            nodes=nodes,
            history_ids=(5,),
        ),
        experiment_history=history,
        execution_journal=journal,
        idea_archive=None,
    )

    assert tuple(
        record.node_id for record in reconciled.experiment_history.records
    ) == (5,)


def test_tree_projection_allows_unexecuted_synthetic_root_baseline() -> None:
    root = SearchNode(node_id=0, branch_name="main")
    child = SearchNode(
        node_id=1,
        parent_node_id=0,
        solution="first measured tree candidate",
        branch_name="candidate-1",
        parent_branch_name="main",
        implementation_base_ref=_TREE_BASE_COMMIT,
        diff_base_ref=_TREE_BASE_COMMIT,
        feedback_base_ref=_TREE_BASE_COMMIT,
        feedback="measured",
        started_at=_TIMESTAMP,
    )
    genesis = _projection("benchmark_tree_search")
    history = project_records(
        predecessor=genesis.experiment_history,
        nodes=(child,),
        embeddings_by_node_revision={(1, 0): _embedding(child.solution, 0.1)},
    )
    record = history.records[0]
    journal, _ = genesis.execution_journal.append_projection(
        node_id=record.node_id,
        execution_revision=record.execution_revision,
        idea_id=None,
        selection_batch_id=None,
        parent_node_id=record.parent_node_id,
        started_at=record.timestamp,
        recorded_at="2026-07-24T00:01:00Z",
        execution_status=ExecutionStatus.COMPLETED,
        evaluation_status=EpisodeEvaluationStatus.PARTIAL,
        evaluator_fingerprint_ids=(),
        measurements={},
        feedback=record.feedback,
        technical_difficulties=record.technical_difficulties,
        artifact_refs=_tree_artifact_refs(record),
        projection=record.to_dict(),
    )

    projection = ReconciledRunStateProjection(
        strategy_state=_strategy_state(
            "benchmark_tree_search",
            nodes=(root, child),
            history_ids=(1,),
        ),
        experiment_history=history,
        execution_journal=journal,
        idea_archive=None,
    )

    assert projection.experiment_history.records == (record,)


def test_tree_projection_rejects_child_spliced_to_unrelated_commit() -> None:
    parent_commit = "a" * 40
    unrelated_commit = "c" * 40
    parent_attempt = EvaluationAttempt(
        commit_sha=parent_commit,
        evaluator_id="evaluator_test",
        fidelity="full",
        fraction=1.0,
        seed=1,
        score=0.5,
    )
    parent = SearchNode(
        node_id=0,
        solution="measured parent",
        branch_name="candidate-0",
        parent_branch_name="main",
        implementation_base_ref=_TREE_BASE_COMMIT,
        diff_base_ref=_TREE_BASE_COMMIT,
        feedback_base_ref=_TREE_BASE_COMMIT,
        feedback="measured",
        score=0.5,
        evaluation_attempts=(parent_attempt,),
        started_at=_TIMESTAMP,
    )
    child = SearchNode(
        node_id=1,
        parent_node_id=0,
        solution="child candidate",
        branch_name="candidate-1",
        parent_branch_name=parent.branch_name,
        implementation_base_ref=unrelated_commit,
        diff_base_ref=unrelated_commit,
        feedback_base_ref=unrelated_commit,
        feedback="measured",
        started_at="2026-07-24T00:01:00Z",
    )
    genesis = _projection("benchmark_tree_search")
    history = project_records(
        predecessor=genesis.experiment_history,
        nodes=(parent, child),
        embeddings_by_node_revision={
            (0, 0): _embedding(parent.solution, 0.1),
            (1, 0): _embedding(child.solution, 0.2),
        },
    )
    journal = genesis.execution_journal
    for position, record in enumerate(history.records):
        is_parent = record.node_id == parent.node_id
        journal, _ = journal.append_projection(
            node_id=record.node_id,
            execution_revision=record.execution_revision,
            idea_id=None,
            selection_batch_id=None,
            parent_node_id=record.parent_node_id,
            started_at=record.timestamp,
            recorded_at=f"2026-07-24T00:0{position + 2}:00Z",
            execution_status=ExecutionStatus.COMPLETED,
            evaluation_status=(
                EpisodeEvaluationStatus.VALID
                if is_parent
                else EpisodeEvaluationStatus.PARTIAL
            ),
            evaluator_fingerprint_ids=(("evaluator_test",) if is_parent else ()),
            measurements={"raw_score": 0.5} if is_parent else {},
            feedback=record.feedback,
            technical_difficulties=record.technical_difficulties,
            artifact_refs=(
                _tree_artifact_refs(record, candidate_commit=parent_commit)
                if is_parent
                else _tree_artifact_refs(
                    record,
                    parent_branch=parent.branch_name,
                    base_commit=unrelated_commit,
                )
            ),
            projection=record.to_dict(),
        )

    with pytest.raises(RunStateProjectionError, match="prior scored ancestor"):
        ReconciledRunStateProjection(
            strategy_state=_strategy_state(
                "benchmark_tree_search",
                nodes=(parent, child),
                history_ids=(0, 1),
            ),
            experiment_history=history,
            execution_journal=journal,
            idea_archive=None,
        )


def test_tree_projection_rejects_reordered_journal_chronology() -> None:
    nodes = tuple(
        SearchNode(
            node_id=node_id,
            parent_node_id=None,
            solution=f"tree candidate {node_id}",
            branch_name=f"candidate-{node_id}",
            parent_branch_name="main",
            implementation_base_ref=_TREE_BASE_COMMIT,
            diff_base_ref=_TREE_BASE_COMMIT,
            feedback_base_ref=_TREE_BASE_COMMIT,
            feedback="measured",
            score=None,
            evaluation_valid=True,
            started_at=_TIMESTAMP,
            build_fidelity="full",
            eval_fidelity="fast",
        )
        for node_id in range(2)
    )
    history = project_records(
        predecessor=build_experiment_history_genesis(
            run_id=_RUN_ID,
            campaign_id=_CAMPAIGN_ID,
            embedding_space_id=_EMBEDDING_SPACE_ID,
            embedding_provider=_EMBEDDING_PROVIDER,
            embedding_model=_EMBEDDING_MODEL,
            embedding_dimensions=_EMBEDDING_DIMENSIONS,
            embedding_canonicalizer_version=(_EMBEDDING_CANONICALIZER_VERSION),
            objective_direction="maximize",
            require_idea_links=False,
        ),
        nodes=nodes,
        embeddings_by_node_revision={
            (0, 0): _embedding(nodes[0].solution, 0.1),
            (1, 0): _embedding(nodes[1].solution, 0.2),
        },
    )
    journal = ExecutionRevisionProjection(
        run_id=_RUN_ID,
        campaign_id=_CAMPAIGN_ID,
        require_contiguous_node_ids=False,
    )
    records = {record.node_id: record for record in history.records}
    for position, node_id in enumerate((1, 0)):
        record = records[node_id]
        journal, _ = journal.append_projection(
            node_id=record.node_id,
            execution_revision=record.execution_revision,
            idea_id=None,
            selection_batch_id=None,
            parent_node_id=None,
            started_at=record.timestamp,
            recorded_at=f"2026-07-24T00:0{position + 1}:00Z",
            execution_status=ExecutionStatus.COMPLETED,
            evaluation_status=EpisodeEvaluationStatus.PARTIAL,
            evaluator_fingerprint_ids=(),
            measurements={},
            feedback=record.feedback,
            technical_difficulties=record.technical_difficulties,
            artifact_refs=_tree_artifact_refs(record),
            projection=record.to_dict(),
        )

    with pytest.raises(RunStateProjectionError, match="execution order"):
        ReconciledRunStateProjection(
            strategy_state=_strategy_state(
                "benchmark_tree_search",
                nodes=nodes,
                history_ids=(0, 1),
            ),
            experiment_history=history,
            execution_journal=journal,
            idea_archive=None,
        )


def test_bundle_decode_rejects_manifest_revision_above_payload_revision(
    bootstrap_pin,
    resolver_case,
) -> None:
    projection = _resolved_projection(
        "generic",
        bootstrap_pin,
        resolver_case,
    )
    layout = _layout("generic")
    predecessor_evidence_id = _identifier(
        "run-derivative-evidence",
        "first-evidence",
    )
    predecessor = projection.build_bundle(
        bootstrap_pin=bootstrap_pin,
        run_state_layout=layout,
        predecessor_checkpoint_head_id=_identifier(
            "run-checkpoint-head",
            "initial-head",
        ),
        predecessor_checkpoint_id=None,
        predecessor_evidence_id=None,
        target_evidence_id=predecessor_evidence_id,
        predecessor_bundle=None,
        predecessor_strategy_state=None,
    )
    bindings = predecessor.generation.run_state_layout.bindings
    history_position = tuple(binding.authority for binding in bindings).index(
        RunStateAuthority.EXPERIMENT_HISTORY
    )
    payloads = list(predecessor.payloads)
    payloads[history_position] = replace(
        projection.experiment_history,
        objective_direction="minimize",
    ).to_json_bytes()
    transitions = []
    for position, (binding, previous, payload) in enumerate(
        zip(
            bindings,
            predecessor.generation.payload_transitions,
            payloads,
            strict=True,
        )
    ):
        transitions.append(
            RunStatePayloadTransition.mint(
                authority_binding_id=binding.authority_binding_id,
                predecessor_digest=previous.target_digest,
                predecessor_revision=previous.target_revision,
                predecessor_size_bytes=previous.target_size_bytes,
                target_digest=tree_or_blob_digest(payload),
                target_revision=1 if position == history_position else 0,
                target_size_bytes=len(payload),
            )
        )
    generation = RunDerivedStateGeneration.build(
        bootstrap_pin_id=bootstrap_pin.bootstrap_pin_id,
        run_state_layout=layout,
        predecessor_checkpoint_head_id=_identifier(
            "run-checkpoint-head",
            "current-head",
        ),
        predecessor_checkpoint_id=_identifier(
            "run-checkpoint",
            "current-checkpoint",
        ),
        predecessor_evidence_id=predecessor_evidence_id,
        target_evidence_id=_identifier(
            "run-derivative-evidence",
            "next-evidence",
        ),
        payload_transitions=tuple(transitions),
    )
    substituted = RunDerivedStateBundle(
        generation=generation,
        payloads=tuple(payloads),
    )

    with pytest.raises(RunStateProjectionError, match="revisions differ"):
        ReconciledRunStateProjection.from_bundle(
            substituted,
            strategy_state=projection.strategy_state,
            bootstrap_pin=bootstrap_pin,
        )
