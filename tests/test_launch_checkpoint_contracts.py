from copy import deepcopy
from dataclasses import replace

import pytest

from kapso.cross_run.canonical import CanonicalizationError, content_id
from kapso.cross_run.contracts import ContractValidationError, TaskContextBinding
from kapso.cross_run.launch.checkpoint_contracts import (
    RunCheckpoint,
    RunCheckpointContractError,
    RunCheckpointHead,
    RunCheckpointStatus,
    RunCheckpointStop,
    RunFeedbackSource,
    RunStrategyKind,
    RunStrategyState,
    RunTerminationDecision,
)
from kapso.cross_run.launch.resume_contracts import (
    RunReleaseUseMode,
    RunSafetyBoundary,
    RunSafetyState,
)
from kapso.execution.evaluation_integrity import AGENT_GENERATED
from kapso.execution.fidelity import EvaluationAttempt
from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.search_strategies.generic.ideation.archive import (
    IDEA_ARCHIVE_SCHEMA,
    IdeaArchiveState,
    archive_is_compatible_descendant,
)
from kapso.execution.search_strategies.generic.ideation.types import (
    IdeationCrossRunIdentity,
)
from test_ideation_domain import planned_batch
from test_launch_resolver import resolver_case
from test_launch_resume_contracts import (
    _bootstrap_pin,
    _empty_frontier,
    _security_observation,
    _subjects,
)
from test_prior_knowledge_gate import citable_access_materialization
from test_run_checkpoint import _strict_generic_strategy

CHECKPOINT_TIME = "2026-07-23T18:00:00Z"


def _evaluation_integrity():
    return {
        "provenance": AGENT_GENERATED,
        "manifest": {},
        "fingerprint": None,
    }


def test_run_checkpoint_head_retains_and_advances_exact_frontier(
    resolver_case,
    tmp_path,
):
    pin = _bootstrap_pin(resolver_case, tmp_path)
    initial_checkpoint = _initial_checkpoint(pin)
    initial_head = RunCheckpointHead.initial(pin)
    genesis_head = initial_head.advance(initial_checkpoint)

    assert initial_head.checkpoint is None
    assert genesis_head.checkpoint == initial_checkpoint
    genesis_head.require_checkpoint(initial_checkpoint)
    assert (
        RunCheckpointHead.from_json_bytes(genesis_head.to_json_bytes()) == genesis_head
    )
    with pytest.raises(RunCheckpointContractError, match="differs"):
        initial_head.require_checkpoint(initial_checkpoint)


def _generic_strategy_state(pin):
    return RunStrategyState.build(
        strategy_kind=RunStrategyKind.GENERIC,
        campaign_id=pin.installation_receipt.campaign_id,
        state={
            "idea_archive_snapshot": {
                "schema": IDEA_ARCHIVE_SCHEMA,
                "campaign_id": pin.installation_receipt.campaign_id,
                "revision": 0,
                "created_at": CHECKPOINT_TIME,
                "updated_at": CHECKPOINT_TIME,
                "batches": [],
                "ideas": [],
                "claims": [],
                "gaps": [],
            },
            "node_history": [],
            "iteration_count": 0,
            "previous_errors": [],
            "evaluation_integrity": _evaluation_integrity(),
            "scores_evaluator_id": "",
            "evaluator_transition": None,
        },
    )


def _contract_generic_state(tmp_path):
    tmp_path.mkdir()
    dumped = _strict_generic_strategy(tmp_path).dump_state()
    materialization = citable_access_materialization()
    task_context = TaskContextBinding.from_dict(
        materialization.prior_knowledge_snapshot.selected_records[0]["payload"][
            "task_context_binding"
        ]
    )
    identity = IdeationCrossRunIdentity(
        launch_manifest_id=content_id("launch-manifest", {"checkpoint": "test"}),
        scope_contract_id=task_context.scope_contract_id,
        knowledge_snapshot_id=(
            materialization.prior_knowledge_snapshot.source_snapshot_id
        ),
        expert_base_release_id=content_id(
            "expert-base-release",
            {"checkpoint": "test"},
        ),
        embedding_space_id=content_id(
            "embedding-space",
            {"checkpoint": "test"},
        ),
        task_context_binding=task_context,
    )
    for batch in dumped["idea_archive_snapshot"]["batches"]:
        batch["cross_run_identity"] = identity.to_dict()
        batch["prior_knowledge"] = materialization.to_dict()
    return {
        "idea_archive_snapshot": dumped["idea_archive_snapshot"],
        "node_history": dumped["node_history"],
        "iteration_count": dumped["iteration_count"],
        "previous_errors": dumped["previous_errors"],
        "evaluation_integrity": dumped["evaluation_integrity"],
        "scores_evaluator_id": dumped["scores_evaluator_id"],
        "evaluator_transition": dumped["evaluator_transition"],
    }


def _initial_safety(pin):
    frontier = _empty_frontier(pin)
    release_use = pin.launch_manifest.release_use_observation
    return RunSafetyState.build(
        predecessor=None,
        bootstrap_pin=pin,
        boundary=RunSafetyBoundary.INITIALIZATION,
        derivative_frontier=frontier,
        security_observation=_security_observation(
            pin,
            _subjects(pin, release_use, frontier),
        ),
        release_use_observation=release_use,
        release_use_mode=RunReleaseUseMode.ONLINE_CURRENT,
    )


def _successor_safety(pin, predecessor):
    frontier = predecessor.derivative_frontier
    release_use = predecessor.release_use_observation
    return RunSafetyState.build(
        predecessor=predecessor,
        bootstrap_pin=pin,
        boundary=RunSafetyBoundary.RESUME,
        derivative_frontier=frontier,
        security_observation=_security_observation(
            pin,
            _subjects(pin, release_use, frontier, predecessor),
            generation_offset=2,
        ),
        release_use_observation=release_use,
        release_use_mode=RunReleaseUseMode.ONLINE_CURRENT,
    )


def _initial_checkpoint(pin):
    return RunCheckpoint.build(
        predecessor=None,
        status=RunCheckpointStatus.ACTIVE,
        last_stop=None,
        completed_iterations=0,
        cumulative_cost=0,
        elapsed_seconds=0,
        cost_by_component={},
        feedback_source=None,
        current_feedback=None,
        termination_decision=None,
        strategy_state=_generic_strategy_state(pin),
        safety_state=_initial_safety(pin),
    )


def test_checkpoint_is_content_addressed_and_requires_exact_successor(
    resolver_case,
    tmp_path,
):
    pin = _bootstrap_pin(resolver_case, tmp_path)
    initial = _initial_checkpoint(pin)
    successor = RunCheckpoint.build(
        predecessor=initial,
        status=RunCheckpointStatus.ACTIVE,
        last_stop=RunCheckpointStop.COST_BUDGET,
        completed_iterations=0,
        cumulative_cost=1.25,
        elapsed_seconds=4.5,
        cost_by_component={"implementation": 1.25},
        feedback_source=None,
        current_feedback=None,
        termination_decision=None,
        strategy_state=initial.strategy_state,
        safety_state=_successor_safety(pin, initial.safety_state),
    )

    assert RunCheckpoint.from_json_bytes(initial.to_json_bytes()) == initial
    assert successor.predecessor_checkpoint_id == initial.run_checkpoint_id
    assert successor.checkpoint_sequence == 1
    successor.require_bootstrap_pin(pin)
    with pytest.raises(RunCheckpointContractError, match="predecessor"):
        successor.require_predecessor(successor)
    with pytest.raises(CanonicalizationError, match="run_checkpoint_id mismatch"):
        replace(successor, last_stop=None)

    old_shape = {
        "schema_version": 2,
        "strategy_type": "generic",
        "goal": "legacy",
    }
    with pytest.raises(ContractValidationError, match="fields mismatch"):
        RunCheckpoint.from_dict(old_shape)


def test_checkpoint_rejects_identity_budget_and_terminal_rollbacks(
    resolver_case,
    tmp_path,
):
    pin = _bootstrap_pin(resolver_case, tmp_path)
    initial = _initial_checkpoint(pin)
    successor_safety = _successor_safety(pin, initial.safety_state)
    foreign_state = _generic_strategy_state(pin).parsed_state()
    foreign_state["idea_archive_snapshot"]["campaign_id"] = "foreign-campaign"
    with pytest.raises(RunCheckpointContractError, match="another launch"):
        RunCheckpoint.build(
            predecessor=initial,
            status=RunCheckpointStatus.ACTIVE,
            last_stop=None,
            completed_iterations=0,
            cumulative_cost=0,
            elapsed_seconds=0,
            cost_by_component={},
            feedback_source=None,
            current_feedback=None,
            termination_decision=None,
            strategy_state=RunStrategyState.build(
                strategy_kind=RunStrategyKind.GENERIC,
                campaign_id="foreign-campaign",
                state=foreign_state,
            ),
            safety_state=successor_safety,
        )

    advanced = RunCheckpoint.build(
        predecessor=initial,
        status=RunCheckpointStatus.ACTIVE,
        last_stop=None,
        completed_iterations=0,
        cumulative_cost=2,
        elapsed_seconds=3,
        cost_by_component={"implementation": 2},
        feedback_source=None,
        current_feedback=None,
        termination_decision=None,
        strategy_state=initial.strategy_state,
        safety_state=successor_safety,
    )
    with pytest.raises(RunCheckpointContractError, match="rolled back"):
        RunCheckpoint.build(
            predecessor=advanced,
            status=RunCheckpointStatus.ACTIVE,
            last_stop=None,
            completed_iterations=0,
            cumulative_cost=1,
            elapsed_seconds=3,
            cost_by_component={"implementation": 2},
            feedback_source=None,
            current_feedback=None,
            termination_decision=None,
            strategy_state=advanced.strategy_state,
            safety_state=_successor_safety(pin, advanced.safety_state),
        )

    stale_feedback = object.__new__(RunCheckpoint)
    for field_name, value in vars(advanced).items():
        object.__setattr__(stale_feedback, field_name, value)
    object.__setattr__(
        stale_feedback,
        "feedback_source",
        RunFeedbackSource(
            node_id=0,
            execution_revision=0,
        ),
    )
    object.__setattr__(stale_feedback, "current_feedback", "stale guidance")
    with pytest.raises(RunCheckpointContractError, match="rolled back"):
        stale_feedback.require_predecessor(initial)

    with pytest.raises(RunCheckpointContractError, match="terminal decision"):
        RunCheckpoint.build(
            predecessor=advanced,
            status=RunCheckpointStatus.COMPLETED,
            last_stop=None,
            completed_iterations=0,
            cumulative_cost=2,
            elapsed_seconds=3,
            cost_by_component={"implementation": 2},
            feedback_source=None,
            current_feedback=None,
            termination_decision=None,
            strategy_state=advanced.strategy_state,
            safety_state=_successor_safety(pin, advanced.safety_state),
        )
    with pytest.raises(RunCheckpointContractError, match="policy reasons"):
        RunTerminationDecision(
            delivery_source=RunFeedbackSource(
                node_id=0,
                execution_revision=0,
            ),
            reasons=(),
        )

    archive_ahead = initial.strategy_state.parsed_state()
    archive_ahead["idea_archive_snapshot"]["revision"] = 1
    impossible_initial_strategy = RunStrategyState.build(
        strategy_kind=RunStrategyKind.GENERIC,
        campaign_id=pin.installation_receipt.campaign_id,
        state=archive_ahead,
    )
    with pytest.raises(
        RunCheckpointContractError,
        match="archive and safety|empty durable frontier",
    ):
        RunCheckpoint.build(
            predecessor=None,
            status=RunCheckpointStatus.ACTIVE,
            last_stop=None,
            completed_iterations=0,
            cumulative_cost=0,
            elapsed_seconds=0,
            cost_by_component={},
            feedback_source=None,
            current_feedback=None,
            termination_decision=None,
            strategy_state=impossible_initial_strategy,
            safety_state=initial.safety_state,
        )


def _tree_node(node_id, parent_id, children_ids):
    node = SearchNode(node_id=node_id, parent_node_id=parent_id).to_dict()
    node.update(
        {
            "parent_id": parent_id,
            "children_ids": children_ids,
            "is_terminated": False,
            "is_root": parent_id is None,
            "node_event_history": [],
            "ideation_repo_memory_sections_consulted": [],
        }
    )
    return node


def test_strategy_state_rejects_tolerant_old_shapes_and_tree_cycles():
    generic = {
        "idea_archive_snapshot": {
            "schema": IDEA_ARCHIVE_SCHEMA,
            "campaign_id": "campaign",
            "revision": 0,
            "created_at": CHECKPOINT_TIME,
            "updated_at": CHECKPOINT_TIME,
            "batches": [],
            "ideas": [],
            "claims": [],
            "gaps": [],
        },
        "node_history": [],
        "iteration_count": 0,
        "previous_errors": [],
        "evaluation_integrity": _evaluation_integrity(),
        "scores_evaluator_id": "",
        "evaluator_transition": None,
        "schema": "kapso.generic_search_state.v5",
    }
    with pytest.raises(RunCheckpointContractError, match="fields"):
        RunStrategyState.build(
            strategy_kind=RunStrategyKind.GENERIC,
            campaign_id="campaign",
            state=generic,
        )
    del generic["schema"]
    generic["scores_evaluator_id"] = "evaluator-a"
    original_evaluator = RunStrategyState.build(
        strategy_kind=RunStrategyKind.GENERIC,
        campaign_id="campaign",
        state=generic,
    )
    generic["scores_evaluator_id"] = "evaluator-b"
    rewritten_evaluator = RunStrategyState.build(
        strategy_kind=RunStrategyKind.GENERIC,
        campaign_id="campaign",
        state=generic,
    )
    with pytest.raises(RunCheckpointContractError, match="evaluator authority"):
        rewritten_evaluator.require_predecessor(original_evaluator)

    cyclic_tree = {
        "nodes": [
            _tree_node(0, 1, [1]),
            _tree_node(1, 0, [0]),
        ],
        "node_history_ids": [],
        "experimentation_count": 0,
        "previous_errors": [],
        "evaluation_integrity": _evaluation_integrity(),
    }
    with pytest.raises(RunCheckpointContractError, match="cycle"):
        RunStrategyState.build(
            strategy_kind=RunStrategyKind.BENCHMARK_TREE_SEARCH,
            campaign_id="campaign",
            state=cyclic_tree,
        )

    negative_event_tree = {
        "nodes": [_tree_node(0, None, [])],
        "node_history_ids": [],
        "experimentation_count": 0,
        "previous_errors": [],
        "evaluation_integrity": _evaluation_integrity(),
    }
    negative_event_tree["nodes"][0]["node_event_history"] = [[-1, "create"]]
    with pytest.raises(RunCheckpointContractError, match="event history"):
        RunStrategyState.build(
            strategy_kind=RunStrategyKind.BENCHMARK_TREE_SEARCH,
            campaign_id="campaign",
            state=negative_event_tree,
        )

    noncontiguous_tree = {
        "nodes": [_tree_node(7, None, [])],
        "node_history_ids": [],
        "experimentation_count": 0,
        "previous_errors": [],
        "evaluation_integrity": _evaluation_integrity(),
    }
    with pytest.raises(
        RunCheckpointContractError,
        match="ordered and contiguous",
    ):
        RunStrategyState.build(
            strategy_kind=RunStrategyKind.BENCHMARK_TREE_SEARCH,
            campaign_id="campaign",
            state=noncontiguous_tree,
        )


def test_strategy_predecessor_preserves_archive_and_tree_history():
    batch = planned_batch()
    original_archive = IdeaArchiveState(
        schema=IDEA_ARCHIVE_SCHEMA,
        campaign_id=batch.campaign_id,
        revision=1,
        created_at=batch.created_at,
        updated_at=batch.updated_at,
        batches=(batch,),
        ideas=(),
        claims=(),
        gaps=(),
    )
    rewritten_archive = replace(
        original_archive,
        revision=2,
        batches=(
            replace(
                batch,
                problem_statement="rewritten historical problem",
            ),
        ),
    )
    assert not archive_is_compatible_descendant(
        original_archive,
        rewritten_archive,
    )

    original_tree = {
        "nodes": [_tree_node(0, None, [])],
        "node_history_ids": [0],
        "experimentation_count": 0,
        "previous_errors": ["first failure"],
        "evaluation_integrity": _evaluation_integrity(),
    }
    original_tree["nodes"][0]["is_terminated"] = True
    original_tree["nodes"][0]["node_event_history"] = [[0, "create"]]
    original = RunStrategyState.build(
        strategy_kind=RunStrategyKind.BENCHMARK_TREE_SEARCH,
        campaign_id="campaign",
        state=original_tree,
    )
    rewritten_tree = {
        "nodes": [_tree_node(0, None, [])],
        "node_history_ids": [],
        "experimentation_count": 0,
        "previous_errors": [],
        "evaluation_integrity": _evaluation_integrity(),
    }
    rewritten = RunStrategyState.build(
        strategy_kind=RunStrategyKind.BENCHMARK_TREE_SEARCH,
        campaign_id="campaign",
        state=rewritten_tree,
    )
    with pytest.raises(RunCheckpointContractError, match="prompt|tree strategy"):
        rewritten.require_predecessor(original)


def test_generic_strategy_revision_requires_one_bounded_transition(tmp_path):
    original_state = _contract_generic_state(tmp_path / "strategy")
    original = RunStrategyState.build(
        strategy_kind=RunStrategyKind.GENERIC,
        campaign_id="campaign_" + "f" * 32,
        state=original_state,
    )

    injected_state = deepcopy(original_state)
    injected_state["node_history"][0]["execution_revision"] = 37
    injected_state["node_history"][0]["score"] = 999.0
    injected = RunStrategyState.build(
        strategy_kind=RunStrategyKind.GENERIC,
        campaign_id="campaign_" + "f" * 32,
        state=injected_state,
    )
    with pytest.raises(RunCheckpointContractError, match="node history"):
        injected.require_predecessor(original)

    unmeasured_state = deepcopy(original_state)
    unmeasured_state["node_history"][0]["execution_revision"] = 1
    unmeasured_state["node_history"][0]["score"] = 999.0
    unmeasured = RunStrategyState.build(
        strategy_kind=RunStrategyKind.GENERIC,
        campaign_id="campaign_" + "f" * 32,
        state=unmeasured_state,
    )
    with pytest.raises(RunCheckpointContractError, match="node history"):
        unmeasured.require_predecessor(original)

    measured_state = deepcopy(original_state)
    measured_node = measured_state["node_history"][0]
    measured_node["execution_revision"] = 1
    measured_node["score"] = 0.7
    measured_node["evaluation_attempts"] = [
        EvaluationAttempt(
            commit_sha="measured-commit",
            evaluator_id="canonical-v1",
            fidelity="full",
            fraction=1.0,
            seed=0,
            score=0.7,
        ).to_dict()
    ]
    measured = RunStrategyState.build(
        strategy_kind=RunStrategyKind.GENERIC,
        campaign_id="campaign_" + "f" * 32,
        state=measured_state,
    )
    measured.require_predecessor(original)

    invalidated_state = deepcopy(original_state)
    invalidated_node = invalidated_state["node_history"][0]
    invalidated_node["execution_revision"] = 1
    invalidated_node["score"] = None
    invalidated = RunStrategyState.build(
        strategy_kind=RunStrategyKind.GENERIC,
        campaign_id="campaign_" + "f" * 32,
        state=invalidated_state,
    )
    invalidated.require_predecessor(original)


def test_tree_candidate_can_be_executed_without_changing_its_revision():
    pending_tree = {
        "nodes": [
            _tree_node(0, None, [1]),
            _tree_node(1, 0, []),
        ],
        "node_history_ids": [],
        "experimentation_count": 0,
        "previous_errors": [],
        "evaluation_integrity": _evaluation_integrity(),
    }
    pending_tree["nodes"][0]["branch_name"] = "main"
    pending_tree["nodes"][0]["solution"] = "root"
    pending_tree["nodes"][1]["solution"] = "candidate"
    pending_tree["nodes"][1]["node_event_history"] = [[0, "create"]]
    pending = RunStrategyState.build(
        strategy_kind=RunStrategyKind.BENCHMARK_TREE_SEARCH,
        campaign_id="campaign",
        state=pending_tree,
    )

    executed_tree = deepcopy(pending_tree)
    executed_child = executed_tree["nodes"][1]
    executed_child["branch_name"] = "experiment-1"
    executed_child["parent_branch_name"] = "main"
    executed_child["workspace_dir"] = "/workspace"
    executed_child["agent_output"] = "implemented"
    executed_child["code_diff"] = "diff"
    executed_child["evaluation_output"] = "evaluated"
    executed_child["feedback"] = "promising"
    executed_child["score"] = 0.7
    executed_child["node_event_history"].append([1, "experiment"])
    executed_tree["node_history_ids"] = [1]
    executed_tree["experimentation_count"] = 1
    executed = RunStrategyState.build(
        strategy_kind=RunStrategyKind.BENCHMARK_TREE_SEARCH,
        campaign_id="campaign",
        state=executed_tree,
    )
    executed.require_predecessor(pending)

    rewritten_tree = deepcopy(executed_tree)
    rewritten_tree["nodes"][1]["feedback"] = "rewritten"
    rewritten = RunStrategyState.build(
        strategy_kind=RunStrategyKind.BENCHMARK_TREE_SEARCH,
        campaign_id="campaign",
        state=rewritten_tree,
    )
    with pytest.raises(RunCheckpointContractError, match="node history"):
        rewritten.require_predecessor(executed)

    revision_bump_tree = deepcopy(executed_tree)
    revision_bump_tree["nodes"][1]["execution_revision"] = 1
    revision_bump_tree["nodes"][1]["feedback"] = "rewritten through revision"
    revision_bump_tree["nodes"][1]["score"] = 999
    with pytest.raises(RunCheckpointContractError, match="must remain zero"):
        RunStrategyState.build(
            strategy_kind=RunStrategyKind.BENCHMARK_TREE_SEARCH,
            campaign_id="campaign",
            state=revision_bump_tree,
        )
