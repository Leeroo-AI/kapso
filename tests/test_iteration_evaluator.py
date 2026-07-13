import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional

import git
import pytest

from kapso.execution.iteration_evaluator import (
    IterationEvaluationError,
    IterationEvaluationResult,
    IterationEvaluationValidationError,
    normalize_result,
)
from kapso.execution.memories.experiment_memory import ExperimentRecord
from kapso.execution.memories.experiment_memory.store import format_experiments
from kapso.execution.orchestrator import OrchestratorAgent, SolveResult
from kapso.execution.run_checkpoint import (
    RunCheckpointIncompatibleError,
    RunCheckpointStore,
)
from kapso.execution.search_strategies.base import SearchNode
from kapso.gated_mcp.gates.experiment_history_gate import (
    ExperimentHistoryGate,
)
from kapso.kapso import Kapso


def _init_workspace(path: Path) -> git.Repo:
    path.mkdir(parents=True)
    repo = git.Repo.init(path)
    with repo.config_writer() as config:
        config.set_value("user", "name", "Evaluator Test")
        config.set_value("user", "email", "evaluator@example.com")
    path.joinpath("README.md").write_text("# Candidate\n")
    repo.git.add(["README.md"])
    repo.git.commit("-m", "initial")
    repo.git.branch("-M", "main")
    return repo


class FakeLLM:
    def get_cumulative_cost(self) -> float:
        return 0.0


class FakeProblemHandler:
    maximize_scoring = True

    def get_problem_context(self) -> str:
        return "Improve the candidate"


class FakeKnowledgeSearch:
    def close(self) -> None:
        pass


class MaterializingWorkspace:
    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir
        self.repo = git.Repo(workspace_dir)
        self.current_cost = 0.0

    def get_cumulative_cost(self) -> float:
        return self.current_cost

    @contextmanager
    def materialize_ref(self, ref: str) -> Iterator[str]:
        worktree_dir = tempfile.mkdtemp(prefix="kapso_evaluator_test_")
        os.rmdir(worktree_dir)
        try:
            self.repo.git.worktree("add", "--detach", worktree_dir, ref)
            yield worktree_dir
        finally:
            try:
                self.repo.git.worktree("remove", "--force", worktree_dir)
            except git.GitCommandError:
                pass


class TwoCandidateStrategy:
    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir
        self.workspace = MaterializingWorkspace(workspace_dir)
        self.node_history: List[SearchNode] = []

    def run(self, context: str, budget_progress: float = 0.0) -> SearchNode:
        repo = git.Repo(self.workspace_dir)
        start = len(self.node_history)
        for offset, score in enumerate((0.1, 0.2)):
            node_id = start + offset
            branch_name = f"candidate_{node_id}"
            repo.create_head(branch_name)
            self.node_history.append(
                SearchNode(
                    node_id=node_id,
                    branch_name=branch_name,
                    parent_branch_name="main",
                    solution=f"candidate {node_id}",
                    feedback=f"feedback {node_id}",
                    score=score,
                    workspace_dir=self.workspace_dir,
                )
            )
        self.workspace.current_cost += 1.0
        return self.node_history[-1]

    def observe_budget(self, snapshot: Any) -> None:
        self.budget_snapshot = snapshot

    def observe_fidelity(self, decision: Any) -> None:
        self.fidelity_decisions = getattr(self, "fidelity_decisions", [])
        self.fidelity_decisions.append(decision)

    def get_experiment_history(
        self,
        best_last: bool = False,
    ) -> List[SearchNode]:
        return self.node_history

    def get_best_experiment(self) -> Optional[SearchNode]:
        if not self.node_history:
            return None
        return max(self.node_history, key=lambda node: node.score or 0.0)

    def get_deliverable_experiment(self) -> Optional[SearchNode]:
        return self.get_best_experiment()

    def dump_state(self) -> Dict[str, Any]:
        return {
            "node_history": [node.to_dict() for node in self.node_history]
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.node_history = [
            SearchNode.from_dict(item)
            for item in state.get("node_history", [])
        ]


def _patch_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    import kapso.execution.orchestrator as orchestrator_module

    monkeypatch.setattr(orchestrator_module, "LLMBackend", FakeLLM)
    monkeypatch.setattr(
        orchestrator_module,
        "load_mode_config",
        lambda config_path, mode: {
            "search_strategy": {"type": "generic", "params": {}}
        },
    )
    monkeypatch.setattr(
        OrchestratorAgent,
        "_create_feedback_generator",
        lambda self, coding_agent=None: object(),
    )

    def create_strategy(
        self: OrchestratorAgent,
        coding_agent: Optional[str],
        workspace_dir: Optional[str],
        start_from_checkpoint: bool,
    ) -> TwoCandidateStrategy:
        assert workspace_dir is not None
        return TwoCandidateStrategy(workspace_dir)

    monkeypatch.setattr(
        OrchestratorAgent,
        "_create_search_strategy",
        create_strategy,
    )


def _orchestrator(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    evaluator: Any,
    *,
    failure_policy: str = "record",
    resume: bool = False,
) -> OrchestratorAgent:
    _patch_orchestrator(monkeypatch)
    return OrchestratorAgent(
        FakeProblemHandler(),
        workspace_dir=str(workspace),
        iteration_evaluator=evaluator,
        iteration_evaluator_failure_policy=failure_policy,
        resume=resume,
        knowledge_search=FakeKnowledgeSearch(),
        goal="Improve support",
    )


@pytest.mark.parametrize("value", [True, "1", float("nan"), float("inf")])
def test_evaluator_rejects_non_finite_or_non_numeric_metrics(
    value: Any,
) -> None:
    with pytest.raises(
        IterationEvaluationValidationError,
        match="finite and numeric",
    ):
        normalize_result(
            IterationEvaluationResult(metrics={"accuracy": value})
        )


def test_evaluator_result_contract_is_strict() -> None:
    with pytest.raises(
        IterationEvaluationValidationError,
        match="primary_metric",
    ):
        normalize_result(
            IterationEvaluationResult(
                metrics={"accuracy": 0.9},
                primary_metric="missing",
            )
        )

    with pytest.raises(
        IterationEvaluationValidationError,
        match="JSON compatible",
    ):
        normalize_result(
            IterationEvaluationResult(
                metrics={},
                metadata={"invalid": object()},
            )
        )

    with pytest.raises(
        IterationEvaluationValidationError,
        match="keys must be strings",
    ):
        normalize_result(
            IterationEvaluationResult(
                metrics={},
                metadata={"nested": {1: "invalid"}},
            )
        )

    with pytest.raises(
        IterationEvaluationValidationError,
        match="must return",
    ):
        normalize_result({"metrics": {}})  # type: ignore[arg-type]


def test_agent_facing_history_does_not_expose_external_metrics() -> None:
    record = ExperimentRecord(
        node_id=0,
        solution="candidate",
        score=0.2,
        feedback="internal feedback",
        branch_name="candidate_0",
        had_error=False,
        error_message="",
        timestamp="now",
        metrics={"holdout_accuracy": 0.99},
        primary_metric="holdout_accuracy",
        external_evaluation_error="secret harness failure",
    )

    formatted = format_experiments([record])
    gate_formatted = ExperimentHistoryGate()._format_experiments(
        [record],
        "Experiments",
    )

    for output in (formatted, gate_formatted):
        assert "holdout_accuracy" not in output
        assert "0.99" not in output
        assert "secret harness failure" not in output


def test_every_finalized_candidate_is_evaluated_and_persisted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    repo = _init_workspace(workspace)
    contexts = []

    def evaluator(context: Any) -> IterationEvaluationResult:
        contexts.append(context)
        materialized_repo = git.Repo(context.workspace_dir)
        assert materialized_repo.head.commit == repo.commit(context.git_ref)
        assert repo.active_branch.name == "main"
        context.node.score = 999.0
        return IterationEvaluationResult(
            metrics={
                "holdout_accuracy": 0.9 - (context.node.node_id * 0.8)
            },
            primary_metric="holdout_accuracy",
            metadata={"suite": "v1"},
        )

    orchestrator = _orchestrator(workspace, monkeypatch, evaluator)
    result = orchestrator.solve(experiment_max_iter=1)

    assert [context.git_ref for context in contexts] == [
        "candidate_0",
        "candidate_1",
    ]
    assert {context.iteration for context in contexts} == {1}
    assert {context.parent_ref for context in contexts} == {"main"}
    assert all(context.workspace_dir != workspace for context in contexts)
    assert all(not context.workspace_dir.exists() for context in contexts)
    assert repo.active_branch.name == "main"

    nodes = orchestrator.search_strategy.node_history
    assert [node.score for node in nodes] == [0.1, 0.2]
    assert nodes[0].metrics == {"holdout_accuracy": 0.9}
    assert nodes[1].metrics["holdout_accuracy"] == pytest.approx(0.1)
    assert result.best_experiment is nodes[1]

    history = json.loads(
        (workspace / ".kapso" / "experiment_history.json").read_text()
    )
    assert [record["node_id"] for record in history] == [0, 1]
    assert history[0]["metrics"] == {"holdout_accuracy": 0.9}
    assert history[1]["primary_metric"] == "holdout_accuracy"

    checkpoint = RunCheckpointStore(str(workspace)).load()
    checkpoint_nodes = checkpoint.strategy_state["node_history"]
    assert checkpoint_nodes[0]["metrics"] == {"holdout_accuracy": 0.9}
    assert checkpoint_nodes[1]["external_evaluation_metadata"] == {
        "suite": "v1"
    }


def test_resume_evaluates_only_new_candidates_with_cumulative_iteration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    _init_workspace(workspace)
    calls = []

    def evaluator(context: Any) -> IterationEvaluationResult:
        calls.append((context.iteration, context.node.node_id))
        return IterationEvaluationResult(
            metrics={"validation_accuracy": context.node.node_id / 10}
        )

    first = _orchestrator(workspace, monkeypatch, evaluator)
    first.solve(experiment_max_iter=1)
    resumed = _orchestrator(
        workspace,
        monkeypatch,
        evaluator,
        resume=True,
    )
    result = resumed.solve(experiment_max_iter=1)

    assert calls == [(1, 0), (1, 1), (2, 2), (2, 3)]
    assert result.cumulative_iterations == 2
    history = json.loads(
        (workspace / ".kapso" / "experiment_history.json").read_text()
    )
    assert [record["node_id"] for record in history] == [0, 1, 2, 3]


def test_resume_rejects_a_different_evaluator_entry_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    _init_workspace(workspace)

    def first_evaluator(context: Any) -> IterationEvaluationResult:
        return IterationEvaluationResult(metrics={})

    def second_evaluator(context: Any) -> IterationEvaluationResult:
        return IterationEvaluationResult(metrics={})

    first = _orchestrator(workspace, monkeypatch, first_evaluator)
    first.solve(experiment_max_iter=1)

    with pytest.raises(
        RunCheckpointIncompatibleError,
        match="configuration",
    ):
        _orchestrator(
            workspace,
            monkeypatch,
            second_evaluator,
            resume=True,
        )


def test_record_policy_persists_evaluator_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    _init_workspace(workspace)

    def evaluator(context: Any) -> IterationEvaluationResult:
        raise RuntimeError("harness unavailable")

    orchestrator = _orchestrator(workspace, monkeypatch, evaluator)
    orchestrator.solve(experiment_max_iter=1)

    assert all(
        node.external_evaluation_error
        == "RuntimeError: harness unavailable"
        for node in orchestrator.search_strategy.node_history
    )
    history = json.loads(
        (workspace / ".kapso" / "experiment_history.json").read_text()
    )
    assert len(history) == 2
    assert history[0]["metrics"] == {}
    assert "harness unavailable" in history[0]["external_evaluation_error"]
    assert RunCheckpointStore(str(workspace)).exists()


def test_raise_policy_stops_before_history_and_checkpoint_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    _init_workspace(workspace)

    def evaluator(context: Any) -> IterationEvaluationResult:
        raise RuntimeError("mandatory harness failed")

    orchestrator = _orchestrator(
        workspace,
        monkeypatch,
        evaluator,
        failure_policy="raise",
    )
    with pytest.raises(
        IterationEvaluationError,
        match="candidate 0 at candidate_0",
    ):
        orchestrator.solve(experiment_max_iter=1)

    assert not (workspace / ".kapso" / "experiment_history.json").exists()
    # The bootstrap checkpoint legitimately exists (pre-loop durable work);
    # what must never persist is the poisoned candidate itself.
    checkpoint = RunCheckpointStore(str(workspace)).load()
    assert checkpoint.completed_iterations == 0
    assert checkpoint.strategy_state.get("node_history", []) == []
    assert {head.name for head in git.Repo(workspace).heads} >= {
        "candidate_0",
        "candidate_1",
    }


def test_public_evolve_forwards_evaluator_and_reports_selected_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kapso.kapso as kapso_module

    workspace = tmp_path / "workspace"
    _init_workspace(workspace)
    captured: Dict[str, Any] = {}
    selected = SearchNode(
        node_id=0,
        branch_name="candidate_0",
        solution="selected",
        score=0.5,
        metrics={"holdout_accuracy": 0.8},
        primary_metric="holdout_accuracy",
    )

    class PublicFakeStrategy:
        def __init__(self) -> None:
            self.workspace = SimpleNamespace(workspace_dir=str(workspace))

        def get_experiment_history(self) -> List[SearchNode]:
            return [selected]

        def checkout_to_best_experiment_branch(self) -> str:
            return "candidate_0"

    class PublicFakeOrchestrator:
        def __init__(self, handler: Any, **kwargs: Any):
            captured.update(kwargs)
            self.search_strategy = PublicFakeStrategy()

        def solve(
            self,
            experiment_max_iter: int,
            time_budget_minutes=None,
            cost_budget=None,
            finalization_reserve_minutes=None,
        ) -> SolveResult:
            return SolveResult(
                best_experiment=selected,
                final_feedback=None,
                stopped_reason="max_iterations",
                iterations_run=1,
                total_cost=0.0,
                cumulative_iterations=1,
            )

    monkeypatch.setattr(
        kapso_module,
        "OrchestratorAgent",
        PublicFakeOrchestrator,
    )

    def evaluator(context: Any) -> IterationEvaluationResult:
        return IterationEvaluationResult(metrics={})

    kapso = Kapso.__new__(Kapso)
    kapso.config_path = None
    kapso.knowledge_search = SimpleNamespace(is_enabled=lambda: False)
    solution = kapso.evolve(
        goal="Improve support",
        output_path=str(workspace),
        max_iterations=1,
        resume=True,
        iteration_evaluator=evaluator,
        iteration_evaluator_failure_policy="raise",
    )

    assert captured["iteration_evaluator"] is evaluator
    assert captured["iteration_evaluator_failure_policy"] == "raise"
    assert solution.metadata["external_metrics"] == {
        "holdout_accuracy": 0.8
    }
    assert solution.metadata["external_primary_metric"] == (
        "holdout_accuracy"
    )
