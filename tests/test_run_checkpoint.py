from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import git
import pytest

import kapso.execution.run_checkpoint as checkpoint_module
from kapso.execution.orchestrator import OrchestratorAgent, SolveResult
from kapso.execution.run_checkpoint import (
    RunCheckpoint,
    RunCheckpointCompletedError,
    RunCheckpointCorruptError,
    RunCheckpointIncompatibleError,
    RunCheckpointMissingError,
    RunCheckpointStore,
    config_fingerprint,
)
from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.search_strategies.benchmark_tree_search import (
    BenchmarkTreeSearch,
    TreeSearchNode,
)
from kapso.execution.search_strategies.generic.strategy import GenericSearch
from kapso.kapso import Kapso


def _checkpoint(**overrides: Any) -> RunCheckpoint:
    values = {
        "strategy_type": "generic",
        "goal": "Improve support",
        "config_fingerprint": config_fingerprint({"mode": "test"}),
        "status": "running",
        "completed_iterations": 1,
        "cumulative_cost": 1.5,
        "current_feedback": "Try a smaller change",
        "strategy_state": {"node_history": []},
    }
    values.update(overrides)
    return RunCheckpoint.create(**values)


def _init_git_workspace(path: Path) -> git.Repo:
    path.mkdir(parents=True)
    repo = git.Repo.init(path)
    with repo.config_writer() as config:
        config.set_value("user", "name", "Checkpoint Test")
        config.set_value("user", "email", "checkpoint@example.com")
    path.joinpath("README.md").write_text("# Test\n")
    repo.git.add(["README.md"])
    repo.git.commit("-m", "initial")
    repo.git.branch("-M", "main")
    return repo


class FakeLLM:
    def get_cumulative_cost(self) -> float:
        return 0.0


class FakeProblemHandler:
    def get_problem_context(self) -> str:
        return "Solve the support problem"


class FakeKnowledgeSearch:
    def close(self) -> None:
        pass


class FakeWorkspace:
    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir
        self.current_cost = 0.0

    def get_cumulative_cost(self) -> float:
        return self.current_cost


class FakeStrategy:
    def __init__(self, workspace_dir: str, stop_next: bool = False):
        self.workspace_dir = workspace_dir
        self.workspace = FakeWorkspace(workspace_dir)
        self.node_history: List[SearchNode] = []
        self.contexts: List[str] = []
        self.stop_next = stop_next
        self.next_agent_output = ""
        self.registered_evaluation: Dict[str, Any] = {}

    def run(self, context: str, budget_progress: float = 0.0) -> SearchNode:
        node_id = len(self.node_history)
        branch_name = f"generic_exp_{node_id}"
        repo = git.Repo(self.workspace_dir)
        if branch_name not in {head.name for head in repo.heads}:
            repo.create_head(branch_name)
        node = SearchNode(
            node_id=node_id,
            branch_name=branch_name,
            solution=f"solution-{node_id}",
            score=0.1,
            feedback=f"feedback-{node_id}",
            should_stop=self.stop_next,
            agent_output=self.next_agent_output,
        )
        self.contexts.append(context)
        self.node_history.append(node)
        self.workspace.current_cost += 1.0
        return node

    def observe_budget(self, snapshot: Any) -> None:
        self.budget_snapshot = snapshot

    def set_registered_evaluation(
        self, *, manifest, command, evaluator_id, subsample_seed
    ) -> None:
        self.registered_evaluation = {
            "manifest": dict(manifest),
            "command": command,
            "evaluator_id": evaluator_id,
            "subsample_seed": subsample_seed,
        }

    def get_experiment_history(
        self,
        best_last: bool = False,
    ) -> List[SearchNode]:
        return self.node_history

    def get_best_experiment(self) -> Optional[SearchNode]:
        return self.node_history[-1] if self.node_history else None

    def dump_state(self) -> Dict[str, Any]:
        return {
            "node_history": [node.to_dict() for node in self.node_history]
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.node_history = [
            SearchNode.from_dict(item)
            for item in state.get("node_history", [])
        ]


def _patch_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stop_next: bool = False,
) -> None:
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
    ) -> FakeStrategy:
        assert workspace_dir is not None
        return FakeStrategy(workspace_dir, stop_next=stop_next)

    monkeypatch.setattr(
        OrchestratorAgent,
        "_create_search_strategy",
        create_strategy,
    )


def _orchestrator(
    workspace: Path,
    *,
    resume: bool = False,
) -> OrchestratorAgent:
    return OrchestratorAgent(
        FakeProblemHandler(),
        workspace_dir=str(workspace),
        resume=resume,
        knowledge_search=FakeKnowledgeSearch(),
        goal="Improve support",
    )


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    store = RunCheckpointStore(str(tmp_path))
    expected = _checkpoint()

    store.save(expected)

    assert store.load() == expected


def test_missing_and_corrupt_checkpoints_fail_clearly(tmp_path: Path) -> None:
    store = RunCheckpointStore(str(tmp_path))
    with pytest.raises(RunCheckpointMissingError, match="run_state.json"):
        store.load()

    store.path.parent.mkdir(parents=True)
    store.path.write_text("{broken")
    with pytest.raises(RunCheckpointCorruptError, match="Could not read"):
        store.load()


def test_structurally_invalid_json_is_rejected_as_checkpoint_error() -> None:
    data = _checkpoint().to_dict()
    data["status"] = []
    with pytest.raises(RunCheckpointCorruptError, match="status"):
        RunCheckpoint.from_dict(data)

    data = _checkpoint().to_dict()
    data["schema_version"] = True
    with pytest.raises(RunCheckpointIncompatibleError, match="schema version"):
        RunCheckpoint.from_dict(data)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("goal", "Different goal", RunCheckpointIncompatibleError),
        (
            "strategy_type",
            "benchmark_tree_search",
            RunCheckpointIncompatibleError,
        ),
        (
            "config_fingerprint",
            config_fingerprint({"mode": "other"}),
            RunCheckpointIncompatibleError,
        ),
    ],
)
def test_resume_rejects_incompatible_campaign(
    field: str,
    value: str,
    error: type[Exception],
) -> None:
    checkpoint = _checkpoint()
    requested = {
        "goal": checkpoint.goal,
        "strategy_type": checkpoint.strategy_type,
        "config_fingerprint": checkpoint.config_fingerprint,
    }
    requested[field] = value

    with pytest.raises(error):
        checkpoint.validate_resume(**requested)


def test_completed_campaign_cannot_resume() -> None:
    checkpoint = _checkpoint(status="completed")

    with pytest.raises(RunCheckpointCompletedError, match="already completed"):
        checkpoint.validate_resume(
            goal=checkpoint.goal,
            strategy_type=checkpoint.strategy_type,
            config_fingerprint=checkpoint.config_fingerprint,
        )


def test_failed_atomic_replace_preserves_previous_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RunCheckpointStore(str(tmp_path))
    original = _checkpoint(completed_iterations=1)
    store.save(original)

    def fail_replace(source: str, destination: Path) -> None:
        raise OSError("simulated interruption")

    monkeypatch.setattr(checkpoint_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="interruption"):
        store.save(_checkpoint(completed_iterations=2))

    assert store.load() == original
    assert not list(store.path.parent.glob(".run_state.*.tmp"))


def test_generic_strategy_state_round_trip() -> None:
    source = GenericSearch.__new__(GenericSearch)
    source.node_history = [
        SearchNode(
            node_id=0,
            branch_name="generic_exp_0",
            feedback="next feedback",
            score=0.5,
        )
    ]
    source.iteration_count = 1
    source.previous_errors = ["old error"]

    restored = GenericSearch.__new__(GenericSearch)
    restored.load_state(source.dump_state())

    assert restored.node_history[0].branch_name == "generic_exp_0"
    assert restored.iteration_count == 1
    assert restored.previous_errors == ["old error"]

    invalid = source.dump_state()
    invalid["iteration_count"] = 2
    with pytest.raises(ValueError, match="must match node_history"):
        restored.load_state(invalid)


def test_search_node_rejects_invalid_runtime_types() -> None:
    with pytest.raises(ValueError, match="score must be finite"):
        SearchNode.from_dict({"node_id": 0, "score": float("nan")})
    with pytest.raises(ValueError, match="should_stop must be a boolean"):
        SearchNode.from_dict({"node_id": 0, "should_stop": "false"})


def test_tree_strategy_state_rebuilds_references() -> None:
    root = TreeSearchNode(node_id=0, solution="root")
    child = TreeSearchNode(node_id=1, parent_node=root, solution="child")
    root.children.append(child)
    child.score = 0.7

    source = BenchmarkTreeSearch.__new__(BenchmarkTreeSearch)
    source.nodes = [root, child]
    source.node_history = [child]
    source.experimentation_count = 1
    source.previous_errors = ["retry this"]

    restored = BenchmarkTreeSearch.__new__(BenchmarkTreeSearch)
    restored.load_state(source.dump_state())

    restored_root, restored_child = restored.nodes
    assert restored_child.parent_node is restored_root
    assert restored_root.children == [restored_child]
    assert restored.node_history[0] is restored_child
    assert restored.experimentation_count == 1


def test_one_iteration_then_resume_restores_feedback_and_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    repo = _init_git_workspace(workspace)
    _patch_orchestrator(monkeypatch)

    first = _orchestrator(workspace)
    first_result = first.solve(experiment_max_iter=1)
    assert first_result.cumulative_iterations == 1

    with pytest.raises(
        RunCheckpointIncompatibleError,
        match="pass resume=True",
    ):
        _orchestrator(workspace)

    resumed = _orchestrator(workspace, resume=True)
    resumed_result = resumed.solve(experiment_max_iter=1)

    assert [node.node_id for node in resumed.search_strategy.node_history] == [
        0,
        1,
    ]
    assert "feedback-0" in resumed.search_strategy.contexts[0]
    assert resumed_result.iterations_run == 1
    assert resumed_result.cumulative_iterations == 2
    assert resumed_result.total_cost == 2.0
    assert {"generic_exp_0", "generic_exp_1"} <= {
        head.name for head in repo.heads
    }

    checkpoint = RunCheckpointStore(str(workspace)).load()
    assert checkpoint.completed_iterations == 2
    assert checkpoint.current_feedback == "feedback-1"
    assert checkpoint.cumulative_cost == 2.0


def test_goal_achieved_checkpoint_is_saved_before_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    _init_git_workspace(workspace)
    _patch_orchestrator(monkeypatch, stop_next=True)

    result = _orchestrator(workspace).solve(experiment_max_iter=1)
    checkpoint = RunCheckpointStore(str(workspace)).load()

    assert result.stopped_reason == "goal_achieved"
    assert checkpoint.status == "completed"
    assert checkpoint.completed_iterations == 1

    with pytest.raises(RunCheckpointCompletedError):
        _orchestrator(workspace, resume=True)


def test_budget_exhaustion_pauses_resumably_with_last_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    _init_git_workspace(workspace)
    _patch_orchestrator(monkeypatch)

    result = _orchestrator(workspace).solve(
        experiment_max_iter=1,
        cost_budget=0.5,
    )
    checkpoint = RunCheckpointStore(str(workspace)).load()

    assert result.stopped_reason == "budget_exhausted"
    # A budget stop is a pause, not a completion: only goal achievement
    # completes a campaign.
    assert checkpoint.status == "running"
    assert checkpoint.last_stop == "cost_budget"
    assert checkpoint.completed_iterations == 1
    assert checkpoint.elapsed_seconds > 0
    assert checkpoint.cost_by_component["workspace_sessions"] == 1.0

    resumed = _orchestrator(workspace, resume=True)
    assert resumed.completed_iterations == 1


def test_durable_clock_continues_across_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    _init_git_workspace(workspace)
    _patch_orchestrator(monkeypatch)

    _orchestrator(workspace).solve(experiment_max_iter=1)
    first = RunCheckpointStore(str(workspace)).load()
    assert first.elapsed_seconds > 0
    assert first.last_stop is None
    assert set(first.cost_by_component) >= {
        "llm_backend",
        "workspace_sessions",
    }

    resumed = _orchestrator(workspace, resume=True)
    assert resumed._prior_elapsed_seconds == first.elapsed_seconds
    resumed.solve(experiment_max_iter=1)
    second = RunCheckpointStore(str(workspace)).load()

    assert second.elapsed_seconds > first.elapsed_seconds
    # 1.0 carried in from the prior slice's component record + 1.0 live.
    assert second.cost_by_component["workspace_sessions"] == 2.0
    assert second.cumulative_cost == 2.0


def test_resume_rejects_missing_candidate_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    repo = _init_git_workspace(workspace)
    _patch_orchestrator(monkeypatch)
    _orchestrator(workspace).solve(experiment_max_iter=1)
    repo.delete_head("generic_exp_0", force=True)

    with pytest.raises(
        RunCheckpointCorruptError,
        match="missing Git branches: generic_exp_0",
    ):
        _orchestrator(workspace, resume=True)


def test_v1_checkpoint_is_rejected_without_migration(tmp_path: Path) -> None:
    store = RunCheckpointStore(str(tmp_path))
    v1_data = _checkpoint().to_dict()
    v1_data["schema_version"] = 1
    for v2_field in ("elapsed_seconds", "cost_by_component", "last_stop"):
        del v1_data[v2_field]
    store.path.parent.mkdir(parents=True)
    store.path.write_text(checkpoint_module.json.dumps(v1_data))

    with pytest.raises(
        RunCheckpointIncompatibleError,
        match="not migrated",
    ):
        store.load()


def test_public_resume_workspace_validation_is_strict(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires an existing output_path"):
        Kapso._validate_resume_workspace(None)

    non_repo = tmp_path / "not-a-repo"
    non_repo.mkdir()
    with pytest.raises(ValueError, match="not a Git repository"):
        Kapso._validate_resume_workspace(str(non_repo))

    workspace = tmp_path / "workspace"
    _init_git_workspace(workspace)
    Kapso._validate_resume_workspace(str(workspace))


def test_public_evolve_forwards_resume_and_reports_cumulative_iterations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kapso.kapso as kapso_module

    workspace = tmp_path / "workspace"
    _init_git_workspace(workspace)
    captured: Dict[str, Any] = {}

    class PublicFakeStrategy:
        def __init__(self) -> None:
            self.workspace = SimpleNamespace(workspace_dir=str(workspace))

        def get_experiment_history(self) -> List[SearchNode]:
            return []

        def checkout_to_best_experiment_branch(self) -> None:
            return None

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
            assert experiment_max_iter == 1
            return SolveResult(
                best_experiment=None,
                final_feedback=None,
                stopped_reason="max_iterations",
                iterations_run=1,
                total_cost=2.0,
                cumulative_iterations=4,
            )

    monkeypatch.setattr(
        kapso_module,
        "OrchestratorAgent",
        PublicFakeOrchestrator,
    )

    kapso = Kapso.__new__(Kapso)
    kapso.config_path = None
    kapso.knowledge_search = SimpleNamespace(is_enabled=lambda: False)
    result = kapso.evolve(
        goal="Improve support",
        output_path=str(workspace),
        max_iterations=1,
        resume=True,
    )

    assert captured["resume"] is True
    assert captured["initial_repo"] is None
    assert result.metadata["iterations"] == 1
    assert result.metadata["cumulative_iterations"] == 4
    assert result.metadata["resumed"] is True
