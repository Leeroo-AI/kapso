import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import git
import pytest

from kapso.environment.handlers.generic import GenericProblemHandler
from kapso.execution.coding_agents.base import CodingAgentConfig
from kapso.execution.evaluation_integrity import (
    AGENT_GENERATED,
    PROVIDED,
    EvaluationIntegrityError,
    build_data_manifest,
    build_evaluation_manifest,
    manifest_fingerprint,
    verify_evaluation_tree,
)
from kapso.execution.memories.experiment_memory import ExperimentHistoryStore
from kapso.execution.search_strategies.base import (
    SearchNode,
    SearchStrategy,
    SearchStrategyConfig,
)
from kapso.execution.search_strategies.benchmark_tree_search import (
    BenchmarkTreeSearch,
)
from kapso.execution.search_strategies.generic.strategy import GenericSearch


def _write_suite(path: Path) -> None:
    path.mkdir(parents=True)
    path.joinpath("evaluate.py").write_text("print('score=1')\n")
    path.joinpath("config.yaml").write_text("threshold: 0.8\n")


def _init_candidate_repo(path: Path) -> git.Repo:
    _write_suite(path / "kapso_evaluation")
    repo = git.Repo.init(path)
    with repo.config_writer() as config:
        config.set_value("user", "name", "Integrity Test")
        config.set_value("user", "email", "integrity@example.com")
    repo.git.add(["kapso_evaluation"])
    repo.git.commit("-m", "provided evaluation baseline")
    repo.git.branch("-M", "main")
    return repo


def _candidate_branch(repo: git.Repo, name: str, mutate) -> None:
    repo.git.checkout("-b", name)
    mutate(Path(repo.working_dir) / "kapso_evaluation")
    repo.git.add("-A")
    repo.git.commit("-m", name)
    repo.git.checkout("main")


class MaterializingWorkspace:
    def __init__(self, path: Path):
        self.repo = git.Repo(path)

    @contextmanager
    def materialize_ref(self, ref: str) -> Iterator[str]:
        worktree = tempfile.mkdtemp(prefix="kapso_integrity_test_")
        os.rmdir(worktree)
        try:
            self.repo.git.worktree("add", "--detach", worktree, ref)
            yield worktree
        finally:
            self.repo.git.worktree("remove", "--force", worktree)


class StubStrategy(SearchStrategy):
    def run(self, context, budget_progress: float = 0.0):
        return None

    def get_experiment_history(self, best_last: bool = False):
        return []

    def get_best_experiment(self):
        return None

    def checkout_to_best_experiment_branch(self):
        return None


def _provided_strategy(workspace: Path) -> GenericSearch:
    strategy = GenericSearch.__new__(GenericSearch)
    strategy.workspace = MaterializingWorkspace(workspace)
    strategy.registered_evaluation_manifest = {}
    strategy.registered_data_manifest = {}
    strategy.evaluation_provenance = PROVIDED
    strategy.provided_evaluation_manifest = build_evaluation_manifest(
        workspace / "kapso_evaluation"
    )
    strategy.provided_evaluation_fingerprint = manifest_fingerprint(
        strategy.provided_evaluation_manifest
    )
    return strategy


def test_manifest_is_deterministic_and_rejects_unsafe_sources(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    _write_suite(suite)

    first = build_evaluation_manifest(suite)
    second = build_evaluation_manifest(suite)

    assert first == second
    assert manifest_fingerprint(first) == manifest_fingerprint(second)

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(EvaluationIntegrityError, match="contains no files"):
        build_evaluation_manifest(empty)

    symlinked = tmp_path / "symlinked"
    symlinked.mkdir()
    symlinked.joinpath("evaluate.py").symlink_to(suite / "evaluate.py")
    with pytest.raises(EvaluationIntegrityError, match="symlinks"):
        build_evaluation_manifest(symlinked)

    suite_link = tmp_path / "suite-link"
    suite_link.symlink_to(suite, target_is_directory=True)
    with pytest.raises(EvaluationIntegrityError, match="cannot be a symlink"):
        build_evaluation_manifest(suite_link)


def test_strategy_setup_tracks_the_exact_provided_suite_before_mutation(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    _write_suite(suite)
    workspace = tmp_path / "workspace"
    config = SearchStrategyConfig(
        problem_handler=SimpleNamespace(),
        llm=SimpleNamespace(),
        coding_agent_config=CodingAgentConfig(
            agent_type="aider",
            model="test",
            debug_model="test",
        ),
        eval_dir=str(suite),
    )

    strategy = StubStrategy(config, workspace_dir=str(workspace))

    assert strategy.evaluation_provenance == PROVIDED
    assert strategy.provided_evaluation_manifest == (
        build_evaluation_manifest(suite)
    )
    repo = git.Repo(workspace)
    tracked = set(repo.git.ls_files("kapso_evaluation").splitlines())
    assert tracked == {
        "kapso_evaluation/config.yaml",
        "kapso_evaluation/evaluate.py",
    }

    missing_workspace = tmp_path / "must-not-be-created"
    config.eval_dir = str(tmp_path / "missing-suite")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        StubStrategy(config, workspace_dir=str(missing_workspace))
    assert not missing_workspace.exists()


def test_runtime_outputs_are_allowed_but_evaluator_changes_are_rejected(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    _write_suite(baseline)
    expected = build_evaluation_manifest(baseline)

    baseline.joinpath("result.json").write_text('{"score": 1}\n')
    assert verify_evaluation_tree(baseline, expected).valid is True

    baseline.joinpath("evaluate.py").write_text("print('fake score')\n")
    changed = verify_evaluation_tree(baseline, expected)
    assert changed.valid is False
    assert "changed: evaluate.py" in changed.error

    baseline.joinpath("evaluate.py").unlink()
    missing = verify_evaluation_tree(baseline, expected)
    assert missing.valid is False
    assert "missing: evaluate.py" in missing.error

    baseline.joinpath("alternate.py").write_text("print('fake')\n")
    executable = baseline.joinpath("alternate")
    executable.write_text("#!/bin/sh\necho fake\n")
    executable.chmod(0o755)
    injected = verify_evaluation_tree(baseline, expected)
    assert injected.valid is False
    assert "unexpected evaluator source:" in injected.error
    assert "alternate.py" in injected.error
    assert "alternate" in injected.error


def test_candidate_tampering_clears_score_and_stop_state(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    repo = _init_candidate_repo(workspace)
    _candidate_branch(
        repo,
        "candidate",
        lambda suite: suite.joinpath("evaluate.py").write_text(
            "print('always passes')\n"
        ),
    )
    strategy = _provided_strategy(workspace)
    node = SearchNode(
        node_id=0,
        branch_name="candidate",
        score=1.0,
        should_stop=True,
    )

    valid = strategy.enforce_evaluation_integrity(node)

    assert valid is False
    assert node.evaluation_provenance == PROVIDED
    assert node.evaluation_valid is False
    assert node.score is None
    assert node.should_stop is False
    assert "changed: evaluate.py" in node.evaluation_integrity_error
    assert node.feedback == node.evaluation_integrity_error
    assert repo.active_branch.name == "main"


def test_unchanged_candidate_and_agent_generated_evaluation_are_valid(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    repo = _init_candidate_repo(workspace)
    repo.create_head("candidate")
    strategy = _provided_strategy(workspace)
    node = SearchNode(node_id=0, branch_name="candidate", score=0.7)

    assert strategy.enforce_evaluation_integrity(node) is True
    assert node.evaluation_valid is True
    assert node.evaluation_provenance == PROVIDED

    generated = GenericSearch.__new__(GenericSearch)
    generated.registered_evaluation_manifest = {}
    generated.registered_data_manifest = {}
    generated.evaluation_provenance = AGENT_GENERATED
    generated_node = SearchNode(node_id=1, score=0.5)
    assert generated.enforce_evaluation_integrity(generated_node) is True
    assert generated_node.evaluation_provenance == AGENT_GENERATED


def test_invalid_evaluations_cannot_win_strategy_or_history_selection(
    tmp_path: Path,
) -> None:
    invalid = SearchNode(
        node_id=0,
        solution="tampered",
        branch_name="candidate_0",
        score=100.0,
        evaluation_valid=False,
        evaluation_provenance=PROVIDED,
    )
    valid = SearchNode(
        node_id=1,
        solution="valid",
        branch_name="candidate_1",
        score=0.2,
    )

    generic = GenericSearch.__new__(GenericSearch)
    generic.node_history = [invalid, valid]
    generic.problem_handler = SimpleNamespace(maximize_scoring=True)
    assert generic.get_best_experiment() is valid

    tree = BenchmarkTreeSearch.__new__(BenchmarkTreeSearch)
    tree.node_history = [invalid, valid]
    tree.problem_handler = SimpleNamespace(maximize_scoring=True)
    assert tree.get_best_experiment() is valid

    store = ExperimentHistoryStore(
        json_path=str(tmp_path / "history.json"),
    )
    store.add_experiment(invalid)
    store.add_experiment(valid)
    assert [record.node_id for record in store.get_top_experiments()] == [1]

    persisted = json.loads((tmp_path / "history.json").read_text())
    assert persisted[0]["evaluation_valid"] is False
    assert persisted[0]["evaluation_provenance"] == PROVIDED


def test_feedback_invalidity_cannot_stop_or_retain_a_score() -> None:
    strategy = GenericSearch.__new__(GenericSearch)
    strategy.goal = "Improve support"
    strategy.budget_snapshot = None
    strategy.budget_snapshot_monotonic = None
    strategy.registered_evaluation_command = ""
    strategy.feedback_generator = SimpleNamespace(
        configured_timeout_seconds=120.0,
        generate=lambda **kwargs: SimpleNamespace(
            feedback="evaluation is not valid",
            score=100.0,
            stop=True,
            evaluation_valid=False,
            duration_seconds=None,
            cost_usd=0.0,
        ),
    )
    node = SearchNode(
        node_id=0,
        branch_name="candidate",
        parent_branch_name="main",
    )

    strategy._generate_feedback(node)

    assert node.evaluation_valid is False
    assert node.score is None
    assert node.should_stop is False


def test_invalid_evaluation_is_stored_but_never_ranked_top(
    tmp_path: Path,
) -> None:
    """Invalid evaluations persist (and stay discoverable via similarity —
    see test_experiment_semantic_search) but never enter the score
    ranking that steers parent selection."""
    store = ExperimentHistoryStore(
        json_path=str(tmp_path / "history.json"),
    )
    invalid = SearchNode(
        node_id=0,
        evaluation_valid=False,
        score=None,
    )
    valid = SearchNode(node_id=1, score=0.2)

    store.add_experiment(invalid)
    store.add_experiment(valid)

    assert [e.node_id for e in store.get_top_experiments()] == [1]
    assert store.get_experiment_count() == 2


def test_integrity_state_round_trip_and_resume_mismatch() -> None:
    source = GenericSearch.__new__(GenericSearch)
    source.evaluation_provenance = PROVIDED
    source.provided_evaluation_manifest = {"evaluate.py": "a" * 64}
    source.provided_evaluation_fingerprint = manifest_fingerprint(
        source.provided_evaluation_manifest
    )
    state = source.dump_evaluation_integrity_state()

    restored = GenericSearch.__new__(GenericSearch)
    restored.evaluation_provenance = PROVIDED
    restored.provided_evaluation_manifest = {"evaluate.py": "a" * 64}
    restored.provided_evaluation_fingerprint = manifest_fingerprint(
        restored.provided_evaluation_manifest
    )
    restored.load_evaluation_integrity_state(state)

    restored.provided_evaluation_manifest = {"evaluate.py": "b" * 64}
    with pytest.raises(ValueError, match="changed on resume"):
        restored.load_evaluation_integrity_state(state)


def test_provided_evaluation_prompt_warns_against_modification(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    _write_suite(suite)
    handler = GenericProblemHandler("Improve support", eval_dir=str(suite))

    context = handler.get_problem_context()

    assert "Do not modify, delete, replace" in context
    assert "result.json" in context


def test_data_tampering_voids_the_score(tmp_path: Path) -> None:
    """The live reward hack: the candidate rewrote data/train.csv so a
    degenerate model scored 1.0 against rigged labels — same evaluator_id,
    different evaluation set. Protected-input enforcement voids it.
    """
    workspace = tmp_path / "workspace"
    repo = _init_candidate_repo(workspace)
    data_dir = workspace / "data"
    data_dir.mkdir()
    (data_dir / "train.csv").write_text("PassengerId,Transported\n1,False\n")
    repo.git.add(["data"])
    repo.git.commit("-m", "evaluation inputs")

    repo.git.checkout("-b", "honest")
    (workspace / "model.py").write_text("WEIGHTS = 2\n")
    repo.git.add("-A")
    repo.git.commit("-m", "honest work")
    repo.git.checkout("main")

    repo.git.checkout("-b", "rigged")
    (data_dir / "train.csv").write_text("PassengerId,Transported\n1,True\n")
    repo.git.add("-A")
    repo.git.commit("-m", "align data with grader constraints")
    repo.git.checkout("main")

    strategy = _provided_strategy(workspace)
    strategy.registered_evaluation_manifest = (
        strategy.provided_evaluation_manifest
    )
    strategy.registered_data_manifest = build_data_manifest(
        workspace, ["data"]
    )

    honest = SearchNode(node_id=0, branch_name="honest", score=0.8)
    assert strategy.enforce_evaluation_integrity(honest) is True
    assert honest.evaluation_valid is True
    assert honest.score == 0.8

    rigged = SearchNode(
        node_id=1, branch_name="rigged", score=1.0, should_stop=True
    )
    assert strategy.enforce_evaluation_integrity(rigged) is False
    assert rigged.evaluation_valid is False
    assert rigged.score is None
    assert rigged.should_stop is False
    assert "modified:data/train.csv" in rigged.evaluation_integrity_error
