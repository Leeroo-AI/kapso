import logging
from pathlib import Path
from typing import Any, Dict, List

import pytest

from kapso.execution.coding_agents.base import CodingAgentConfig
from kapso.execution.coding_agents.factory import CodingAgentFactory
from kapso.execution.experiment_workspace.experiment_workspace import (
    ExperimentWorkspace,
)
from kapso.execution.memories.repo_memory import (
    RepoMemoryManager,
    RepoMemoryResponseError,
)
from kapso.execution.memories.repo_memory.builders import (
    infer_repo_model_update,
    infer_repo_model_with_retry,
)
from kapso.execution.search_strategies.base import (
    SearchNode,
    SearchStrategy,
    SearchStrategyConfig,
)


VALID_MODEL = '{"summary": "Useful memory", "sections": {}}'
VALID_PLAN = (
    '{"files_to_read": [{"path": "README.md", "why": "overview"}]}'
)


class SequenceLLM:
    def __init__(self, responses: List[Any]):
        self.responses = iter(responses)
        self.calls: List[Dict[str, Any]] = []

    def llm_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> str:
        self.calls.append(
            {"model": model, "messages": messages, "kwargs": kwargs}
        )
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response


class FakeCodingAgent:
    def __init__(self) -> None:
        self.cleanup_called = False

    def initialize(self, workspace: str) -> None:
        self.workspace = workspace

    def supports_native_git(self) -> bool:
        return False

    def cleanup(self) -> None:
        self.cleanup_called = True

    def get_cumulative_cost(self) -> float:
        return 0.0


class MemoryUpdateFailure(RuntimeError):
    pass


class MinimalSearchStrategy(SearchStrategy):
    def run(self, context: Any, budget_progress: float = 0.0) -> None:
        return None

    def get_experiment_history(
        self,
        best_last: bool = False,
    ) -> List[SearchNode]:
        return []

    def get_best_experiment(self) -> None:
        return None

    def checkout_to_best_experiment_branch(self) -> None:
        return None

    def export_checkpoint(self) -> None:
        return None

    def import_checkpoint(self) -> None:
        return None


def _repo_map() -> Dict[str, Any]:
    return {
        "files": ["README.md"],
        "key_files": ["README.md"],
        "entrypoints": [],
    }


def _prepare_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    repo_root.joinpath("README.md").write_text("# Example\n")
    return repo_root


def _agent_config() -> CodingAgentConfig:
    return CodingAgentConfig(
        agent_type="fake",
        model="test-model",
        debug_model="test-model",
        agent_specific={},
    )


def _strategy_config(
    seed_repo: Path,
    llm: SequenceLLM,
    policy: str,
) -> SearchStrategyConfig:
    return SearchStrategyConfig(
        problem_handler=object(),
        llm=llm,
        coding_agent_config=_agent_config(),
        params={"repo_memory_failure_policy": policy},
        initial_repo=str(seed_repo),
    )


def _failing_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
) -> tuple[ExperimentWorkspace, Any, FakeCodingAgent, MemoryUpdateFailure]:
    agent = FakeCodingAgent()
    monkeypatch.setattr(
        CodingAgentFactory,
        "create",
        classmethod(lambda cls, config: agent),
    )

    failure = MemoryUpdateFailure("malformed memory response")

    def fail_update(cls: type, **kwargs: Any) -> None:
        raise failure

    monkeypatch.setattr(
        RepoMemoryManager,
        "update_after_experiment",
        classmethod(fail_update),
    )

    workspace = ExperimentWorkspace(
        coding_agent_config=_agent_config(),
        workspace_dir=str(tmp_path / "workspace"),
        repo_memory_failure_policy=policy,
    )
    session = workspace.create_experiment_session(
        branch_name="candidate",
        parent_branch_name="main",
        llm=object(),
    )
    Path(session.session_folder, "candidate.txt").write_text("completed\n")
    session.schedule_repo_memory_update(
        solution_spec="Add candidate",
        run_result={"score": 1.0},
    )
    return workspace, session, agent, failure


def test_initial_inference_repairs_json_without_replanning(
    tmp_path: Path,
) -> None:
    repo_root = _prepare_repo(tmp_path)
    llm = SequenceLLM([VALID_PLAN, "{broken", VALID_MODEL])

    result = infer_repo_model_with_retry(
        llm=llm,
        model="test-model",
        repo_root=str(repo_root),
        repo_map=_repo_map(),
        max_retries=2,
    )

    planning_calls = [
        call
        for call in llm.calls
        if "files_to_read" in call["messages"][0]["content"]
    ]
    assert result["summary"] == "Useful memory"
    assert len(planning_calls) == 1
    assert len(llm.calls) == 3
    assert len(llm.calls[-1]["messages"]) == 3
    assert "could not be accepted" in llm.calls[-1]["messages"][-1][
        "content"
    ]


def test_initial_inference_stops_after_configured_retry_limit(
    tmp_path: Path,
) -> None:
    repo_root = _prepare_repo(tmp_path)
    llm = SequenceLLM([VALID_PLAN, "bad", "still bad", "also bad"])

    with pytest.raises(RepoMemoryResponseError):
        infer_repo_model_with_retry(
            llm=llm,
            model="test-model",
            repo_root=str(repo_root),
            repo_map=_repo_map(),
            max_retries=2,
        )

    assert len(llm.calls) == 4


def test_provider_failure_is_not_retried(tmp_path: Path) -> None:
    repo_root = _prepare_repo(tmp_path)
    llm = SequenceLLM(
        [VALID_PLAN, PermissionError("invalid provider credentials")]
    )

    with pytest.raises(PermissionError, match="credentials"):
        infer_repo_model_with_retry(
            llm=llm,
            model="test-model",
            repo_root=str(repo_root),
            repo_map=_repo_map(),
            max_retries=4,
        )

    assert len(llm.calls) == 2


def test_incremental_update_repairs_invalid_schema(tmp_path: Path) -> None:
    repo_root = _prepare_repo(tmp_path)
    llm = SequenceLLM(["{}", VALID_MODEL])

    result = infer_repo_model_update(
        llm=llm,
        model="test-model",
        repo_root=str(repo_root),
        repo_map=_repo_map(),
        previous_model={"summary": "Old", "sections": {}},
        diff_summary="README changed",
        changed_files=["README.md"],
        max_retries=1,
    )

    assert result["summary"] == "Useful memory"
    assert len(llm.calls) == 2


def test_warn_policy_keeps_deterministic_baseline_after_bootstrap_failure(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    seed_repo = _prepare_repo(tmp_path)
    llm = SequenceLLM([PermissionError("missing bootstrap credentials")])

    with caplog.at_level(logging.WARNING):
        strategy = MinimalSearchStrategy(
            _strategy_config(seed_repo, llm, policy="warn"),
            workspace_dir=str(tmp_path / "workspace"),
        )

    doc = RepoMemoryManager.load_from_worktree(strategy.workspace_dir)
    assert doc is not None
    assert doc["repo_map"]["files"]
    assert "deterministic repository map only" in caplog.text


def test_fail_policy_propagates_bootstrap_failure(tmp_path: Path) -> None:
    seed_repo = _prepare_repo(tmp_path)
    llm = SequenceLLM([PermissionError("missing bootstrap credentials")])

    with pytest.raises(PermissionError, match="bootstrap credentials"):
        MinimalSearchStrategy(
            _strategy_config(seed_repo, llm, policy="fail"),
            workspace_dir=str(tmp_path / "workspace"),
        )


def test_warn_policy_pushes_and_cleans_up_after_memory_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    workspace, session, agent, _ = _failing_session(
        tmp_path,
        monkeypatch,
        policy="warn",
    )

    with caplog.at_level(logging.WARNING):
        workspace.finalize_session(session)

    assert workspace.repo.git.show("candidate:candidate.txt") == "completed"
    assert agent.cleanup_called
    assert not Path(session.session_folder).exists()
    assert "will still be pushed" in caplog.text


def test_fail_policy_pushes_and_cleans_up_before_reraising(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, session, agent, failure = _failing_session(
        tmp_path,
        monkeypatch,
        policy="fail",
    )

    with pytest.raises(MemoryUpdateFailure) as error:
        workspace.finalize_session(session)

    assert error.value is failure
    assert workspace.repo.git.show("candidate:candidate.txt") == "completed"
    assert agent.cleanup_called
    assert not Path(session.session_folder).exists()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"repo_memory_failure_policy": "ignore"}, "warn.*fail"),
        ({"repo_memory_max_retries": -1}, "non-negative integer"),
        ({"repo_memory_max_retries": True}, "non-negative integer"),
    ],
)
def test_workspace_rejects_invalid_memory_configuration(
    tmp_path: Path,
    kwargs: Dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ExperimentWorkspace(
            coding_agent_config=_agent_config(),
            workspace_dir=str(tmp_path / "workspace"),
            **kwargs,
        )
