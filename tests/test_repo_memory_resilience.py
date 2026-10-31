import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import git
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
    build_repo_map,
    infer_repo_model_update,
    infer_repo_model_with_retry,
)
from kapso.execution.search_strategies.base import (
    SearchNode,
    SearchStrategy,
    SearchStrategyConfig,
)


VALID_MODEL = '{"summary": "Useful memory", "sections": {}}'
VALID_UPDATE = '{"summary": "Useful memory", "sections": {}, "removed_sections": []}'
VALID_PLAN = (
    '{"files_to_read": [{"path": "README.md", "why": "overview"}]}'
)


class SequenceLLM:
    def __init__(self, responses: List[Any]):
        self.responses = iter(responses)
        self.calls: List[Dict[str, Any]] = []

    def llm_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> str:
        self.calls.append({"messages": messages, "kwargs": kwargs})
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
            repo_root=str(repo_root),
            repo_map=_repo_map(),
            max_retries=4,
        )

    assert len(llm.calls) == 2


def test_incremental_update_repairs_invalid_schema(tmp_path: Path) -> None:
    repo_root = _prepare_repo(tmp_path)
    llm = SequenceLLM(["{}", VALID_UPDATE])

    result = infer_repo_model_update(
        llm=llm,
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
        ({"repo_memory_update_payload_max_chars": 0}, "positive integer"),
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


def _opt_section(name: str, path: str = "src/other.py") -> Dict[str, Any]:
    return {
        "title": name,
        "one_liner": name,
        "claims": [
            {
                "statement": f"{name} " + "x" * 600,
                "evidence": [{"path": path, "line": 1}],
            }
        ],
    }


def test_incremental_update_sends_full_previous_model_and_keeps_omitted_sections(
    tmp_path: Path,
) -> None:
    repo_root = _prepare_repo(tmp_path)
    previous_sections = {
        f"opt.s{i:02d}": _opt_section(f"s{i:02d}") for i in range(40)
    }
    # An id the builder has no special knowledge of must survive untouched too.
    previous_sections["notes.unrecognized"] = _opt_section("unrecognized")
    previous_model = {"summary": "Old", "sections": previous_sections}
    previous_json = json.dumps(previous_model, indent=2)
    assert len(previous_json) > 20000
    update = {
        "summary": "New",
        "sections": {"opt.s00": _opt_section("changed-00")},
        "removed_sections": [],
    }
    llm = SequenceLLM([json.dumps(update)])

    result = infer_repo_model_update(
        llm=llm,
        repo_root=str(repo_root),
        repo_map=_repo_map(),
        previous_model=previous_model,
        diff_summary="README changed",
        changed_files=["README.md"],
        max_retries=0,
    )

    prompt = llm.calls[0]["messages"][0]["content"]
    assert previous_json in prompt
    assert result["summary"] == "New"
    assert set(result["sections"]) == set(previous_sections)
    assert result["sections"]["opt.s00"] == update["sections"]["opt.s00"]
    assert result["sections"]["opt.s39"] == previous_sections["opt.s39"]
    assert (
        result["sections"]["notes.unrecognized"]
        == previous_sections["notes.unrecognized"]
    )


def test_incremental_update_removes_only_listed_sections(tmp_path: Path) -> None:
    repo_root = _prepare_repo(tmp_path)
    previous_model = {
        "summary": "Old",
        "sections": {"opt.a": _opt_section("a"), "opt.b": _opt_section("b")},
    }
    patch = {"summary": "New", "sections": {}, "removed_sections": ["opt.a"]}
    llm = SequenceLLM([json.dumps(patch)])

    result = infer_repo_model_update(
        llm=llm,
        repo_root=str(repo_root),
        repo_map=_repo_map(),
        previous_model=previous_model,
        diff_summary="README changed",
        changed_files=["README.md"],
        max_retries=0,
    )

    assert set(result["sections"]) == {"opt.b"}
    assert result["sections"]["opt.b"] == previous_model["sections"]["opt.b"]


@pytest.mark.parametrize(
    "patch",
    [
        {"summary": "New", "sections": {}},
        {"summary": "New", "sections": {}, "removed_sections": ["core.gotchas"]},
        {"summary": "New", "sections": {}, "removed_sections": ["opt.missing"]},
        {
            "summary": "New",
            "sections": {"opt.a": _opt_section("a2")},
            "removed_sections": ["opt.a"],
        },
    ],
    ids=[
        "missing-removed-sections",
        "removes-core-section",
        "removes-unknown-section",
        "removes-and-returns-same-section",
    ],
)
def test_incremental_update_rejects_invalid_patch(
    tmp_path: Path,
    patch: Dict[str, Any],
) -> None:
    repo_root = _prepare_repo(tmp_path)
    previous_model = {
        "summary": "Old",
        "sections": {
            "core.gotchas": _opt_section("gotchas"),
            "opt.a": _opt_section("a"),
        },
    }
    llm = SequenceLLM([json.dumps(patch)])

    with pytest.raises(RepoMemoryResponseError):
        infer_repo_model_update(
            llm=llm,
            repo_root=str(repo_root),
            repo_map=_repo_map(),
            previous_model=previous_model,
            diff_summary="README changed",
            changed_files=["README.md"],
            max_retries=0,
        )


def test_incremental_update_repairs_invalid_patch(tmp_path: Path) -> None:
    repo_root = _prepare_repo(tmp_path)
    previous_model = {
        "summary": "Old",
        "sections": {"core.gotchas": _opt_section("gotchas")},
    }
    bad_patch = {"summary": "New", "sections": {}, "removed_sections": ["core.gotchas"]}
    good_patch = {"summary": "New", "sections": {}, "removed_sections": []}
    llm = SequenceLLM([json.dumps(bad_patch), json.dumps(good_patch)])

    result = infer_repo_model_update(
        llm=llm,
        repo_root=str(repo_root),
        repo_map=_repo_map(),
        previous_model=previous_model,
        diff_summary="README changed",
        changed_files=["README.md"],
        max_retries=1,
    )

    assert result["summary"] == "New"
    assert set(result["sections"]) == {"core.gotchas"}
    assert len(llm.calls) == 2


def test_incremental_update_requires_sections_citing_changed_files(
    tmp_path: Path,
) -> None:
    repo_root = _prepare_repo(tmp_path)
    previous_model = {
        "summary": "Old",
        "sections": {
            "opt.cites": _opt_section("cites", path="README.md"),
            "opt.other": _opt_section("other"),
        },
    }
    omitting = {"summary": "New", "sections": {}, "removed_sections": []}
    returning = {
        "summary": "New",
        "sections": {"opt.cites": _opt_section("refreshed", path="README.md")},
        "removed_sections": [],
    }
    removing = {"summary": "New", "sections": {}, "removed_sections": ["opt.cites"]}

    def update(patch: Dict[str, Any]) -> Dict[str, Any]:
        return infer_repo_model_update(
            llm=SequenceLLM([json.dumps(patch)]),
            repo_root=str(repo_root),
            repo_map=_repo_map(),
            previous_model=previous_model,
            diff_summary="README changed",
            changed_files=["README.md"],
            max_retries=0,
        )

    with pytest.raises(RepoMemoryResponseError) as rejected:
        update(omitting)
    assert "opt.cites" in str(rejected.value)
    assert "opt.other" not in str(rejected.value)

    refreshed = update(returning)
    assert refreshed["sections"]["opt.cites"] == returning["sections"]["opt.cites"]
    assert refreshed["sections"]["opt.other"] == previous_model["sections"]["opt.other"]

    removed = update(removing)
    assert set(removed["sections"]) == {"opt.other"}


def test_update_after_experiment_sends_the_whole_diff_summary_to_the_model(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    repo = git.Repo.init(root)
    with repo.config_writer() as config:
        config.set_value("user", "name", "tester")
        config.set_value("user", "email", "tester@example.com")
    (root / "README.md").write_text("# Example\n")
    repo.git.add("README.md")
    repo.git.commit("-m", "base")
    base_commit = repo.head.commit.hexsha
    RepoMemoryManager.write_to_worktree(
        str(root),
        {
            "schema_version": 2,
            "book": {"summary": "Old", "sections": {"opt.a": _opt_section("a")}},
        },
    )
    (root / "pkg").mkdir()
    for index in range(400):
        (root / "pkg" / f"module_{index:04d}_{'n' * 40}.py").write_text("x = 1\n")
    repo.git.add("pkg")
    repo.git.commit("-m", "many files")
    diff_summary = repo.git.diff("--stat", base_commit, repo.head.commit.hexsha)
    assert len(diff_summary) > 8000
    llm = SequenceLLM([VALID_UPDATE])

    RepoMemoryManager.update_after_experiment(
        repo_root=str(root),
        llm=llm,
        branch_name="exp",
        parent_branch_name="main",
        base_commit_sha=base_commit,
        solution_spec="spec",
        run_result={},
        max_retries=0,
    )

    assert diff_summary in llm.calls[0]["messages"][0]["content"]


def test_repo_memory_reads_unusual_file_names_exactly_as_on_disk(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    repo = git.Repo.init(root)
    with repo.config_writer() as config:
        config.set_value("user", "name", "tester")
        config.set_value("user", "email", "tester@example.com")
    (root / "README.md").write_text("# Example\n")
    (root / "old_name.py").write_text("print('moved')\n" * 20)
    repo.git.add("-A")
    repo.git.commit("-m", "base")
    base_commit = repo.head.commit.hexsha
    RepoMemoryManager.write_to_worktree(
        str(root),
        {
            "schema_version": 2,
            "book": {"summary": "Old", "sections": {"opt.a": _opt_section("a")}},
        },
    )
    unusual_names = ["café.py", "日本語.py", "has space.py", 'quo"te.py']
    for name in unusual_names:
        (root / name).write_text("x = 1\n")
    repo.git.mv("old_name.py", "renamed_café.py")
    repo.git.add("-A")
    repo.git.commit("-m", "unusual names")

    assert set(unusual_names) <= set(build_repo_map(str(root))["files"])

    RepoMemoryManager.update_after_experiment(
        repo_root=str(root),
        llm=SequenceLLM([VALID_UPDATE]),
        branch_name="exp",
        parent_branch_name="main",
        base_commit_sha=base_commit,
        solution_spec="spec",
        run_result={},
        max_retries=0,
    )

    experiment = RepoMemoryManager.load_from_worktree(str(root))["experiments"][-1]
    expected = {*unusual_names, "renamed_café.py"}
    assert expected <= set(experiment["changed_files"])
    assert expected <= {entry["path"] for entry in experiment["diff_numstat"]}
    assert not any(entry["path"].startswith('"') for entry in experiment["diff_numstat"])


def test_update_shows_the_model_every_changed_file_it_must_refresh(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    changed = [f"file_{index:02d}.py" for index in range(25)]
    for name in changed:
        (root / name).write_text("a\nb\nc\n")
    previous_model = {
        "summary": "Old",
        "sections": {"core.x": _opt_section("core.x", path="file_24.py")},
    }
    refreshed = {
        "summary": "New",
        "sections": {"core.x": _opt_section("core.x", path="file_24.py")},
        "removed_sections": [],
    }
    llm = SequenceLLM([json.dumps(refreshed)])

    update = infer_repo_model_update(
        llm=llm,
        repo_root=str(root),
        repo_map=_repo_map(),
        previous_model=previous_model,
        diff_summary="25 files changed",
        changed_files=changed,
        max_retries=0,
    )

    prompt = llm.calls[0]["messages"][0]["content"]
    assert all(f"=== FILE: {name} ===" in prompt for name in changed)
    assert update["sections"]["core.x"] == refreshed["sections"]["core.x"]
