"""A campaign's limits on evaluator changes must outlive the process.

The real orchestrator and the real EvaluationMaintainer run with a scripted
agent and a faked calibration command, the same seam as the wiring tests. Each
process slice is a new OrchestratorAgent over the same workspace.
"""

from pathlib import Path

import git
import pytest

import kapso.execution.orchestrator as orchestrator_module
from kapso.execution.evaluation_maintainer import EvaluationMaintainerError
from kapso.execution.orchestrator import OrchestratorAgent
from kapso.execution.run_checkpoint import RunCheckpointStore

from tests.harness_support import fake_config
from tests.test_evaluation_maintainer_wiring import (
    MAINTAINER_BLOCK,
    ScriptedMaintainerAgent,
    patch_maintainer_environment,
    write_entrypoint,
)
from tests.test_run_checkpoint import (
    FakeKnowledgeSearch,
    FakeProblemHandler,
    _init_git_workspace,
    _orchestrator,
    _patch_orchestrator,
)

CHANGE_REQUEST = (
    "results\n<evaluation_change_request>the harness crashes on the code "
    "suite</evaluation_change_request>"
)
REJECT = (
    "<change_verdict>reject</change_verdict>"
    "<reason>candidate error, not an evaluation defect</reason>"
)
ACCEPT = "<change_verdict>accept</change_verdict><reason>real defect</reason>"


class SimulatedKill(BaseException):
    """The process dying mid-session; not an Exception, so nothing absorbs it."""


def _use_cap(monkeypatch: pytest.MonkeyPatch, cap: int) -> None:
    monkeypatch.setattr(
        orchestrator_module,
        "load_config",
        lambda config_path=None: fake_config(
            evaluation_maintainer={**MAINTAINER_BLOCK, "max_change_requests": cap},
        ),
    )


def _slice(workspace: Path, *, resume: bool, iterations: int = 1):
    """One process's run, every candidate of it asking for a change."""
    orchestrator = _orchestrator(workspace, resume=resume)
    orchestrator.search_strategy.next_agent_output = CHANGE_REQUEST
    orchestrator.solve(experiment_max_iter=iterations)
    return orchestrator


@pytest.fixture
def workspace(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "workspace"
    _init_git_workspace(path)
    _patch_orchestrator(monkeypatch)
    patch_maintainer_environment(
        monkeypatch, ScriptedMaintainerAgent(write_entrypoint, output=REJECT)
    )
    return path


def _sessions_paid() -> int:
    return len(ScriptedMaintainerAgent.calls)


def test_the_cap_is_per_campaign_not_per_process(workspace):
    _slice(workspace, resume=False, iterations=2)
    assert _sessions_paid() == 2  # setup, then the one request the cap allows

    # Three more processes, each with a candidate asking again.
    for _ in range(3):
        _slice(workspace, resume=True)

    assert _sessions_paid() == 2
    assert RunCheckpointStore(str(workspace)).load().change_requests_filed == 1


def test_a_cap_of_two_spans_the_resume(workspace, monkeypatch):
    _use_cap(monkeypatch, 2)
    _slice(workspace, resume=False)
    assert _sessions_paid() == 2

    resumed = _slice(workspace, resume=True, iterations=3)

    assert _sessions_paid() == 3
    assert resumed._change_requests_filed == 2


def test_a_session_killed_midway_still_counts(workspace, monkeypatch):
    """The session is paid for whether or not it finishes, and the process
    can die before any later save, so the count reaches disk first."""
    store = RunCheckpointStore(str(workspace))
    filed_when_the_session_started = []

    def die_in_the_triage(root: Path) -> None:
        write_entrypoint(root)
        if _sessions_paid() == 2:
            filed_when_the_session_started.append(store.load().change_requests_filed)
            raise SimulatedKill()

    patch_maintainer_environment(
        monkeypatch, ScriptedMaintainerAgent(die_in_the_triage, output=REJECT)
    )
    with pytest.raises(SimulatedKill):
        _slice(workspace, resume=False)
    assert filed_when_the_session_started == [1]

    # The round that died is redone and asks again: the cap of one is spent.
    _slice(workspace, resume=True)
    assert _sessions_paid() == 2


def test_the_count_survives_the_heartbeat_and_the_boundary_saves(workspace):
    orchestrator = _slice(workspace, resume=False)
    store = RunCheckpointStore(str(workspace))

    orchestrator._persist_budget_totals()
    assert store.load().change_requests_filed == 1

    orchestrator._save_run_checkpoint(status="running")
    assert store.load().change_requests_filed == 1


def _provided_workspace(path: Path) -> Path:
    """A workspace whose kapso_evaluation/ holds the person's copied files,
    with the originals beside it as the eval_dir they launched with."""
    _init_git_workspace(path)
    for evaluation in (path / "kapso_evaluation", path.parent / "provided"):
        evaluation.mkdir()
        (evaluation / "evaluate.py").write_text("SCORING = 'provided'\n")
    repo = git.Repo(path)
    repo.git.add(["kapso_evaluation"])
    repo.git.commit("-m", "seed the provided evaluation")
    return path


def _provided_orchestrator(path: Path, *, resume: bool) -> OrchestratorAgent:
    return OrchestratorAgent(
        FakeProblemHandler(),
        workspace_dir=str(path),
        resume=resume,
        knowledge_search=FakeKnowledgeSearch(),
        goal="Improve support",
        eval_dir=str(path.parent / "provided"),
    )


def _weaken_on_the_change_request(root: Path) -> None:
    write_entrypoint(root)
    if _sessions_paid() > 1:
        (root / "kapso_evaluation" / "evaluate.py").write_text("SCORING = 'weakened'\n")


def test_provided_files_stay_immutable_after_a_resume(tmp_path, monkeypatch):
    workspace = _provided_workspace(tmp_path / "workspace")
    _patch_orchestrator(monkeypatch)
    patch_maintainer_environment(
        monkeypatch,
        ScriptedMaintainerAgent(_weaken_on_the_change_request, output=ACCEPT),
    )
    _provided_orchestrator(workspace, resume=False).solve(experiment_max_iter=1)
    assert _sessions_paid() == 1

    resumed = _provided_orchestrator(workspace, resume=True)
    resumed.search_strategy.next_agent_output = CHANGE_REQUEST
    with pytest.raises(EvaluationMaintainerError, match="modified"):
        resumed.solve(experiment_max_iter=1)

    assert "immutable: true" in ScriptedMaintainerAgent.calls[-1]
    assert [
        version.version for version in resumed.evaluation_maintainer.registry.versions()
    ] == [1]


def test_our_own_files_can_still_change_after_a_resume(tmp_path, monkeypatch):
    workspace = _provided_workspace(tmp_path / "workspace")
    _patch_orchestrator(monkeypatch)

    def edit_only_the_wrapper(root: Path) -> None:
        write_entrypoint(root)
        if _sessions_paid() > 1:
            (root / "kapso_evaluation" / "kapso_eval.py").write_text("V2 = True\n")

    patch_maintainer_environment(
        monkeypatch, ScriptedMaintainerAgent(edit_only_the_wrapper, output=ACCEPT)
    )
    _provided_orchestrator(workspace, resume=False).solve(experiment_max_iter=1)

    resumed = _provided_orchestrator(workspace, resume=True)
    resumed.search_strategy.next_agent_output = CHANGE_REQUEST
    resumed.solve(experiment_max_iter=1)

    assert resumed.evaluation_maintainer.registry.head().version == 2
