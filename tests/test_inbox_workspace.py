"""The inbox continuation takes a branch as the stopped session left it.

What must hold: with continue_branch the session lands on the branch's
head with every commit the stopped session pushed; the default path
still recreates the branch from its parent (the crash-corpse rule).
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from kapso.execution.coding_agents.base import CodingAgentConfig
from kapso.execution.experiment_workspace import experiment_session as session_module
from kapso.execution.experiment_workspace.experiment_workspace import ExperimentWorkspace


def _workspace(tmp_path: Path) -> ExperimentWorkspace:
    return ExperimentWorkspace(
        coding_agent_config=CodingAgentConfig(
            agent_type="openhands", model="m", debug_model="m", agent_specific={}
        ),
        workspace_dir=str(tmp_path / "workspace"),
    )


def _commit(workspace, relative_path, content, message):
    path = Path(workspace.workspace_dir, relative_path)
    path.write_text(content)
    workspace.repo.git.add([relative_path])
    workspace.repo.git.commit("-m", message)


@pytest.fixture
def stub_agent(monkeypatch):
    monkeypatch.setattr(
        session_module.CodingAgentFactory,
        "create",
        classmethod(lambda cls, config: SimpleNamespace(
            initialize=lambda folder: None,
            cleanup=lambda: None,
            supports_native_git=lambda: False,
        )),
    )


def test_a_file_dropped_into_kapso_datasets_reaches_the_session(tmp_path, stub_agent):
    """The person drops a requested file into the campaign's kapso_datasets/
    after launch (untracked); the continued session's clone must have it,
    and a tracked file of the same name is not overwritten."""
    workspace = _workspace(tmp_path)
    Path(workspace.workspace_dir, "kapso_datasets").mkdir()
    _commit(workspace, "kapso_datasets/.gitkeep", "", "setup")
    _commit(workspace, "kapso_datasets/train.csv", "tracked\n", "data")
    workspace.create_branch("generic_exp_3")
    _commit(workspace, "PLAN.md", "next: rank\n", "session commit")
    workspace.switch_branch("main")
    dropped = Path(workspace.workspace_dir, "kapso_datasets", "private", "extra.txt")
    dropped.parent.mkdir()
    dropped.write_text("100\n200\n")
    Path(workspace.workspace_dir, "kapso_datasets", "train.csv").write_text("edited in place\n")

    session = workspace.create_experiment_session("generic_exp_3", "main", continue_branch=True)

    assert Path(session.session_folder, "kapso_datasets", "private", "extra.txt").read_text() == "100\n200\n"
    assert Path(session.session_folder, "kapso_datasets", "train.csv").read_text() == "tracked\n"


def test_continue_branch_keeps_the_stopped_sessions_commits(tmp_path, stub_agent):
    workspace = _workspace(tmp_path)
    _commit(workspace, "train.py", "parent\n", "baseline")
    workspace.create_branch("generic_exp_3")
    _commit(workspace, "train.py", "first\n", "session commit 1")
    _commit(workspace, "PLAN.md", "next: embed\n", "session commit 2")
    workspace.switch_branch("main")

    session = workspace.create_experiment_session("generic_exp_3", "main", continue_branch=True)

    assert session.repo.active_branch.name == "generic_exp_3"
    assert Path(session.session_folder, "train.py").read_text() == "first\n"
    assert Path(session.session_folder, "PLAN.md").read_text() == "next: embed\n"
    assert session.base_commit_sha == workspace.repo.commit("generic_exp_3").hexsha
    assert Path(session.run_dir).is_dir()


def test_default_path_still_recreates_from_the_parent(tmp_path, stub_agent):
    workspace = _workspace(tmp_path)
    _commit(workspace, "train.py", "parent\n", "baseline")
    workspace.create_branch("generic_exp_3")
    _commit(workspace, "train.py", "corpse\n", "dead attempt")
    workspace.switch_branch("main")

    session = workspace.create_experiment_session("generic_exp_3", "main")

    assert session.repo.active_branch.name == "generic_exp_3"
    assert Path(session.session_folder, "train.py").read_text() == "parent\n"
