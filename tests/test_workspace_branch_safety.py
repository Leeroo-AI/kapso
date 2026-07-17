from pathlib import Path
from types import SimpleNamespace

import pytest

from kapso.execution.coding_agents.base import CodingAgentConfig
from kapso.execution.experiment_workspace.experiment_workspace import (
    ExperimentWorkspace,
    WorkspaceCheckoutError,
)
from kapso.execution.search_strategies.base import SearchNode
from kapso.execution.search_strategies.generic.strategy import GenericSearch


def _agent_config() -> CodingAgentConfig:
    return CodingAgentConfig(
        agent_type="openhands",
        model="test-model",
        debug_model="test-model",
        agent_specific={},
    )


def _workspace(tmp_path: Path) -> ExperimentWorkspace:
    return ExperimentWorkspace(
        coding_agent_config=_agent_config(),
        workspace_dir=str(tmp_path / "workspace"),
    )


def _commit_file(
    workspace: ExperimentWorkspace,
    relative_path: str,
    content: str,
    message: str,
) -> None:
    path = Path(workspace.workspace_dir, relative_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    # -f: some scenarios deliberately commit paths the workspace ignores
    # (e.g. a session that force-committed interpreter junk).
    workspace.repo.git.add(["-f", relative_path])
    workspace.repo.git.commit("-m", message)


def test_fresh_workspace_uses_main(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    assert workspace.get_current_branch() == "main"
    assert "main" in {branch.name for branch in workspace.repo.branches}


def test_switch_branch_preserves_untracked_and_ignored_files(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    _commit_file(workspace, "version.txt", "main\n", "add main version")
    _commit_file(workspace, ".gitignore", "*.cache\n", "ignore cache files")

    workspace.create_branch("candidate")
    _commit_file(workspace, "version.txt", "candidate\n", "update candidate")
    workspace.switch_branch("main")

    untracked = Path(workspace.workspace_dir, "notes.txt")
    ignored = Path(workspace.workspace_dir, "runtime.cache")
    untracked.write_text("keep me\n")
    ignored.write_text("keep me too\n")

    workspace.switch_branch("candidate")

    assert untracked.read_text() == "keep me\n"
    assert ignored.read_text() == "keep me too\n"
    assert workspace.get_current_branch() == "candidate"


def test_untracked_shadow_of_target_tracked_path_is_replaced(
    tmp_path: Path,
) -> None:
    """A checkout materializes the branch: local shadows of branch-owned
    paths (the __pycache__-kills-the-final-checkout failure) never abort
    it, while untracked files the target does not track survive untouched.
    """
    workspace = _workspace(tmp_path)
    workspace.create_branch("candidate")
    _commit_file(
        workspace,
        "__pycache__/train.cpython-312.pyc",
        "candidate bytecode\n",
        "session committed interpreter junk",
    )
    workspace.switch_branch("main")

    shadow = Path(workspace.workspace_dir, "__pycache__/train.cpython-312.pyc")
    shadow.parent.mkdir(exist_ok=True)
    shadow.write_text("local junk\n")
    novel = Path(workspace.workspace_dir, "notes.md")
    novel.write_text("keep me\n")

    workspace.switch_branch("candidate")

    assert workspace.get_current_branch() == "candidate"
    assert shadow.read_text() == "candidate bytecode\n"
    assert novel.read_text() == "keep me\n"


def test_modified_tracked_file_still_fails_checkout_loudly(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    _commit_file(workspace, "shared.txt", "main version\n", "add shared")
    workspace.create_branch("candidate")
    _commit_file(
        workspace, "shared.txt", "candidate version\n", "candidate edit"
    )
    workspace.switch_branch("main")

    shared = Path(workspace.workspace_dir, "shared.txt")
    shared.write_text("uncommitted local edit\n")

    with pytest.raises(WorkspaceCheckoutError) as error:
        workspace.switch_branch("candidate")

    assert shared.read_text() == "uncommitted local edit\n"
    assert workspace.get_current_branch() == "main"
    assert error.value.branch_name == "candidate"
    assert any("shared.txt" in line for line in error.value.status_lines)


def test_reopening_workspace_preserves_experiment_branch_and_state(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    workspace.create_branch("generic_exp_0")
    _commit_file(workspace, "candidate.py", "VALUE = 1\n", "add candidate")
    checkpoint = Path(workspace.workspace_dir, "checkpoint.pkl")
    checkpoint.write_bytes(b"checkpoint")

    reopened = ExperimentWorkspace(
        coding_agent_config=_agent_config(),
        workspace_dir=workspace.workspace_dir,
    )

    branches = {branch.name for branch in reopened.repo.branches}
    assert branches >= {"main", "generic_exp_0"}
    assert reopened.get_current_branch() == "main"
    assert checkpoint.read_bytes() == b"checkpoint"


def test_detached_workspace_recreates_main_without_losing_head(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    head = workspace.repo.head.commit.hexsha
    workspace.repo.git.checkout("--detach", head)
    workspace.repo.git.branch("-D", "main")

    reopened = ExperimentWorkspace(
        coding_agent_config=_agent_config(),
        workspace_dir=workspace.workspace_dir,
    )

    assert reopened.get_current_branch() == "main"
    assert reopened.repo.head.commit.hexsha == head


def test_materialize_ref_uses_exact_candidate_without_switching_root(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    _commit_file(workspace, "version.txt", "main\n", "add main version")
    workspace.create_branch("candidate")
    _commit_file(workspace, "version.txt", "candidate\n", "update candidate")
    workspace.switch_branch("main")

    with workspace.materialize_ref("candidate") as candidate_dir:
        materialized_path = Path(candidate_dir)
        assert (
            materialized_path.joinpath("version.txt").read_text()
            == "candidate\n"
        )
        assert workspace.get_current_branch() == "main"

    assert not materialized_path.exists()
    assert workspace.get_current_branch() == "main"


def test_materialize_unknown_ref_fails_clearly(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    with pytest.raises(ValueError, match="Unknown Git ref"):
        with workspace.materialize_ref("missing"):
            pass


def test_generic_strategy_returns_verified_best_branch(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    workspace.create_branch("generic_exp_0")
    _commit_file(workspace, "candidate.py", "VALUE = 1\n", "add candidate")
    workspace.switch_branch("main")

    strategy = GenericSearch.__new__(GenericSearch)
    strategy.workspace = workspace
    strategy.registered_evaluator_id = ""
    strategy.node_history = [
        SearchNode(node_id=0, branch_name="generic_exp_0", score=0.9)
    ]
    strategy.problem_handler = SimpleNamespace(maximize_scoring=True)

    selected = strategy.checkout_to_best_experiment_branch()

    assert selected == "generic_exp_0"
    assert workspace.get_current_branch() == selected


def test_session_creation_clears_crash_corpses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crashed attempt leaves a session folder (and, if it died between
    push and checkpoint, a workspace branch) under the redo's name. Session
    creation owns both: the redo clones fresh and branches from the parent,
    never from the corpse.
    """
    from kapso.execution.experiment_workspace import (
        experiment_session as session_module,
    )

    workspace = _workspace(tmp_path)
    _commit_file(workspace, "train.py", "parent version\n", "baseline")

    # Corpse branch: a pushed attempt whose node never landed in history.
    workspace.create_branch("generic_exp_0")
    _commit_file(workspace, "train.py", "corpse version\n", "dead attempt")
    workspace.switch_branch("main")

    # Corpse session folder from the same dead attempt.
    corpse_dir = Path(workspace.workspace_dir, "sessions", "generic_exp_0")
    corpse_dir.mkdir(parents=True)
    (corpse_dir / "leftover.txt").write_text("junk\n")

    monkeypatch.setattr(
        session_module.CodingAgentFactory,
        "create",
        classmethod(
            lambda cls, config: SimpleNamespace(
                initialize=lambda folder: None,
                cleanup=lambda: None,
                supports_native_git=lambda: False,
            )
        ),
    )
    session = workspace.create_experiment_session("generic_exp_0", "main")

    assert not (Path(session.session_folder) / "leftover.txt").exists()
    session_train = Path(session.session_folder, "train.py")
    assert session_train.read_text() == "parent version\n"
    assert session.repo.active_branch.name == "generic_exp_0"


def test_seeded_bytecode_is_neither_copied_nor_ever_tracked(tmp_path: Path) -> None:
    """Seeding from a plain directory containing __pycache__ used to track
    the bytecode in the seed commit; regenerated .pyc then made those files
    dirty and every later branch checkout failed loudly. The seed copy must
    drop bytecode and the workspace .gitignore must keep it ignored."""
    seed = tmp_path / "seed"
    (seed / "__pycache__").mkdir(parents=True)
    (seed / "pkg").mkdir()
    (seed / "main.py").write_text("print('hi')\n")
    (seed / "__pycache__" / "main.cpython-312.pyc").write_bytes(b"\x00stale")
    (seed / "pkg" / "util.pyc").write_bytes(b"\x00stale")

    workspace = ExperimentWorkspace(
        coding_agent_config=_agent_config(),
        workspace_dir=str(tmp_path / "workspace"),
        initial_repo=str(seed),
    )

    tracked = workspace.repo.git.ls_files().splitlines()
    assert not [f for f in tracked if f.endswith(".pyc")]
    gitignore = Path(workspace.workspace_dir, ".gitignore").read_text()
    assert "__pycache__/" in gitignore
    assert "*.pyc" in gitignore

    # Regenerated bytecode must stay ignored: branch switches survive it.
    workspace.repo.git.checkout("-b", "exp_0")
    pycache = Path(workspace.workspace_dir, "__pycache__")
    pycache.mkdir(exist_ok=True)
    (pycache / "main.cpython-312.pyc").write_bytes(b"\x00fresh")
    workspace.switch_branch("main")
    assert workspace.get_current_branch() == "main"
