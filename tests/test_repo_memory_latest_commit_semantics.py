"""
Latest-commit semantics for RepoMemory updates
=============================================

We want `.kapso/repo_memory.json` to reflect a *committed* code state, not a dirty
worktree. The engine now schedules RepoMemory updates to run inside
`ExperimentSession.close_session()` after final commits and before push/cleanup.

This test asserts the resulting git history shape:
- The branch HEAD commit message is the RepoMemory metadata commit
- The recorded `head_commit` / `code_head_commit` inside RepoMemory matches the
  parent commit of the memory commit (the last code/data commit)

NOTE: This test uses real LLM calls via `LLMBackend()` and costs money.
"""

from __future__ import annotations

import json
from pathlib import Path

import git
from dotenv import load_dotenv

load_dotenv()

from kapso.core.llm import LLMBackend
from kapso.execution.coding_agents.base import CodingAgentConfig
from kapso.execution.experiment_workspace.experiment_workspace import ExperimentWorkspace
from kapso.execution.memories.repo_memory import RepoMemoryManager


def test_repo_memory_update_runs_after_final_commit(tmp_path: Path):
    llm = LLMBackend()

    # Use an agent adapter that can initialize without external deps/keys.
    agent_cfg = CodingAgentConfig(
        agent_type="openhands",
        model="gpt-4.1-mini",
        debug_model="gpt-4.1-mini",
        agent_specific={},
    )

    workspace_dir = tmp_path / "ws"
    workspace = ExperimentWorkspace(
        coding_agent_config=agent_cfg,
        workspace_dir=str(workspace_dir),
    )

    # Seed a minimal codebase on main.
    (workspace_dir / "README.md").write_text("Hello RepoMemory\n")
    workspace.repo.git.add(["README.md"])
    workspace.repo.git.commit("-m", "feat: add README")

    # Ensure baseline memory exists and is committed on main so experiment branches inherit it.
    RepoMemoryManager.ensure_exists_in_worktree(str(workspace_dir))
    workspace.repo.git.add([RepoMemoryManager.MEMORY_REL_PATH])
    if workspace.repo.is_dirty(untracked_files=True):
        workspace.repo.git.commit("-m", "chore(kapso): add baseline repo memory")

    branch_name = "exp_latest_commit_semantics"
    session = workspace.create_experiment_session(branch_name=branch_name, parent_branch_name="main", llm=llm)

    # Make a committed code change (simulates agent commits).
    Path(session.session_folder, "main.py").write_text("print('hi')\n")
    session.repo.git.add(["main.py"])
    session.repo.git.commit("-m", "feat: add main.py")

    # Create run output (committed by close_session via commit_folder).
    Path(session.run_dir, "result.txt").write_text("ok\n")

    # Create an uncommitted changes.log (committed by close_session final commit).
    Path(session.session_folder, "changes.log").write_text("RepoMemory sections consulted: none\n")

    # Schedule memory update (it will run during close_session after the commits above).
    session.schedule_repo_memory_update(
        solution_spec="Add a main.py entrypoint",
        run_result={"score": 1.0, "run_had_error": False},
    )

    # Finalize: commits + RepoMemory update + push back to workspace repo.
    workspace.finalize_session(session)

    # Inspect the resulting branch in the workspace repo.
    head = workspace.repo.commit(branch_name)
    assert "update repo memory" in head.message.lower()

    doc_raw = workspace.repo.git.show(f"{branch_name}:{RepoMemoryManager.MEMORY_REL_PATH}")
    doc = json.loads(doc_raw)
    experiments = doc.get("experiments", []) or []
    assert experiments, "Expected at least one experiment recorded in RepoMemory"

    last = experiments[-1] or {}
    assert last.get("branch") == branch_name

    # The code state this memory describes should be the parent commit of the memory commit.
    parent_sha = head.parents[0].hexsha if head.parents else ""
    assert last.get("head_commit") == parent_sha
    assert last.get("code_head_commit") == parent_sha

