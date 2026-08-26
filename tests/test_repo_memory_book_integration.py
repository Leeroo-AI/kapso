"""RepoMemory Book — disk and git integration.

The v1→v2 auto-migration was deleted (stale-code audit 2026-08-26, B9;
Rule 7). What these paths now guarantee: a v1 file on disk or on a
branch FAILS LOUD instead of being silently upgraded, while a missing
file/branch stays the documented None default.
"""

from __future__ import annotations

import json
from pathlib import Path

import git
import pytest

from kapso.execution.memories.repo_memory import RepoMemoryManager


def write_v1(root: Path) -> Path:
    kapso_dir = root / ".kapso"
    kapso_dir.mkdir()
    memory_path = kapso_dir / "repo_memory.json"
    memory_path.write_text(json.dumps({
        "schema_version": 1,
        "repo_model": {"summary": "x", "claims": []},
    }, indent=2))
    return memory_path


def test_v1_file_on_disk_fails_loud_not_silently_upgraded(tmp_path: Path) -> None:
    write_v1(tmp_path)
    with pytest.raises(ValueError, match="not migrated"):
        RepoMemoryManager.ensure_exists_in_worktree(str(tmp_path))
    with pytest.raises(ValueError, match="not migrated"):
        RepoMemoryManager.load_from_worktree(str(tmp_path))


def test_load_from_git_branch_missing_is_none_but_v1_raises(tmp_path: Path) -> None:
    repo = git.Repo.init(tmp_path)
    (tmp_path / "README.md").write_text("# Repo\n")
    repo.git.add("-A")
    repo.git.commit("-m", "no memory yet")
    # Missing file: the documented None default.
    assert RepoMemoryManager.load_from_git_branch(
        repo, repo.active_branch.name
    ) is None

    write_v1(tmp_path)
    repo.git.add("-A")
    repo.git.commit("-m", "v1 memory")
    # Present-but-unsupported: fail loud (Rule 2), never a silent upgrade.
    with pytest.raises(ValueError, match="not migrated"):
        RepoMemoryManager.load_from_git_branch(repo, repo.active_branch.name)


def test_v2_round_trips_through_disk_and_branch(tmp_path: Path) -> None:
    repo = git.Repo.init(tmp_path)
    doc = RepoMemoryManager.ensure_exists_in_worktree(str(tmp_path))
    assert "repo_model" not in doc
    repo.git.add("-A")
    repo.git.commit("-m", "book memory")
    loaded = RepoMemoryManager.load_from_git_branch(
        repo, repo.active_branch.name
    )
    assert loaded is not None
    assert loaded["book"]["toc"]
    assert "repo_model" not in loaded
