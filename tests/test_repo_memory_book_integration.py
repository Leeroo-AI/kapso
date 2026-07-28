"""Disk and Git integration for canonical repository memory."""

from __future__ import annotations

import json
from pathlib import Path

import git
import pytest

from kapso.execution.memories.repo_memory import RepoMemoryManager


def test_ensure_exists_rejects_superseded_flat_document(tmp_path: Path) -> None:
    memory_root = tmp_path / ".kapso"
    memory_root.mkdir()
    memory_path = memory_root / "repo_memory.json"
    memory_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repo_model": {"summary": "old", "claims": []},
            }
        )
    )

    with pytest.raises(ValueError, match="fields are invalid"):
        RepoMemoryManager.ensure_exists_in_worktree(str(tmp_path))
    assert json.loads(memory_path.read_text())["schema_version"] == 1


def test_load_from_git_branch_reads_exact_book_document(tmp_path: Path) -> None:
    repo = git.Repo.init(tmp_path)
    with repo.config_writer() as config:
        config.set_value("user", "name", "Repository Memory Test")
        config.set_value("user", "email", "repo-memory@example.com")
    (tmp_path / "README.md").write_text("# Repo\n")
    expected = RepoMemoryManager.ensure_exists_in_worktree(str(tmp_path))
    repo.git.add("-A")
    repo.git.commit("-m", "add repository memory")

    loaded = RepoMemoryManager.load_from_git_branch(
        repo,
        repo.active_branch.name,
    )
    assert loaded == expected


def test_load_from_git_branch_returns_none_when_memory_is_absent(
    tmp_path: Path,
) -> None:
    repo = git.Repo.init(tmp_path)
    with repo.config_writer() as config:
        config.set_value("user", "name", "Repository Memory Test")
        config.set_value("user", "email", "repo-memory@example.com")
    (tmp_path / "README.md").write_text("# Repo\n")
    repo.git.add("README.md")
    repo.git.commit("-m", "initial")

    assert RepoMemoryManager.load_from_git_branch(repo, repo.active_branch.name) is None
