"""
Integration tests for RepoMemory Book (schema v2).

These tests validate behavior across:
- Disk persistence (v1 file upgraded on disk so coding agents can read Book/TOC)
- Git branch loading (read memory from a branch without checkout)
"""

from __future__ import annotations

import json
from pathlib import Path

import git

from src.execution.memories.repo_memory import RepoMemoryManager


def test_ensure_exists_in_worktree_persists_v1_to_v2_migration(tmp_path: Path) -> None:
    # Write a v1 memory file to disk.
    prax = tmp_path / ".kapso"
    prax.mkdir()
    memory_path = prax / "repo_memory.json"
    v1 = {
        "schema_version": 1,
        "generated_at": "2026-01-01T00:00:00Z",
        "repo_model": {
            "summary": "My repo",
            "entrypoints": [{"path": "main.py", "how_to_run": "python main.py"}],
            "where_to_edit": [{"path": "foo.py", "role": "core"}],
            "claims": [],
        },
        "experiments": [],
        "quality": {"evidence_ok": False, "missing_evidence": [], "claim_count": 0},
    }
    memory_path.write_text(json.dumps(v1, indent=2))

    # ensure_exists should upgrade the file on disk (so agents can read Book/TOC).
    doc = RepoMemoryManager.ensure_exists_in_worktree(str(tmp_path))
    assert doc.get("schema_version") == 2
    assert "book" in doc

    reloaded = json.loads(memory_path.read_text())
    assert reloaded.get("schema_version") == 2
    assert "book" in reloaded
    assert "sections" in reloaded["book"]


def test_load_from_git_branch_migrates_v1_doc(tmp_path: Path) -> None:
    repo = git.Repo.init(tmp_path)
    (tmp_path / "README.md").write_text("# Repo\n")

    prax = tmp_path / ".kapso"
    prax.mkdir()
    (prax / "repo_memory.json").write_text(
        json.dumps({"schema_version": 1, "repo_model": {"summary": "x", "claims": []}}, indent=2)
    )

    repo.git.add("-A")
    repo.git.commit("-m", "init with v1 memory")

    doc = RepoMemoryManager.load_from_git_branch(repo, repo.active_branch.name)
    assert doc is not None
    assert doc.get("schema_version") == 2
    assert "book" in doc

