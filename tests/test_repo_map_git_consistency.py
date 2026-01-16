"""
RepoMap integration test (portable + git-consistent).

This test is intentionally small and fast. It does NOT call any LLMs.

We validate that the deterministic RepoMap builder:
- Stores a portable `repo_root` (".", not absolute /tmp/... paths)
- Excludes infrastructure/meta paths even if they exist in git:
  - `.kapso/*` (RepoMemory storage)
  - `sessions/*` (nested experiment clones)
  - `changes.log` (observability/audit metadata)
"""

from __future__ import annotations

import json
from pathlib import Path

import git

from src.repo_memory.builders import build_repo_map


def test_build_repo_map_git_filters_meta_paths(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    repo = git.Repo.init(repo_dir)

    # "Real" code file
    (repo_dir / "main.py").write_text("print('hello')\n")

    # Dotfile (should not be mangled to "gitignore")
    (repo_dir / ".gitignore").write_text("*.log\n")

    # Observability metadata (must be excluded from repo structure map)
    (repo_dir / "changes.log").write_text("RepoMemory sections consulted: none\n")

    # RepoMemory storage (must be excluded)
    prax = repo_dir / ".kapso"
    prax.mkdir()
    (prax / "repo_memory.json").write_text(json.dumps({"schema_version": 2}, indent=2))

    # Nested experiment clone (must be excluded)
    sess = repo_dir / "sessions" / "experiment_0"
    sess.mkdir(parents=True)
    (sess / "junk.txt").write_text("x\n")

    # Commit everything (even meta paths) to ensure filtering is robust.
    repo.git.add("-A")
    repo.git.commit("-m", "init")

    repo_map = build_repo_map(str(repo_dir))
    assert repo_map.get("repo_root") == "."

    files = repo_map.get("files", []) or []
    assert "main.py" in files
    assert ".gitignore" in files
    assert "gitignore" not in files  # ensure we don't strip leading '.'

    # Meta exclusions
    assert "changes.log" not in files
    assert not any(p.startswith(".kapso/") for p in files)
    assert "kapso/repo_memory.json" not in files  # guard against dot-stripping regressions
    assert not any(p.startswith("sessions/") for p in files)

