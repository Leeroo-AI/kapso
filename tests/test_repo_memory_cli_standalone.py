"""
RepoMemory CLI tests
====================

We expose RepoMemory sections to coding agents via a CLI:
`python -m src.execution.memories.repo_memory.cli`.

These tests validate:
- the CLI can render a single section by ID from a synthetic repo_memory.json
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_repo_memory_cli_get_section(tmp_path: Path):
    repo_root = tmp_path / "repo"
    kapso_dir = repo_root / ".kapso"
    kapso_dir.mkdir(parents=True)

    (repo_root / "README.md").write_text("hello\n")

    doc = {
        "schema_version": 2,
        "generated_at": "2026-01-01T00:00:00Z",
        "book": {
            "summary": "Test repo",
            "toc": [{"id": "core.architecture", "title": "Architecture", "one_liner": "Test"}],
            "sections": {
                "core.architecture": {
                    "title": "Architecture",
                    "one_liner": "Test",
                    "claims": [
                        {
                            "kind": "architecture",
                            "statement": "Has a README",
                            "confidence": 1.0,
                            "evidence": [{"path": "README.md", "quote": "hello"}],
                        }
                    ],
                }
            },
        },
    }
    (kapso_dir / "repo_memory.json").write_text(json.dumps(doc))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.execution.memories.repo_memory.cli",
            "--repo-root",
            str(repo_root),
            "get-section",
            "core.architecture",
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),  # Run from project root
    )
    assert result.returncode == 0, result.stderr
    assert "Architecture" in result.stdout
    assert "Has a README" in result.stdout
