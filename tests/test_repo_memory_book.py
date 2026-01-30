"""
Unit tests for RepoMemory "Book" (schema v2) behavior.

These tests are intentionally lightweight:
- No real LLM calls
- No Neo4j/Weaviate dependencies
- Focus on migration, TOC rendering, section access, and evidence validation
"""

from __future__ import annotations

import json
from typing import Dict, List

import pytest

from kapso.execution.memories.repo_memory import RepoMemoryManager
from kapso.execution.memories.repo_memory.builders import validate_evidence


def test_migrate_v1_to_v2_preserves_data() -> None:
    v1_doc = {
        "schema_version": 1,
        "generated_at": "2026-01-01T00:00:00Z",
        "repo_model": {
            "summary": "Test summary",
            "entrypoints": [{"path": "main.py", "how_to_run": "python main.py"}],
            "where_to_edit": [{"path": "src/core.py", "role": "core logic"}],
            "claims": [
                {
                    "kind": "architecture",
                    "statement": "Uses layered architecture",
                    "confidence": 0.9,
                    "evidence": [{"path": "README.md", "quote": "Architecture"}],
                }
            ],
        },
        "quality": {"evidence_ok": True, "missing_evidence": [], "claim_count": 1},
    }

    v2_doc = RepoMemoryManager.migrate_v1_to_v2(dict(v1_doc))
    assert v2_doc["schema_version"] == 2
    assert "book" in v2_doc
    assert v2_doc["book"]["summary"] == "Test summary"

    sections = v2_doc["book"]["sections"]
    assert sections["core.entrypoints"]["content"] == v1_doc["repo_model"]["entrypoints"]
    assert sections["core.where_to_edit"]["content"] == v1_doc["repo_model"]["where_to_edit"]


def test_migrate_v1_to_v2_idempotent() -> None:
    v1_doc = {"schema_version": 1, "repo_model": {"summary": "x", "claims": []}}
    once = RepoMemoryManager.migrate_v1_to_v2(dict(v1_doc))
    twice = RepoMemoryManager.migrate_v1_to_v2(dict(once))
    assert twice == once


def test_render_summary_and_toc_bounded() -> None:
    doc = RepoMemoryManager.migrate_v1_to_v2({"schema_version": 1, "repo_model": {"summary": "x", "claims": []}})
    out = RepoMemoryManager.render_summary_and_toc(doc, max_chars=120)
    assert len(out) <= 120


def test_render_summary_and_toc_format() -> None:
    doc = RepoMemoryManager.migrate_v1_to_v2({"schema_version": 1, "repo_model": {"summary": "x", "claims": []}})
    out = RepoMemoryManager.render_summary_and_toc(doc, max_chars=2000)
    assert "## Summary" in out
    assert "## Table of Contents" in out
    assert "core.architecture" in out


def test_get_section_found_renders_claims() -> None:
    doc = {
        "schema_version": 2,
        "book": {
            "summary": "x",
            "sections": {
                "core.architecture": {
                    "title": "Architecture",
                    "one_liner": "Design",
                    "claims": [
                        {
                            "kind": "architecture",
                            "statement": "Uses plugins",
                            "evidence": [{"path": "foo.py", "quote": "class Plugin"}],
                        }
                    ],
                }
            },
        },
    }
    out = RepoMemoryManager.get_section(doc, "core.architecture", max_chars=2000)
    assert "Uses plugins" in out
    assert "evidence:" in out


def test_get_section_not_found() -> None:
    doc = RepoMemoryManager.migrate_v1_to_v2({"schema_version": 1, "repo_model": {"summary": "x", "claims": []}})
    out = RepoMemoryManager.get_section(doc, "does.not.exist", max_chars=500)
    assert "not found" in out.lower()


def test_list_sections_returns_toc_metadata() -> None:
    doc = RepoMemoryManager.migrate_v1_to_v2({"schema_version": 1, "repo_model": {"summary": "x", "claims": []}})
    toc = RepoMemoryManager.list_sections(doc)
    assert isinstance(toc, list)
    assert any(item.get("id") == "core.architecture" for item in toc)


def test_evidence_validation_v2() -> None:
    # Create a tiny repo in a temp dir.
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "foo.py").write_text("class Plugin:\n    pass\n")
        model_v2 = {
            "summary": "x",
            "sections": {
                "core.architecture": {
                    "title": "Architecture",
                    "one_liner": "Design",
                    "claims": [
                        {
                            "kind": "architecture",
                            "statement": "Has Plugin class",
                            "evidence": [{"path": "foo.py", "quote": "class Plugin:"}],
                        }
                    ],
                }
            },
        }
        check = validate_evidence(tmp, model_v2)
        assert check.ok


#
# NOTE:
# We intentionally do NOT test `infer_repo_model_initial()` here because it calls a real LLM.
# Real LLM coverage lives in `tests/test_repo_memory.py`.

