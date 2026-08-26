"""RepoMemory Book (the one schema) — hermetic pins.

The v1 flat `repo_model` and its migration shim were deleted (stale-code
audit 2026-08-26, B9; Rule 7: pre-release formats are not migrated).
These tests pin what replaced them: v1 documents are REJECTED with a
clear story, the Book normalizes to core shells + a stable TOC, and the
builder validator accepts only the sections shape.
"""

from __future__ import annotations

import pytest

from kapso.execution.memories.repo_memory import RepoMemoryManager
from kapso.execution.memories.repo_memory.builders import (
    RepoMemoryResponseError,
    _validate_repo_model,
)


def v2_doc(summary: str = "x", sections: dict | None = None) -> dict:
    return {
        "schema_version": 2,
        "book": {"summary": summary, "sections": sections or {}},
    }


def test_v1_document_is_rejected_with_a_clear_story() -> None:
    v1 = {"schema_version": 1, "repo_model": {"summary": "x", "claims": []}}
    with pytest.raises(ValueError, match="not migrated"):
        RepoMemoryManager._require_v2(v1)


def test_v2_without_a_book_is_rejected() -> None:
    with pytest.raises(ValueError, match="no 'book'"):
        RepoMemoryManager._require_v2({"schema_version": 2})


def test_require_v2_normalizes_core_shells_and_toc() -> None:
    doc = RepoMemoryManager._require_v2(v2_doc())
    sections = doc["book"]["sections"]
    for section_id in RepoMemoryManager.CORE_SECTIONS:
        assert section_id in sections
    assert doc["book"]["toc"]
    assert doc["quality"]["section_count"] == len(sections)
    assert doc["quality"]["claim_count"] == 0


def test_render_summary_and_toc_bounded_and_formatted() -> None:
    doc = v2_doc()
    out = RepoMemoryManager.render_summary_and_toc(doc, max_chars=120)
    assert len(out) <= 120
    out = RepoMemoryManager.render_summary_and_toc(v2_doc(), max_chars=2000)
    assert "## Summary" in out
    assert "## Table of Contents" in out
    assert "core.architecture" in out


def test_get_section_found_renders_claims() -> None:
    doc = v2_doc(sections={
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
    })
    out = RepoMemoryManager.get_section(doc, "core.architecture", max_chars=2000)
    assert "Uses plugins" in out
    assert "evidence:" in out


def test_get_section_not_found_and_list_sections() -> None:
    out = RepoMemoryManager.get_section(v2_doc(), "does.not.exist", max_chars=500)
    assert "not found" in out.lower()
    toc = RepoMemoryManager.list_sections(v2_doc())
    assert any(item.get("id") == "core.architecture" for item in toc)


def test_bootstrap_skeleton_is_book_only(tmp_path) -> None:
    doc = RepoMemoryManager.ensure_exists_in_worktree(str(tmp_path))
    assert "repo_model" not in doc
    assert doc["schema_version"] == 2
    assert set(doc["book"]["sections"]) >= set(RepoMemoryManager.CORE_SECTIONS)
    # And the persisted file round-trips through the strict loader.
    loaded = RepoMemoryManager.load_from_worktree(str(tmp_path))
    assert "repo_model" not in loaded
    assert loaded["book"]["toc"]


def test_builder_validator_accepts_only_the_sections_shape() -> None:
    valid = {"summary": "s", "sections": {"core.architecture": {}}}
    assert _validate_repo_model(valid) is valid
    # The old dual-format sniff accepted the flat v1 fields; now a
    # sections-less response is a malformed response to REPAIR.
    with pytest.raises(RepoMemoryResponseError, match="'sections'"):
        _validate_repo_model({
            "summary": "s", "entrypoints": [], "where_to_edit": [],
            "claims": [],
        })
