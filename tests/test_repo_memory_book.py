"""Contracts for the canonical Book-shaped repository memory."""

from __future__ import annotations

from pathlib import Path

import pytest

from kapso.execution.memories.repo_memory import RepoMemoryManager
from kapso.execution.memories.repo_memory.builders import validate_evidence


def book_document(
    *,
    summary: str = "Test repository",
    sections: dict | None = None,
) -> dict:
    book = RepoMemoryManager._build_book_from_model(
        {"summary": summary, "sections": sections or {}}
    )
    return {
        "schema_version": RepoMemoryManager.SCHEMA_VERSION,
        "generated_at": "2026-01-01T00:00:00Z",
        "repo_map": {},
        "book": book,
        "experiments": [],
        "quality": {
            "evidence_ok": True,
            "missing_evidence": [],
            "section_count": len(book["sections"]),
            "claim_count": RepoMemoryManager._count_claims_in_book_sections(
                book["sections"]
            ),
        },
    }


def test_superseded_flat_document_is_rejected() -> None:
    with pytest.raises(ValueError, match="fields are invalid"):
        RepoMemoryManager._require_document(
            {
                "schema_version": 1,
                "repo_model": {"summary": "old", "claims": []},
            }
        )


def test_render_summary_and_toc_is_bounded() -> None:
    output = RepoMemoryManager.render_summary_and_toc(
        book_document(),
        max_chars=120,
    )
    assert len(output) <= 120


def test_render_summary_and_toc_exposes_canonical_sections() -> None:
    output = RepoMemoryManager.render_summary_and_toc(
        book_document(),
        max_chars=2000,
    )
    assert "## Summary" in output
    assert "## Table of Contents" in output
    assert "core.architecture" in output


def test_get_section_renders_claims() -> None:
    document = book_document(
        sections={
            "core.architecture": {
                "title": "Architecture",
                "one_liner": "Design",
                "claims": [
                    {
                        "kind": "architecture",
                        "statement": "Uses plugins",
                        "evidence": [
                            {"path": "foo.py", "quote": "class Plugin"}
                        ],
                    }
                ],
            }
        }
    )
    output = RepoMemoryManager.get_section(
        document,
        "core.architecture",
        max_chars=2000,
    )
    assert "Uses plugins" in output
    assert "evidence:" in output


def test_get_section_reports_unknown_identifier() -> None:
    output = RepoMemoryManager.get_section(
        book_document(),
        "does.not.exist",
        max_chars=500,
    )
    assert "not found" in output.lower()


def test_list_sections_returns_canonical_toc() -> None:
    toc = RepoMemoryManager.list_sections(book_document())
    assert any(item["id"] == "core.architecture" for item in toc)


def test_evidence_validation_accepts_book_sections(tmp_path: Path) -> None:
    (tmp_path / "foo.py").write_text("class Plugin:\n    pass\n")
    model = {
        "summary": "x",
        "sections": {
            "core.architecture": {
                "title": "Architecture",
                "one_liner": "Design",
                "claims": [
                    {
                        "kind": "architecture",
                        "statement": "Has Plugin class",
                        "evidence": [
                            {"path": "foo.py", "quote": "class Plugin:"}
                        ],
                    }
                ],
            }
        },
    }
    assert validate_evidence(str(tmp_path), model).ok
