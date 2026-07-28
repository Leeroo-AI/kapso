"""Repository-memory MCP tools expose the canonical Book."""

from __future__ import annotations

import asyncio
from pathlib import Path

from kapso.execution.memories.repo_memory import RepoMemoryManager
from kapso.gated_mcp.gates.base import GateConfig
from kapso.gated_mcp.gates.repo_memory_gate import RepoMemoryGate


def write_memory(
    repo_root: Path,
    *,
    summary: str,
    sections: dict,
) -> None:
    repo_root.mkdir()
    document = RepoMemoryManager.ensure_exists_in_worktree(str(repo_root))
    document["book"] = RepoMemoryManager._build_book_from_model(
        {"summary": summary, "sections": sections}
    )
    document["quality"]["section_count"] = len(document["book"]["sections"])
    document["quality"]["claim_count"] = (
        RepoMemoryManager._count_claims_in_book_sections(document["book"]["sections"])
    )
    RepoMemoryManager.write_to_worktree(str(repo_root), document)


def gate(repo_root: Path) -> RepoMemoryGate:
    return RepoMemoryGate(
        GateConfig(enabled=True, params={"repo_root": str(repo_root)})
    )


def test_repo_memory_gate_get_section(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    write_memory(
        repo_root,
        summary="Test repo",
        sections={
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
    )

    result = asyncio.run(
        gate(repo_root).handle_call(
            "get_repo_memory_section",
            {"section_id": "core.architecture"},
        )
    )
    assert "Architecture" in result[0].text
    assert "Has a README" in result[0].text


def test_repo_memory_gate_list_sections(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    write_memory(
        repo_root,
        summary="Test repo",
        sections={
            "core.architecture": {
                "title": "Architecture",
                "one_liner": "System design",
                "claims": [],
            },
            "core.gotchas": {
                "title": "Gotchas",
                "one_liner": "Common pitfalls",
                "claims": [],
            },
        },
    )

    result = asyncio.run(gate(repo_root).handle_call("list_repo_memory_sections", {}))
    assert "core.architecture" in result[0].text
    assert "core.gotchas" in result[0].text


def test_repo_memory_gate_get_summary(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    write_memory(
        repo_root,
        summary="This is a test repository for unit testing.",
        sections={
            "core.architecture": {
                "title": "Architecture",
                "one_liner": "Test",
                "claims": [],
            }
        },
    )

    result = asyncio.run(gate(repo_root).handle_call("get_repo_memory_summary", {}))
    assert "test repository" in result[0].text
    assert "core.architecture" in result[0].text
