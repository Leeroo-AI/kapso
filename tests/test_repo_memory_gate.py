"""
RepoMemory MCP Gate tests
=========================

Tests the RepoMemoryGate MCP tools for accessing .kapso/repo_memory.json.

These tests validate:
- The gate can list sections
- The gate can get section content
- The gate can get summary + TOC
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest


def test_repo_memory_gate_get_section(tmp_path: Path):
    """Test get_repo_memory_section tool."""
    from kapso.gated_mcp.gates.repo_memory_gate import RepoMemoryGate
    from kapso.gated_mcp.gates.base import GateConfig
    
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

    config = GateConfig(enabled=True, params={"repo_root": str(repo_root)})
    gate = RepoMemoryGate(config)
    
    async def run_test():
        result = await gate.handle_call("get_repo_memory_section", {"section_id": "core.architecture"})
        return result[0].text
    
    text = asyncio.run(run_test())
    assert "Architecture" in text
    assert "Has a README" in text


def test_repo_memory_gate_list_sections(tmp_path: Path):
    """Test list_repo_memory_sections tool."""
    from kapso.gated_mcp.gates.repo_memory_gate import RepoMemoryGate
    from kapso.gated_mcp.gates.base import GateConfig
    
    repo_root = tmp_path / "repo"
    kapso_dir = repo_root / ".kapso"
    kapso_dir.mkdir(parents=True)

    doc = {
        "schema_version": 2,
        "generated_at": "2026-01-01T00:00:00Z",
        "book": {
            "summary": "Test repo",
            "toc": [
                {"id": "core.architecture", "title": "Architecture", "one_liner": "System design"},
                {"id": "core.gotchas", "title": "Gotchas", "one_liner": "Common pitfalls"},
            ],
            "sections": {
                "core.architecture": {"title": "Architecture", "one_liner": "System design", "claims": []},
                "core.gotchas": {"title": "Gotchas", "one_liner": "Common pitfalls", "claims": []},
            },
        },
    }
    (kapso_dir / "repo_memory.json").write_text(json.dumps(doc))

    config = GateConfig(enabled=True, params={"repo_root": str(repo_root)})
    gate = RepoMemoryGate(config)
    
    async def run_test():
        result = await gate.handle_call("list_repo_memory_sections", {})
        return result[0].text
    
    text = asyncio.run(run_test())
    assert "core.architecture" in text
    assert "core.gotchas" in text


def test_repo_memory_gate_get_summary(tmp_path: Path):
    """Test get_repo_memory_summary tool."""
    from kapso.gated_mcp.gates.repo_memory_gate import RepoMemoryGate
    from kapso.gated_mcp.gates.base import GateConfig
    
    repo_root = tmp_path / "repo"
    kapso_dir = repo_root / ".kapso"
    kapso_dir.mkdir(parents=True)

    doc = {
        "schema_version": 2,
        "generated_at": "2026-01-01T00:00:00Z",
        "book": {
            "summary": "This is a test repository for unit testing.",
            "toc": [{"id": "core.architecture", "title": "Architecture", "one_liner": "Test"}],
            "sections": {
                "core.architecture": {"title": "Architecture", "one_liner": "Test", "claims": []},
            },
        },
    }
    (kapso_dir / "repo_memory.json").write_text(json.dumps(doc))

    config = GateConfig(enabled=True, params={"repo_root": str(repo_root)})
    gate = RepoMemoryGate(config)
    
    async def run_test():
        result = await gate.handle_call("get_repo_memory_summary", {})
        return result[0].text
    
    text = asyncio.run(run_test())
    assert "test repository" in text
    assert "core.architecture" in text
