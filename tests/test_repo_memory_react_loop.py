"""
RepoMemory ideation ReAct loop tests
===================================

We keep these tests lightweight and focused:
- Parsing robustness for the strict JSON protocol
- An integration-style smoke test that runs the loop on a tiny repo

NOTE: The integration test uses a real LLM call via `LLMBackend()` and costs money.
"""

from __future__ import annotations

import json
from pathlib import Path

import git
from dotenv import load_dotenv

load_dotenv()

from src.core.llm import LLMBackend
from src.execution.ideation.repo_memory_react import _extract_json_obj, ideate_solution_with_repo_memory_react


def test_extract_json_obj_strict():
    assert _extract_json_obj('{"action":"final","solution":"ok"}') == {
        "action": "final",
        "solution": "ok",
    }


def test_extract_json_obj_fallback_block():
    text = "noise\n\n{\"action\":\"get_section\",\"section_id\":\"core.architecture\"}\n\nnoise"
    assert _extract_json_obj(text) == {"action": "get_section", "section_id": "core.architecture"}


def test_ideation_react_loop_smoke(tmp_path: Path):
    # Minimal git repo with a committed RepoMemory file on main.
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    repo = git.Repo.init(repo_dir)

    (repo_dir / ".kapso").mkdir()
    doc = {
        "schema_version": 2,
        "generated_at": "2026-01-01T00:00:00Z",
        "book": {
            "summary": "Test repo for ideation loop",
            "toc": [{"id": "core.architecture", "title": "Architecture", "one_liner": "Test"}],
            "sections": {
                "core.architecture": {"title": "Architecture", "one_liner": "Test", "claims": []}
            },
        },
        "experiments": [],
    }
    (repo_dir / ".kapso" / "repo_memory.json").write_text(json.dumps(doc))
    repo.git.add([".kapso/repo_memory.json"])
    repo.git.commit("-m", "chore: add repo memory")

    llm = LLMBackend()
    solution, sections = ideate_solution_with_repo_memory_react(
        llm=llm,
        model="gpt-4.1-mini",
        repo=repo,
        base_branch="master" if "master" in [h.name for h in repo.heads] else "main",
        problem="Return a very short solution that says 'OK'.",
        output_requirements="Return {\"action\":\"final\",\"solution\":\"OK\"}. Do not call get_section.",
        max_rounds=2,
    )

    assert isinstance(solution, str)
    assert len(solution.strip()) > 0
    assert isinstance(sections, list)

