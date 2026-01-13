"""
Prompt externalization smoke tests
=================================

These tests ensure that the prompt templates we rely on are:
- present on disk
- loadable via `src.core.prompt_loader.load_prompt`

This is important because prompt tuning is a first-class requirement for RepoMemory.
"""

from __future__ import annotations

from src.core import prompt_loader


PROMPT_PATHS = [
    # RepoMemory builders
    "repo_memory/prompts/plan_files_to_read.md",
    "repo_memory/prompts/infer_repo_model_initial.md",
    "repo_memory/prompts/infer_repo_model_retry.md",
    "repo_memory/prompts/infer_repo_model_update.md",
    # Execution prompts (coding + ideation)
    "execution/prompts/coding_agent_implement.md",
    "execution/prompts/coding_agent_debug.md",
    "execution/prompts/ideation_solution_react.md",
]


def test_prompt_files_exist_and_load():
    # Clear cache to ensure we actually read from disk in this test.
    prompt_loader.load_prompt.cache_clear()

    for path in PROMPT_PATHS:
        text = prompt_loader.load_prompt(path)
        assert isinstance(text, str)
        assert len(text.strip()) > 20, f"Prompt seems empty: {path}"

