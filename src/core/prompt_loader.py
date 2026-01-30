"""
Prompt loader (tunable, minimal)
===============================

We externalize large prompt strings into plain text files so they are:
- easy to audit and tune without touching python logic
- reusable across different call sites (builders, ideation, coding)
- testable (we can assert prompt files exist and are loaded)

Override mechanism
------------------
By default, prompts are loaded from the repository under `src/<relative_path>`.

If you set `KAPSO_PROMPTS_DIR=/some/dir`, we will load prompts from:
  /some/dir/<relative_path>

This lets you tune prompts in a separate directory (e.g. mounted volume) without
editing or forking the codebase.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict


_DEFAULT_PROMPTS_ROOT = Path(__file__).resolve().parents[1]  # .../src


def _get_prompts_root() -> Path:
    """Return the root directory where prompt files are loaded from."""
    override = os.environ.get("KAPSO_PROMPTS_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return _DEFAULT_PROMPTS_ROOT


@lru_cache(maxsize=256)
def load_prompt(relative_path: str) -> str:
    """
    Load a prompt file as text.

    Args:
        relative_path: Path relative to `src/`, e.g. "execution/memories/repo_memory/prompts/infer_repo_model_initial.md"
    """
    rel = (relative_path or "").lstrip("/").strip()
    if not rel:
        raise ValueError("load_prompt(relative_path) requires a non-empty path")

    path = _get_prompts_root() / rel
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def render_prompt(template: str, variables: Dict[str, str]) -> str:
    """
    Render a prompt template using a tiny `{{var}}` replacement scheme.

    Why not `str.format`?
    - Our prompts frequently include JSON schemas with `{}` which would require
      escaping and is extremely error-prone.

    This replacement is intentionally simple and explicit.
    """
    text = template or ""
    for key, value in (variables or {}).items():
        text = text.replace("{{" + str(key) + "}}", str(value))
    return text

