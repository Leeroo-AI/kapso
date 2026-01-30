"""
RepoMemory builders (deterministic + agentic)
============================================

This file contains:
- A deterministic `RepoMap` builder (file tree, key files, entrypoints).
- An agentic `RepoModel` inference workflow using an injected LLM.
- Evidence validation utilities (quotes must exist in repo files).

Design notes:
- We keep the deterministic map always-on because it is cheap and reliable.
- The semantic model is best-effort and must be evidence-backed.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

from kapso.core.prompt_loader import load_prompt, render_prompt


class LLMLike(Protocol):
    """Minimal interface we need for repo-model inference (enables deterministic testing)."""

    def llm_completion(self, model: str, messages: List[Dict[str, str]], **kwargs) -> str: ...


_IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    ".kapso",  # RepoMemory lives here; exclude it from "what does the repo do?"
    "sessions",  # ExperimentWorkspace nested clones
}


def _safe_read_text(path: str, max_chars: int) -> str:
    """Read a file as text and cap size to keep prompts bounded."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception:
        return ""

    if len(text) <= max_chars:
        return text

    # Keep head+tail so evidence quotes are likely to remain present.
    head = text[: max_chars // 2]
    tail = text[-(max_chars // 2) :]
    return head + "\n\n... (truncated) ...\n\n" + tail


def build_repo_map(
    repo_root: str,
    max_files: int = 5000,
    max_depth: int = 12,
) -> Dict[str, Any]:
    """
    Deterministically summarize the repository structure.
    
    This is used both as:
    - a stable "repo map" for coding agents, and
    - an input to the agentic repo-model inference workflow.
    """
    # IMPORTANT:
    # - RepoMemory is committed into git branches and should be portable across machines/runs.
    # - Experiments run inside a temporary clone (`sessions/<branch>`), which is deleted on close.
    #   This means persisting absolute paths like `/tmp/.../sessions/...` into memory is wrong.
    #
    # Therefore:
    # - We store `repo_map["repo_root"]` as "." (portable, stable).
    # - When possible, we enumerate files via git (so the RepoMap matches what is committed and
    #   respects `.gitignore`). We still keep a filesystem `os.walk` fallback for non-git dirs.
    repo_root = os.path.abspath(repo_root)
    file_paths: List[str] = []
    languages: Dict[str, int] = {}

    # Preferred enumeration path: ask git for the file list.
    #
    # Why:
    # - RepoMap must reflect the actual repo state (tracked + untracked-not-ignored),
    #   not transient/ignored artifacts like `sessions/*` or `.kapso/*`.
    # - This also avoids phantom files such as `changes.log` when it is ignored.
    #
    # We still explicitly filter out `changes.log` because it is observability metadata,
    # not part of "what does the repo do?".
    git_files: Optional[List[str]] = None
    try:
        out = subprocess.check_output(
            ["git", "-C", repo_root, "ls-files", "--cached", "--others", "--exclude-standard"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", "replace")
        git_files = [ln.strip() for ln in out.splitlines() if ln.strip()]
    except Exception:
        git_files = None

    if git_files is not None:
        for rel_path in git_files:
            if len(file_paths) >= max_files:
                break

            # git always uses "/" separators; keep paths in that canonical form.
            #
            # IMPORTANT: Do NOT use `lstrip("./")` here.
            # `lstrip` removes *any* leading '.' characters, which corrupts dotfiles:
            # - ".gitignore" would become "gitignore"
            # - ".kapso/repo_memory.json" would become "kapso/repo_memory.json"
            # That breaks portability and also defeats our `.kapso` exclusion filter.
            if rel_path.startswith("./"):
                rel_path = rel_path[2:]
            if not rel_path:
                continue

            # Explicitly exclude observability metadata from the semantic repo map.
            if rel_path == "changes.log":
                continue

            # Exclude meta directories (RepoMemory itself, sessions, VCS dirs, etc.).
            top_level = rel_path.split("/", 1)[0]
            if top_level in _IGNORE_DIRS:
                continue

            # Keep behavior consistent with the filesystem traversal by enforcing a max depth.
            # Depth is measured as number of path separators.
            if rel_path.count("/") > max_depth:
                continue

            file_paths.append(rel_path)

            _, ext = os.path.splitext(rel_path)
            if ext:
                languages[ext.lower()] = languages.get(ext.lower(), 0) + 1
    else:
        # Fallback enumeration path: walk the filesystem.
        # This is used for non-git directories (rare in the experimentation engine).
        root_depth = repo_root.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, filenames in os.walk(repo_root):
            # Prune ignored dirs in-place to prevent descending into them.
            dirnames[:] = [d for d in dirnames if d not in _IGNORE_DIRS]

            depth = dirpath.rstrip(os.sep).count(os.sep) - root_depth
            if depth > max_depth:
                dirnames[:] = []
                continue

            for fname in filenames:
                if len(file_paths) >= max_files:
                    break
                abs_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(abs_path, repo_root)

                # Explicitly exclude observability metadata from the semantic repo map.
                if rel_path == "changes.log":
                    continue

                file_paths.append(rel_path)

                _, ext = os.path.splitext(fname)
                if ext:
                    languages[ext.lower()] = languages.get(ext.lower(), 0) + 1

    file_paths.sort()

    # "Key files" are cheap anchors that often describe how the repo works.
    key_file_candidates = [
        "README.md",
        "README.rst",
        "README.txt",
        "pyproject.toml",
        "requirements.txt",
        "setup.py",
        "package.json",
        "Dockerfile",
        "Makefile",
    ]
    # Derive key files from the enumerated file list (portable, avoids filesystem drift).
    key_file_set = set(file_paths)
    key_files = [p for p in key_file_candidates if p in key_file_set]

    # Simple entrypoint heuristics (cheap and usually correct).
    entrypoint_names = {"main.py", "app.py", "server.py", "cli.py", "main.cpp", "main.cc", "main.c"}
    entrypoints = [p for p in file_paths if os.path.basename(p) in entrypoint_names]

    return {
        # Keep this portable: repo roots are often under /tmp in E2E tests.
        "repo_root": ".",
        "file_count": len(file_paths),
        "files": file_paths[:2000],  # Keep bounded in memory file.
        "languages_by_extension": dict(sorted(languages.items(), key=lambda kv: kv[1], reverse=True)),
        "key_files": key_files,
        "entrypoints": entrypoints[:50],
    }


@dataclass
class EvidenceCheck:
    ok: bool
    missing: List[str]


def _build_toc_from_sections(sections: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Build a simple Table of Contents from a v2 `sections` dict.
    
    This is intentionally lightweight and deterministic:
    - Sort by section id for stability
    - Include only id/title/one_liner (no content)
    """
    sections = sections or {}
    toc: List[Dict[str, str]] = []
    for sid in sorted(sections.keys()):
        sec = sections.get(sid, {}) or {}
        toc.append(
            {
                "id": sid,
                "title": (sec.get("title") or sid),
                "one_liner": (sec.get("one_liner") or ""),
            }
        )
    return toc


def _sections_to_flat_claims(sections: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flatten v2 sections -> a single list of claims (legacy compatibility helper).
    
    This is useful when a consumer still expects a v1-style `repo_model.claims[]`.
    """
    sections = sections or {}
    flat: List[Dict[str, Any]] = []
    for sec in sections.values():
        for claim in (sec or {}).get("claims", []) or []:
            if isinstance(claim, dict):
                flat.append(claim)
    return flat


def _add_line_numbers(text: str) -> str:
    """
    Add line numbers to file content for evidence referencing.
    
    Format: "  N| content" where N is right-aligned line number.
    This allows the LLM to reference specific lines by number instead of
    requiring exact verbatim quotes.
    """
    if not text:
        return ""
    lines = text.split('\n')
    width = len(str(len(lines)))
    numbered = []
    for i, line in enumerate(lines, 1):
        numbered.append(f"{i:>{width}}| {line}")
    return '\n'.join(numbered)


def _format_file_payload(file_blobs: List[Tuple[str, str]]) -> str:
    """
    Format file blobs into a payload with line numbers.
    
    Each file is formatted as:
        === FILE: path/to/file.py ===
          1| first line
          2| second line
          ...
    """
    parts = []
    for path, content in file_blobs:
        if content:
            numbered_content = _add_line_numbers(content)
            parts.append(f"=== FILE: {path} ===\n{numbered_content}")
    return "\n\n".join(parts)


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Parse JSON robustly.
    
    We intentionally keep this simple:
    - prefer full parse
    - otherwise try the first {...} block
    """
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError("LLM did not return JSON")
    return json.loads(m.group(0))


def plan_files_to_read(
    llm: LLMLike,
    model: str,
    repo_map: Dict[str, Any],
    max_files_to_read: int = 20,
) -> List[str]:
    """Ask the LLM which files to read to infer RepoModel (agentic file selection)."""
    files = repo_map.get("files", [])[:500]
    key_files = repo_map.get("key_files", [])
    entrypoints = repo_map.get("entrypoints", [])

    template = load_prompt("execution/memories/repo_memory/prompts/plan_files_to_read.md")
    prompt = render_prompt(
        template,
        {
            "max_files_to_read": str(max_files_to_read),
            "key_files_json": json.dumps(key_files),
            "entrypoints_json": json.dumps(entrypoints),
            "files_json": json.dumps(files),
        },
    )
    # Deterministic planning: this output is structural JSON, not creative writing.
    data = _extract_json(
        llm.llm_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
    )
    chosen = []
    for item in data.get("files_to_read", [])[:max_files_to_read]:
        p = (item or {}).get("path", "")
        if p and p in repo_map.get("files", []):
            chosen.append(p)

    # Always include key files + entrypoints if present (cheap + high leverage).
    #
    # Why:
    # - Key files (README, pyproject, requirements, etc.) often contain the "story" of the repo.
    # - Entrypoints show the real runtime data flow and output contracts.
    # - Making this deterministic improves semantic memory quality and reduces dependence
    #   on the LLM planner picking the obvious files.
    must_include: List[str] = []
    for p in list(key_files or []) + list(entrypoints or []):
        if p and p in repo_map.get("files", []) and p not in must_include:
            must_include.append(p)

    # Preserve the LLM's ordering for the rest (it often clusters related modules).
    for p in chosen:
        if p not in must_include:
            must_include.append(p)

    return must_include[:max_files_to_read]


def infer_repo_model_initial(
    llm: LLMLike,
    model: str,
    repo_root: str,
    repo_map: Dict[str, Any],
    max_file_chars: int = 20000,
    max_files_to_read: int = 20,
) -> Dict[str, Any]:
    """
    Build a semantic repo model from scratch using agentic file selection.
    
    Output is JSON with evidence-backed claims.
    """
    files_to_read = plan_files_to_read(llm, model=model, repo_map=repo_map, max_files_to_read=max_files_to_read)
    file_blobs: List[Tuple[str, str]] = []
    for rel in files_to_read:
        abs_path = os.path.join(repo_root, rel)
        file_blobs.append((rel, _safe_read_text(abs_path, max_chars=max_file_chars)))

    files_payload = _format_file_payload(file_blobs)

    template = load_prompt("execution/memories/repo_memory/prompts/infer_repo_model_initial.md")
    prompt = render_prompt(
        template,
        {
            "repo_map_key_files_json": json.dumps(repo_map.get("key_files", [])),
            "repo_map_entrypoints_json": json.dumps(repo_map.get("entrypoints", [])),
            "files_payload": files_payload,
        },
    )
    # Deterministic: evidence quotes must be exact; reduce randomness to avoid drift.
    repo_model = _extract_json(
        llm.llm_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
    )
    return repo_model


def infer_repo_model_with_retry(
    llm: LLMLike,
    model: str,
    repo_root: str,
    repo_map: Dict[str, Any],
    max_file_chars: int = 20000,
    max_files_to_read: int = 20,
    max_retries: int = 4,  # Kept for API compatibility, but not used
) -> Dict[str, Any]:
    """
    Build a semantic repo model.
    
    Note: With line-number-based evidence, validation and retry are no longer needed.
    This function now simply delegates to infer_repo_model_initial.
    The max_retries parameter is kept for API compatibility but is not used.
    """
    return infer_repo_model_initial(
        llm=llm,
        model=model,
        repo_root=repo_root,
        repo_map=repo_map,
        max_file_chars=max_file_chars,
        max_files_to_read=max_files_to_read,
    )


def infer_repo_model_update(
    llm: LLMLike,
    model: str,
    repo_root: str,
    repo_map: Dict[str, Any],
    previous_model: Dict[str, Any],
    diff_summary: str,
    changed_files: List[str],
    max_file_chars: int = 15000,
) -> Dict[str, Any]:
    """
    Incrementally update RepoModel using the previous model + diffs + changed files.
    
    The updated model uses line-number-based evidence references.
    """
    changed_blobs: List[Tuple[str, str]] = []
    for rel in changed_files[:20]:
        abs_path = os.path.join(repo_root, rel)
        if os.path.exists(abs_path):
            changed_blobs.append((rel, _safe_read_text(abs_path, max_chars=max_file_chars)))
    changed_payload = _format_file_payload(changed_blobs)

    template = load_prompt("execution/memories/repo_memory/prompts/infer_repo_model_update.md")
    prompt = render_prompt(
        template,
        {
            "diff_summary": diff_summary,
            "previous_model_json": json.dumps(previous_model, indent=2)[:20000],
            "changed_payload": changed_payload,
        },
    )
    return _extract_json(
        llm.llm_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
    )

