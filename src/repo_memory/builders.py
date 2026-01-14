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

from src.core.prompt_loader import load_prompt, render_prompt


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
    ".praxium",  # RepoMemory lives here; exclude it from "what does the repo do?"
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
    #   not transient/ignored artifacts like `sessions/*` or `.praxium/*`.
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
            # - ".praxium/repo_memory.json" would become "praxium/repo_memory.json"
            # That breaks portability and also defeats our `.praxium` exclusion filter.
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


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace for fuzzy quote matching (preserves semantics)."""
    # Collapse all whitespace sequences to single space, strip ends
    return " ".join(text.split())


def _normalize_punct_whitespace(text: str) -> str:
    """
    Normalize whitespace and also remove whitespace around punctuation.

    Why:
    - LLMs sometimes "flatten" multi-line code into a single-line quote, e.g.:
        code:  lora_config = LoraConfig(
                 r=8,
                 lora_alpha=8,
               )
        quote: lora_config = LoraConfig(r=8, lora_alpha=8,
      This is still grounded, but fails strict substring matching.
    - We want to tolerate *whitespace-only* differences around punctuation ((), commas, =, etc.)
      without allowing the model to invent new tokens.
    """
    raw = _normalize_whitespace(text or "")
    # Remove whitespace around common punctuation tokens used in code/config.
    # Keep this conservative: only punctuation, not alphanumeric boundaries.
    return re.sub(r"\s*([()\[\]{}.,:;=])\s*", r"\1", raw)


def _strip_markdown_backticks(text: str) -> str:
    """
    Remove Markdown inline-code backticks.
    
    Why:
    - Repos often wrap identifiers in backticks in README/docs.
    - LLMs sometimes omit backticks when copying evidence quotes.
    - Allowing a backtick-stripped fallback keeps evidence grounded while
      avoiding flaky failures on purely presentational characters.
    """
    return (text or "").replace("`", "")


def _quote_in_text(quote: str, text: str) -> bool:
    """
    Check if quote appears in text, with whitespace-tolerant matching.
    
    First tries exact match, then tries normalized whitespace match.
    This handles LLMs that slightly modify spacing in quotes.
    """
    # Exact match (preferred)
    if quote in text:
        return True
    
    # Whitespace-normalized match (fallback)
    norm_quote = _normalize_whitespace(quote)
    norm_text = _normalize_whitespace(text)
    if norm_quote and norm_quote in norm_text:
        return True

    # Markdown backtick-stripped match (fallback)
    # Keep this narrow: only strip backticks (common LLM mismatch) and still
    # require substring containment after whitespace normalization.
    bt_quote = _normalize_whitespace(_strip_markdown_backticks(quote))
    bt_text = _normalize_whitespace(_strip_markdown_backticks(text))

    if bt_quote and bt_quote in bt_text:
        return True

    # Punctuation/whitespace normalized match (fallback)
    #
    # This specifically addresses cases where the model collapses newlines around punctuation,
    # e.g. "LoraConfig(r=8, lora_alpha=8," when the file is formatted as multi-line args.
    pw_quote = _normalize_punct_whitespace(_strip_markdown_backticks(quote))
    pw_text = _normalize_punct_whitespace(_strip_markdown_backticks(text))
    return bool(pw_quote) and pw_quote in pw_text


def validate_evidence(repo_root: str, repo_model: Dict[str, Any]) -> EvidenceCheck:
    """
    Validate that every evidence quote exists in the referenced file.
    
    This is the primary guardrail that prevents "hallucinated repo memory".
    Matching is whitespace-tolerant (LLMs often add/remove spaces in quotes).
    """
    repo_root = os.path.abspath(repo_root)
    missing: List[str] = []

    # v2 (book-model) shape: {"summary": "...", "sections": {section_id: {"claims": [...]}}}
    if isinstance((repo_model or {}).get("sections"), dict):
        sections = (repo_model or {}).get("sections", {}) or {}
        for sid, sec in sections.items():
            claims = (sec or {}).get("claims", []) or []
            for idx, claim in enumerate(claims):
                for eidx, ev in enumerate((claim or {}).get("evidence", []) or []):
                    path = (ev or {}).get("path", "")
                    quote = (ev or {}).get("quote", "")
                    if not path or not quote:
                        missing.append(f"{sid}.claims[{idx}].evidence[{eidx}]: missing path/quote")
                        continue
                    abs_path = os.path.join(repo_root, path)
                    if not os.path.exists(abs_path):
                        missing.append(f"{sid}.claims[{idx}].evidence[{eidx}]: file not found: {path}")
                        continue
                    try:
                        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                            text = f.read()
                    except Exception:
                        missing.append(f"{sid}.claims[{idx}].evidence[{eidx}]: unreadable file: {path}")
                        continue
                    if not _quote_in_text(quote, text):
                        missing.append(f"{sid}.claims[{idx}].evidence[{eidx}]: quote not found in {path}")

        return EvidenceCheck(ok=len(missing) == 0, missing=missing)

    # v1 (legacy) shape: {"claims": [...]}
    claims = (repo_model or {}).get("claims", [])
    for idx, claim in enumerate(claims):
        for eidx, ev in enumerate((claim or {}).get("evidence", []) or []):
            path = (ev or {}).get("path", "")
            quote = (ev or {}).get("quote", "")
            if not path or not quote:
                missing.append(f"claims[{idx}].evidence[{eidx}]: missing path/quote")
                continue
            abs_path = os.path.join(repo_root, path)
            if not os.path.exists(abs_path):
                missing.append(f"claims[{idx}].evidence[{eidx}]: file not found: {path}")
                continue
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
            except Exception:
                missing.append(f"claims[{idx}].evidence[{eidx}]: unreadable file: {path}")
                continue
            if not _quote_in_text(quote, text):
                missing.append(f"claims[{idx}].evidence[{eidx}]: quote not found in {path}")

    return EvidenceCheck(ok=len(missing) == 0, missing=missing)


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

    template = load_prompt("repo_memory/prompts/plan_files_to_read.md")
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

    files_payload = "\n\n".join([f"=== FILE: {p} ===\n{t}" for p, t in file_blobs if t])

    template = load_prompt("repo_memory/prompts/infer_repo_model_initial.md")
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
    max_retries: int = 4,
) -> Dict[str, Any]:
    """
    Build a semantic repo model with retry on evidence validation failure.
    
    If evidence validation fails, retries with explicit feedback about which
    quotes were invalid, giving the LLM a chance to correct them.
    """
    # Plan + read files ONCE and reuse the same file payload for retries.
    #
    # Why:
    # - Evidence failures are usually about mis-copied quotes/paths, not missing files.
    # - Re-planning on each retry can remove the relevant file from the prompt, making it
    #   impossible for the model to fix its own evidence.
    files_to_read = plan_files_to_read(
        llm, model=model, repo_map=repo_map, max_files_to_read=max_files_to_read
    )
    file_blobs: List[Tuple[str, str]] = []
    for rel in files_to_read:
        abs_path = os.path.join(repo_root, rel)
        file_blobs.append((rel, _safe_read_text(abs_path, max_chars=max_file_chars)))
    files_payload = "\n\n".join([f"=== FILE: {p} ===\n{t}" for p, t in file_blobs if t])

    # Initial inference.
    template = load_prompt("repo_memory/prompts/infer_repo_model_initial.md")
    prompt = render_prompt(
        template,
        {
            "repo_map_key_files_json": json.dumps(repo_map.get("key_files", [])),
            "repo_map_entrypoints_json": json.dumps(repo_map.get("entrypoints", [])),
            "files_payload": files_payload,
        },
    )
    repo_model = _extract_json(
        llm.llm_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
    )

    check = validate_evidence(repo_root, repo_model)
    if check.ok:
        return repo_model

    # Retry with feedback about invalid quotes.
    template = load_prompt("repo_memory/prompts/infer_repo_model_retry.md")
    for _ in range(max_retries):
        error_feedback = "\n".join(f"- {err}" for err in check.missing[:10])
        retry_prompt = render_prompt(
            template,
            {
                "error_feedback": error_feedback,
                "previous_model_json": json.dumps(repo_model, indent=2)[:15000],
                "files_payload": files_payload,
            },
        )
        repo_model = _extract_json(
            llm.llm_completion(
                model=model,
                messages=[{"role": "user", "content": retry_prompt}],
                temperature=0,
            )
        )
        check = validate_evidence(repo_root, repo_model)
        if check.ok:
            return repo_model

    # After all retries, return what we have (caller will validate and may raise).
    return repo_model


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
    
    The updated model must remain evidence-backed.
    """
    changed_blobs: List[Tuple[str, str]] = []
    for rel in changed_files[:20]:
        abs_path = os.path.join(repo_root, rel)
        if os.path.exists(abs_path):
            changed_blobs.append((rel, _safe_read_text(abs_path, max_chars=max_file_chars)))
    changed_payload = "\n\n".join([f"=== FILE: {p} ===\n{t}" for p, t in changed_blobs if t])

    template = load_prompt("repo_memory/prompts/infer_repo_model_update.md")
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

