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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple


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
    repo_root = os.path.abspath(repo_root)
    file_paths: List[str] = []
    languages: Dict[str, int] = {}

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
    key_files = [p for p in key_file_candidates if os.path.exists(os.path.join(repo_root, p))]

    # Simple entrypoint heuristics (cheap and usually correct).
    entrypoint_names = {"main.py", "app.py", "server.py", "cli.py", "main.cpp", "main.cc", "main.c"}
    entrypoints = [p for p in file_paths if os.path.basename(p) in entrypoint_names]

    return {
        "repo_root": repo_root,
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


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace for fuzzy quote matching (preserves semantics)."""
    # Collapse all whitespace sequences to single space, strip ends
    return " ".join(text.split())


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
    return norm_quote in norm_text


def validate_evidence(repo_root: str, repo_model: Dict[str, Any]) -> EvidenceCheck:
    """
    Validate that every evidence quote exists in the referenced file.
    
    This is the primary guardrail that prevents "hallucinated repo memory".
    Matching is whitespace-tolerant (LLMs often add/remove spaces in quotes).
    """
    repo_root = os.path.abspath(repo_root)
    missing: List[str] = []

    claims = (repo_model or {}).get("claims", [])
    for idx, claim in enumerate(claims):
        for eidx, ev in enumerate(claim.get("evidence", []) or []):
            path = ev.get("path", "")
            quote = ev.get("quote", "")
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

    prompt = f"""You are inferring a repository's architecture and algorithms.

You will choose up to {max_files_to_read} files to read. Choose files that maximize:
- understanding of core algorithms/ideas
- entrypoints (how to run)
- configuration and evaluation contracts

You MUST return valid JSON only:
{{
  "files_to_read": [
    {{"path": "README.md", "why": "..." }}
  ]
}}

Repo key files: {key_files}
Repo entrypoints: {entrypoints}
Sample file list (paths): {files}
"""
    data = _extract_json(llm.llm_completion(model=model, messages=[{"role": "user", "content": prompt}]))
    chosen = []
    for item in data.get("files_to_read", [])[:max_files_to_read]:
        p = (item or {}).get("path", "")
        if p and p in repo_map.get("files", []):
            chosen.append(p)
    # Always include key files if present (cheap + high leverage)
    for p in key_files:
        if p in repo_map.get("files", []) and p not in chosen:
            chosen.insert(0, p)
    return chosen[:max_files_to_read]


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

    prompt = f"""You are inferring repository memory for an automated coding system.

Return ONLY valid JSON. Every claim MUST include evidence.

CRITICAL: Evidence quotes must be EXACT VERBATIM substrings that appear in the file.
- Copy quotes character-for-character from the file content below
- Do NOT paraphrase, summarize, or modify quotes in any way
- The quote must exist as a continuous substring in the file
- Shorter quotes (e.g., function signatures, class names) are safer than long ones

Required JSON schema:
{{
  "summary": "High-level what the repo does",
  "entrypoints": [{{"path": "main.py", "how_to_run": "python main.py --help"}}],
  "where_to_edit": [{{"path": "src/foo.py", "role": "core algorithm implementation"}}],
  "claims": [
    {{
      "kind": "algorithm|architecture|contract|deployment|other",
      "statement": "...",
      "confidence": 0.0,
      "evidence": [{{"path": "path/in/repo.py", "quote": "EXACT verbatim substring from file - copy carefully"}}]
    }}
  ]
}}

RepoMap key files: {repo_map.get("key_files", [])}
RepoMap entrypoints: {repo_map.get("entrypoints", [])}

FILE CONTENTS (authoritative - copy quotes EXACTLY from here):
{files_payload}
"""
    repo_model = _extract_json(llm.llm_completion(model=model, messages=[{"role": "user", "content": prompt}]))
    return repo_model


def infer_repo_model_with_retry(
    llm: LLMLike,
    model: str,
    repo_root: str,
    repo_map: Dict[str, Any],
    max_file_chars: int = 20000,
    max_files_to_read: int = 20,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Build a semantic repo model with retry on evidence validation failure.
    
    If evidence validation fails, retries with explicit feedback about which
    quotes were invalid, giving the LLM a chance to correct them.
    """
    repo_model = infer_repo_model_initial(
        llm=llm,
        model=model,
        repo_root=repo_root,
        repo_map=repo_map,
        max_file_chars=max_file_chars,
        max_files_to_read=max_files_to_read,
    )
    
    check = validate_evidence(repo_root, repo_model)
    if check.ok:
        return repo_model
    
    # Retry with feedback about invalid quotes
    for attempt in range(max_retries):
        files_to_read = plan_files_to_read(llm, model=model, repo_map=repo_map, max_files_to_read=max_files_to_read)
        file_blobs: List[Tuple[str, str]] = []
        for rel in files_to_read:
            abs_path = os.path.join(repo_root, rel)
            file_blobs.append((rel, _safe_read_text(abs_path, max_chars=max_file_chars)))
        files_payload = "\n\n".join([f"=== FILE: {p} ===\n{t}" for p, t in file_blobs if t])
        
        error_feedback = "\n".join(f"- {err}" for err in check.missing[:10])
        
        retry_prompt = f"""Your previous response had evidence quotes that don't exist in the files.

ERRORS (quotes not found):
{error_feedback}

Please regenerate the repo model with CORRECT quotes.
- Quotes must be EXACT VERBATIM substrings from the file content
- Copy character-for-character, do NOT paraphrase
- Use shorter quotes (function/class names) if unsure

Your previous model (fix the evidence):
{json.dumps(repo_model, indent=2)[:15000]}

FILE CONTENTS (copy quotes EXACTLY from here):
{files_payload}

Return ONLY valid JSON with the same schema, but with corrected evidence quotes.
"""
        repo_model = _extract_json(llm.llm_completion(model=model, messages=[{"role": "user", "content": retry_prompt}]))
        check = validate_evidence(repo_root, repo_model)
        if check.ok:
            return repo_model
    
    # After all retries, return what we have (caller will validate and may raise)
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

    prompt = f"""You are updating repository memory after code changes.

Return ONLY valid JSON and keep it evidence-backed:
- Preserve previous claims if still supported.
- Update/add/remove claims as needed based on the diff and changed files.
- Every NEW or MODIFIED claim MUST include evidence quotes from the provided changed files.

Schema is identical to initial inference:
{{
  "summary": "...",
  "entrypoints": [...],
  "where_to_edit": [...],
  "claims": [{{"kind": "...", "statement": "...", "confidence": 0.0, "evidence": [...]}}]
}}

DIFF SUMMARY:
{diff_summary}

PREVIOUS MODEL (may contain stale items):
{json.dumps(previous_model, indent=2)[:20000]}

CHANGED FILE CONTENTS (authoritative for new/updated claims):
{changed_payload}
"""
    return _extract_json(llm.llm_completion(model=model, messages=[{"role": "user", "content": prompt}]))

