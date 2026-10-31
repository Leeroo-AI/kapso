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
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, Tuple

from kapso.core.config import load_config
from kapso.core.prompt_loader import load_prompt, render_prompt

# Changed-file text one update call may carry; a run's own config
# overrides it through the strategy params.
UPDATE_PAYLOAD_MAX_CHARS = load_config()["search_strategy"]["params"][
    "repo_memory_update_payload_max_chars"
]
_FILE_SEPARATOR = "\n\n"


class LLMLike(Protocol):
    """Minimal interface we need for repo-model inference (enables deterministic testing)."""

    def llm_completion(self, messages: List[Dict[str, str]], **kwargs) -> str: ...


class RepoMemoryResponseError(ValueError):
    """Raised when an LLM response is not valid RepoMemory JSON."""


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


def _read_numbered_text(path: str) -> str:
    """
    Read a file whole as line-numbered text.

    Format: "  N| content" where N is the line's real position in the file.
    The model cites these numbers as evidence, so every line keeps its own.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    if not text:
        return ""
    lines = text.split("\n")
    if text.endswith("\n"):
        lines.pop()
    width = len(str(len(lines)))
    return "\n".join(f"{n:>{width}}| {line}" for n, line in enumerate(lines, 1))


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
            ["git", "-C", repo_root, "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", "replace")
        # -z: NUL-separated, unquoted names (without it git C-quotes non-ASCII/special paths).
        git_files = [name for name in out.split("\0") if name]
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


# The NUL heuristic version control uses: a NUL byte in the first 8 KiB marks
# a binary file.
_BINARY_PROBE_BYTES = 8192


def _is_text_file(path: str) -> bool:
    with open(path, "rb") as f:
        return b"\0" not in f.read(_BINARY_PROBE_BYTES)


def _format_withheld(withheld: List[Tuple[str, int]]) -> str:
    """The changed files the update does not show, each by name and size, so
    the model knows they changed and that their lines are never cited."""
    return _FILE_SEPARATOR.join(
        f"=== FILE: {path} ({size} bytes, not shown) ===" for path, size in withheld
    )


def _format_file_payload(file_blobs: List[Tuple[str, Optional[str]]]) -> str:
    """
    Format line-numbered file blobs (see `_read_numbered_text`) into a payload.

    Each file is formatted as:
        === FILE: path/to/file.py ===
          1| first line
          2| second line
          ...

    A deleted file (content None) is listed as `=== FILE: path (deleted) ===`.
    """
    parts = []
    for path, content in file_blobs:
        if content is None:
            parts.append(f"=== FILE: {path} (deleted) ===")
        elif content:
            parts.append(f"=== FILE: {path} ===\n{content}")
    return _FILE_SEPARATOR.join(parts)


def _window_file_blobs(
    file_blobs: List[Tuple[str, Optional[str]]],
    max_payload_chars: int,
) -> List[List[Tuple[str, Optional[str]]]]:
    """
    Group file blobs into batches whose formatted payload fits the budget.

    A file is never split: the model must see every line it re-cites. The
    update withholds a file larger than the budget before windowing, so a
    batch holds at most the budget's worth of whole files.
    """
    batches: List[List[Tuple[str, Optional[str]]]] = [[]]
    batch_chars = 0
    for blob in file_blobs:
        blob_chars = len(_format_file_payload([blob])) + len(_FILE_SEPARATOR)
        if batches[-1] and batch_chars + blob_chars > max_payload_chars:
            batches.append([])
            batch_chars = 0
        batches[-1].append(blob)
        batch_chars += blob_chars
    return batches


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Parse JSON robustly.
    
    We intentionally keep this simple:
    - prefer full parse
    - otherwise try the first {...} block
    """
    if not isinstance(text, str):
        raise RepoMemoryResponseError("LLM response must be text")

    text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise RepoMemoryResponseError("LLM did not return JSON")
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as exc:
            raise RepoMemoryResponseError(
                f"LLM returned malformed JSON: {exc.msg}"
            ) from exc

    if not isinstance(data, dict):
        raise RepoMemoryResponseError("LLM JSON must be an object")
    return data


def _validate_repo_model(model: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the stable outer shape used by RepoMemory builders.

    The Book is the one schema: a response without a 'sections' object is
    a malformed response to repair, not a format to fall back to.
    """
    if not isinstance(model.get("summary"), str):
        raise RepoMemoryResponseError(
            "RepoMemory JSON requires a string 'summary'"
        )
    if not isinstance(model.get("sections"), dict):
        raise RepoMemoryResponseError(
            "RepoMemory JSON requires a 'sections' object"
        )
    return model


def _evidence_paths(section: Any) -> Set[str]:
    """Return the file paths cited as evidence by a section's claims."""
    claims = section.get("claims") if isinstance(section, dict) else None
    return {
        item["path"]
        for claim in (claims if isinstance(claims, list) else [])
        if isinstance(claim, dict) and isinstance(claim.get("evidence"), list)
        for item in claim["evidence"]
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }


def _merge_repo_model_update(
    previous_model: Dict[str, Any],
    update: Dict[str, Any],
    changed_files: List[str],
) -> Dict[str, Any]:
    """Apply an incremental update patch onto the previous RepoMemory model.

    The update carries only changed/added sections plus an explicit
    `removed_sections` list; every other previous section is carried over
    verbatim, so a section can never be lost by being omitted.

    A carried-over section would keep stale evidence line numbers if it cites
    a changed file, so such a section must be returned (refreshed) or removed.
    """
    _validate_repo_model(update)
    removed = update.get("removed_sections")
    if not isinstance(removed, list) or not all(
        isinstance(sid, str) for sid in removed
    ):
        raise RepoMemoryResponseError(
            "RepoMemory update JSON requires a 'removed_sections' list of "
            "section ids"
        )
    previous_sections = previous_model["sections"]
    unknown = [sid for sid in removed if sid not in previous_sections]
    if unknown:
        raise RepoMemoryResponseError(
            "removed_sections names sections not in the previous model: "
            f"{unknown}"
        )
    core = [sid for sid in removed if sid.startswith("core.")]
    if core:
        raise RepoMemoryResponseError(
            "core sections cannot be removed (return them with empty claims "
            f"instead): {core}"
        )
    conflicting = [sid for sid in removed if sid in update["sections"]]
    if conflicting:
        raise RepoMemoryResponseError(
            f"sections both removed and returned: {conflicting}"
        )
    changed = set(changed_files)
    stale = [
        sid
        for sid, sec in previous_sections.items()
        if sid not in removed
        and sid not in update["sections"]
        and _evidence_paths(sec) & changed
    ]
    if stale:
        raise RepoMemoryResponseError(
            "sections citing changed files must be returned with refreshed "
            f"evidence: {stale}"
        )
    sections = {
        sid: sec for sid, sec in previous_sections.items() if sid not in removed
    }
    sections.update(update["sections"])
    return {"summary": update["summary"], "sections": sections}


def _complete_repo_model(
    *,
    llm: LLMLike,
    prompt: str,
    max_retries: int,
    validate: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """Complete and parse a RepoMemory model, repairing only bad responses."""
    if isinstance(max_retries, bool) or not isinstance(max_retries, int):
        raise ValueError("max_retries must be a non-negative integer")
    if max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer")

    messages = [{"role": "user", "content": prompt}]
    for attempt in range(max_retries + 1):
        # Provider, authentication, and transport failures intentionally escape
        # this loop. Only response parsing/schema failures are retryable.
        # temperature is left at the backend default: reasoning-first models
        # (e.g. the gpt-5.6 family) reject explicit temperature=0.
        response = llm.llm_completion(
            messages=messages,
            role="repo_memory",
        )
        try:
            return validate(_extract_json(response))
        except RepoMemoryResponseError as exc:
            if attempt >= max_retries:
                raise

            retry_template = load_prompt(
                "execution/memories/repo_memory/prompts/"
                "infer_repo_model_retry.md"
            )
            retry_prompt = render_prompt(
                retry_template,
                {"validation_error": str(exc)},
            )
            response_text = (
                response if isinstance(response, str) else repr(response)
            )
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response_text[:20000]},
                {"role": "user", "content": retry_prompt},
            ]


def plan_files_to_read(
    llm: LLMLike,
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
            messages=[{"role": "user", "content": prompt}],
            role="repo_memory",
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


def _blobs_within_budget(
    repo_root: str,
    files: List[str],
    max_payload_chars: int,
) -> Tuple[List[Tuple[str, Optional[str]]], List[Tuple[str, int]]]:
    """The files as line-numbered text, each within the budget on its own,
    and the ones withheld by name and size: a binary file has no lines to
    cite, and a file whose payload alone exceeds the budget is a dataset or
    a model dump, not code (a campaign commits its datasets, and a CLI
    refuses a prompt past its input limit — the baseline inference of the
    release gate's S1 sent a 1 MB csv whole and Codex refused the call).
    A file no longer on disk is carried as `None`: the call that carries a
    deleted file has to say it is gone."""
    blobs: List[Tuple[str, Optional[str]]] = []
    withheld: List[Tuple[str, int]] = []
    for rel in files:
        abs_path = os.path.join(repo_root, rel)
        if not os.path.exists(abs_path):
            blobs.append((rel, None))
            continue
        blob = (rel, _read_numbered_text(abs_path)) if _is_text_file(abs_path) else None
        if blob is None or len(_format_file_payload([blob])) > max_payload_chars:
            withheld.append((rel, os.path.getsize(abs_path)))
            continue
        blobs.append(blob)
    return blobs, withheld


def _patch_model_over_windows(
    llm: LLMLike,
    model: Dict[str, Any],
    batches: List[List[Tuple[str, Optional[str]]]],
    diff_summary: str,
    max_retries: int,
    withheld: List[Tuple[str, int]],
) -> Dict[str, Any]:
    """One update call per batch of whole files, each patching the model the
    previous call produced and validated against that batch's files; the
    withheld files are listed in the first call, so nothing is cut from
    what the model is told and their evidence is kept as it is."""
    template = load_prompt("execution/memories/repo_memory/prompts/infer_repo_model_update.md")
    for index, batch in enumerate(batches):
        batch_files = [rel for rel, _ in batch]
        payload = _format_file_payload(batch)
        if index == 0 and withheld:
            payload = _FILE_SEPARATOR.join(part for part in (_format_withheld(withheld), payload) if part)
        prompt = render_prompt(
            template,
            {
                "diff_summary": diff_summary,
                "previous_model_json": json.dumps(model, indent=2),
                "changed_payload": payload,
            },
        )
        model = _complete_repo_model(
            llm=llm,
            prompt=prompt,
            max_retries=max_retries,
            validate=lambda update, model=model, batch_files=batch_files: (
                _merge_repo_model_update(model, update, batch_files)
            ),
        )
    return model


def infer_repo_model_initial(
    llm: LLMLike,
    repo_root: str,
    repo_map: Dict[str, Any],
    max_files_to_read: int = 20,
    max_retries: int = 0,
    max_payload_chars: int = UPDATE_PAYLOAD_MAX_CHARS,
) -> Dict[str, Any]:
    """
    Build a semantic repo model from scratch using agentic file selection.

    Output is JSON with evidence-backed claims.

    The files the planner chose are read whole, within the same budget the
    update keeps: a binary or over-budget file is listed by name and size,
    never shown, and files past the first budget's worth reach the model in
    further calls, each patching the model the previous call produced.
    """
    files_to_read = plan_files_to_read(
        llm,
        repo_map=repo_map,
        max_files_to_read=max_files_to_read,
    )
    file_blobs, withheld = _blobs_within_budget(repo_root, files_to_read, max_payload_chars)
    first, *rest = _window_file_blobs(file_blobs, max_payload_chars)
    files_payload = _format_file_payload(first)
    if withheld:
        files_payload = _FILE_SEPARATOR.join(part for part in (_format_withheld(withheld), files_payload) if part)

    template = load_prompt("execution/memories/repo_memory/prompts/infer_repo_model_initial.md")
    prompt = render_prompt(
        template,
        {
            "repo_map_key_files_json": json.dumps(repo_map.get("key_files", [])),
            "repo_map_entrypoints_json": json.dumps(repo_map.get("entrypoints", [])),
            "files_payload": files_payload,
        },
    )
    model = _complete_repo_model(
        llm=llm,
        prompt=prompt,
        max_retries=max_retries,
        validate=_validate_repo_model,
    )
    return _patch_model_over_windows(
        llm, model, rest, "the repository's remaining files, read after the first call's budget was spent", max_retries, [],
    )


def infer_repo_model_with_retry(
    llm: LLMLike,
    repo_root: str,
    repo_map: Dict[str, Any],
    max_files_to_read: int = 20,
    max_retries: int = 4,
    max_payload_chars: int = UPDATE_PAYLOAD_MAX_CHARS,
) -> Dict[str, Any]:
    """
    Build a semantic repo model with bounded structured-response repair.

    File selection and file reads happen once. A retry only asks the model to
    repair malformed JSON or an invalid outer schema.
    """
    return infer_repo_model_initial(
        llm=llm,
        repo_root=repo_root,
        repo_map=repo_map,
        max_files_to_read=max_files_to_read,
        max_retries=max_retries,
        max_payload_chars=max_payload_chars,
    )


def infer_repo_model_update(
    llm: LLMLike,
    repo_root: str,
    repo_map: Dict[str, Any],
    previous_model: Dict[str, Any],
    diff_summary: str,
    changed_files: List[str],
    max_retries: int = 0,
    max_payload_chars: int = UPDATE_PAYLOAD_MAX_CHARS,
) -> Dict[str, Any]:
    """
    Incrementally update RepoModel using the previous model + diffs + changed files.
    
    The LLM sees the complete previous model and returns a patch (changed
    sections + `removed_sections`); this returns that patch merged onto the
    previous model, so untouched sections are never lost.
    The updated model uses line-number-based evidence references.

    When the changed files do not fit in `max_payload_chars`, they are
    windowed: one call per batch of whole files, each patching the model the
    previous call produced and validated against that batch's files.

    The memory cites lines of code. A binary file has none, and a file whose
    payload alone exceeds the budget is a dataset or a model dump, not code
    (a campaign commits its datasets, and a CLI refuses a prompt past its
    input limit): neither is shown. The first call lists them by name and
    size instead, so nothing is cut from what the model is told, and their
    evidence is kept as it is.
    """
    changed_blobs, withheld = _blobs_within_budget(repo_root, changed_files, max_payload_chars)
    batches = _window_file_blobs(changed_blobs, max_payload_chars)
    return _patch_model_over_windows(llm, previous_model, batches, diff_summary, max_retries, withheld)
