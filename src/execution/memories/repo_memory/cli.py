#!/usr/bin/env python3
"""
RepoMemory CLI (tiny, auditable)
===============================

This CLI exists for one reason: make RepoMemory section retrieval easy and auditable
for coding agents that can run shell commands (currently: Claude Code via the "Bash"
tool).

Why a CLI instead of an MCP tool?
- Zero external dependencies
- Works with the current Claude Code allowlist (Bash is already enabled)
- The exact command invocation can be logged in `changes.log` for auditability
"""

from __future__ import annotations

import argparse
import contextlib
from functools import lru_cache
import io
import json
import os
import sys
import warnings
from typing import Any, Dict

@lru_cache(maxsize=1)
def _repo_memory_manager():
    """
    Import RepoMemoryManager with stdout redirected to stderr.

    Why:
    - Importing the `src` package currently triggers some import-time prints
      (e.g., CodingAgentFactory registration).
    - This CLI is used as a "tool" and must keep stdout clean (only the requested
      section content), so we route any noisy import-time prints to stderr.
    """
    # Keep stdout clean: this CLI is used as a "tool", so stdout should contain
    # ONLY the requested RepoMemory content (no import-time noise).
    with contextlib.redirect_stdout(io.StringIO()):
        with warnings.catch_warnings():
            # Some dependency stacks emit deprecation warnings at import-time.
            # These are not actionable for "get-section" usage and pollute output.
            warnings.simplefilter("ignore", category=DeprecationWarning)
            from src.execution.memories.repo_memory.manager import RepoMemoryManager  # local import by design
    return RepoMemoryManager


def _load_doc(repo_root: str) -> Dict[str, Any]:
    """Load `.kapso/repo_memory.json` from a repo worktree and migrate to v2."""
    RepoMemoryManager = _repo_memory_manager()
    repo_root = os.path.abspath(repo_root)
    path = os.path.join(repo_root, RepoMemoryManager.MEMORY_REL_PATH)
    if not os.path.exists(path):
        raise FileNotFoundError(f"RepoMemory file not found: {path}")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        doc = json.load(f)
    return RepoMemoryManager.migrate_v1_to_v2(doc)


def cmd_get_section(args: argparse.Namespace) -> int:
    RepoMemoryManager = _repo_memory_manager()
    doc = _load_doc(args.repo_root)
    text = RepoMemoryManager.get_section(doc, args.section_id, max_chars=args.max_chars)
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")
    return 0


def cmd_list_sections(args: argparse.Namespace) -> int:
    RepoMemoryManager = _repo_memory_manager()
    doc = _load_doc(args.repo_root)
    toc = RepoMemoryManager.list_sections(doc)
    for item in toc:
        sid = (item or {}).get("id", "")
        title = (item or {}).get("title", "")
        if sid:
            sys.stdout.write(f"{sid}\t{title}\n")
    return 0


def cmd_summary_toc(args: argparse.Namespace) -> int:
    RepoMemoryManager = _repo_memory_manager()
    doc = _load_doc(args.repo_root)
    text = RepoMemoryManager.render_summary_and_toc(doc, max_chars=args.max_chars)
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RepoMemory CLI")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Path to the repo root that contains `.kapso/repo_memory.json` (default: .)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_get = sub.add_parser("get-section", help="Print one RepoMemory section by id")
    p_get.add_argument("section_id", help="Section id (e.g., core.architecture)")
    p_get.add_argument("--max-chars", type=int, default=8000, help="Max output chars")
    p_get.set_defaults(func=cmd_get_section)

    p_list = sub.add_parser("list-sections", help="List available section IDs (TOC)")
    p_list.set_defaults(func=cmd_list_sections)

    p_brief = sub.add_parser("summary-toc", help="Print RepoMemory Summary + TOC")
    p_brief.add_argument("--max-chars", type=int, default=3000, help="Max output chars")
    p_brief.set_defaults(func=cmd_summary_toc)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

