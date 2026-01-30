"""
RepoMemory observation helpers.

These utilities exist to make RepoMemory behavior observable in real runs:
- How RepoMemory changes across experiments (book stats / diffs)
- Which RepoMemory sections the coding agent claims to have consulted

Why a separate module?
- Keep `src/repo_memory/manager.py` from growing further (it is already large).
- Keep parsing / logging concerns out of core persistence logic.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# Section IDs are used as a stable "contract" (e.g., core.architecture, opt.payment_flow)
_SECTION_ID_PATTERN = re.compile(r"^[a-z0-9_.-]+$")


def book_claim_counts(book: Dict[str, Any]) -> Dict[str, int]:
    """
    Return {section_id -> claim_count} for a v2 book.
    
    This is small and stable, so it is safe to log/store in experiment metadata.
    """
    sections = (book or {}).get("sections", {})
    if not isinstance(sections, dict):
        return {}
    out: Dict[str, int] = {}
    for sid, sec in sections.items():
        claims = (sec or {}).get("claims", [])
        out[str(sid)] = len(claims) if isinstance(claims, list) else 0
    return out


def book_stats(book: Dict[str, Any]) -> Dict[str, Any]:
    """Small, log-friendly summary of a RepoMemory v2 book."""
    summary = ((book or {}).get("summary") or "").strip()
    toc = (book or {}).get("toc", []) or []
    toc_ids = [str((t or {}).get("id", "")) for t in toc if (t or {}).get("id")]
    return {
        "summary": summary[:500],
        "toc_ids": toc_ids,
        "claim_counts": book_claim_counts(book),
    }


def diff_book_stats(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute a small diff between two book snapshots.
    
    This is intentionally conservative:
    - Summaries compared as strings
    - Section changes reported as claim-count deltas
    """
    b = book_stats(before or {})
    a = book_stats(after or {})

    b_counts = b.get("claim_counts", {}) or {}
    a_counts = a.get("claim_counts", {}) or {}

    all_sections = sorted(set(b_counts.keys()) | set(a_counts.keys()))
    deltas = {sid: int(a_counts.get(sid, 0)) - int(b_counts.get(sid, 0)) for sid in all_sections}

    return {
        "summary_changed": (b.get("summary") or "") != (a.get("summary") or ""),
        "before_summary": b.get("summary"),
        "after_summary": a.get("summary"),
        "claim_count_deltas": deltas,
    }


def extract_repo_memory_sections_consulted(changes_log_text: str, agent_output: str = "") -> List[str]:
    """
    Extract section IDs from a `changes.log` file and/or agent output.
    
    Expected format in changes.log (we instruct agents to write this):
      RepoMemory sections consulted: core.architecture, core.where_to_edit
    Or:
      RepoMemory sections consulted: none
    
    Also detects MCP tool calls in agent output:
      get_repo_memory_section(section_id="core.architecture")
    
    Args:
        changes_log_text: Content of changes.log file
        agent_output: Raw agent output (optional, for detecting MCP calls)
    
    Returns:
        Sorted unique section IDs (empty list means none or not specified).
    """
    ids: List[str] = []
    
    # 1. Extract from changes.log (explicit declaration)
    text = changes_log_text or ""
    # Find the first occurrence (case-insensitive).
    #
    # Agents don't always respect "put it on its own line". In practice we see cases like:
    #   "... did X. RepoMemory sections consulted: core.architecture, core.where_to_edit"
    # So we search for the substring anywhere in the line and parse the RHS.
    rhs = ""
    for raw_line in text.splitlines():
        m = re.search(r"repomemory sections consulted:\s*(.*)$", raw_line, flags=re.IGNORECASE)
        if m:
            rhs = (m.group(1) or "").strip()
            break
    if rhs and rhs.lower() not in ("none", "n/a", "na"):
        # Remove common wrappers: brackets, quotes.
        rhs = rhs.strip().strip("[](){}")
        # Split on commas or whitespace.
        parts = re.split(r"[,\s]+", rhs)
        for p in parts:
            p = (p or "").strip().strip('"').strip("'")
            if not p:
                continue
            if p.lower() in ("none", "n/a", "na"):
                continue
            if _SECTION_ID_PATTERN.match(p):
                ids.append(p)
    
    # 2. Extract from agent output (MCP tool calls)
    # Pattern: get_repo_memory_section(section_id="core.architecture")
    # or: get_repo_memory_section(section_id='core.architecture')
    # or: "section_id": "core.architecture"
    output = agent_output or ""
    mcp_patterns = [
        r'get_repo_memory_section\s*\(\s*section_id\s*=\s*["\']([^"\']+)["\']',
        r'"section_id"\s*:\s*["\']([^"\']+)["\']',
    ]
    for pattern in mcp_patterns:
        for match in re.finditer(pattern, output, flags=re.IGNORECASE):
            section_id = match.group(1).strip()
            if _SECTION_ID_PATTERN.match(section_id):
                ids.append(section_id)
    
    return sorted(set(ids))

