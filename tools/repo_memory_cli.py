#!/usr/bin/env python3
"""
RepoMemory CLI (standalone; no `import src`)
==========================================

This file is intentionally **not** inside the `src/` package.

Why:
- `python -m src.<...>` imports `src/__init__.py`, which today triggers heavy imports,
  agent registration prints, and warnings.
- Coding agents use this command as a "tool", so stdout must stay clean and only
  contain the requested RepoMemory content.

This script is the simplest reliable approach: read `.praxium/repo_memory.json`
directly and render sections without importing Praxium internals.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple


def _load_doc(repo_root: str) -> Dict[str, Any]:
    repo_root = os.path.abspath(repo_root)
    path = os.path.join(repo_root, ".praxium", "repo_memory.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"RepoMemory file not found: {path}")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def _ensure_book_v2(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Best-effort migration to a v2 "book" view.

    In Praxium today, `.praxium/repo_memory.json` should already contain `book`.
    This is only a safety net for older v1 documents.
    """
    doc = doc or {}
    if isinstance(doc.get("book"), dict) and isinstance((doc.get("book") or {}).get("sections"), dict):
        return doc

    # Minimal derivation from v1-ish fields
    repo_model = doc.get("repo_model", {}) if isinstance(doc.get("repo_model"), dict) else {}
    summary = (repo_model.get("summary") or "").strip()
    claims = repo_model.get("claims", []) if isinstance(repo_model.get("claims"), list) else []
    entrypoints = repo_model.get("entrypoints", []) if isinstance(repo_model.get("entrypoints"), list) else []
    where = repo_model.get("where_to_edit", []) if isinstance(repo_model.get("where_to_edit"), list) else []

    sections: Dict[str, Any] = {
        "core.architecture": {
            "title": "Architecture",
            "one_liner": "System design and module structure",
            "claims": claims,
        },
        "core.entrypoints": {
            "title": "Entrypoints",
            "one_liner": "How to run the application",
            "content": entrypoints,
        },
        "core.where_to_edit": {
            "title": "Where to edit",
            "one_liner": "Key files for modifications",
            "content": where,
        },
    }

    toc = _build_toc(sections)
    doc["book"] = {"summary": summary, "sections": sections, "toc": toc}
    return doc


def _build_toc(sections: Dict[str, Any]) -> List[Dict[str, str]]:
    toc: List[Dict[str, str]] = []
    for sid in sorted((sections or {}).keys()):
        sec = (sections or {}).get(sid, {}) or {}
        toc.append(
            {
                "id": sid,
                "title": str(sec.get("title") or sid),
                "one_liner": str(sec.get("one_liner") or ""),
            }
        )
    return toc


def _render_summary_toc(doc: Dict[str, Any], max_chars: int) -> str:
    doc = _ensure_book_v2(doc)
    book = doc.get("book", {}) or {}
    summary = (book.get("summary") or "").strip() or "(missing)"
    toc = book.get("toc") or _build_toc(book.get("sections", {}) or {})

    lines = [
        "# Repo Memory (book)",
        f"Schema: v{doc.get('schema_version')}",
        f"GeneratedAt: {doc.get('generated_at')}",
        "",
        "## Summary",
        summary,
        "",
        "## Table of Contents (section IDs)",
    ]
    for item in toc:
        sid = (item or {}).get("id", "")
        title = (item or {}).get("title", "")
        one = (item or {}).get("one_liner", "")
        if sid:
            suffix = f": {one}" if one else ""
            lines.append(f"- [{sid}] {title}{suffix}")
    lines.extend(
        [
            "",
            "## How to read details",
            "- Open `.praxium/repo_memory.json`",
            "- Find `book.sections[section_id]` from the TOC above",
        ]
    )
    text = "\n".join(lines)
    return text[:max_chars] if len(text) > max_chars else text


def _render_section(doc: Dict[str, Any], section_id: str, max_chars: int) -> str:
    doc = _ensure_book_v2(doc)
    sections = ((doc.get("book") or {}).get("sections") or {}) if isinstance((doc.get("book") or {}).get("sections"), dict) else {}
    if section_id not in sections:
        available = list(sections.keys())
        msg = f"Section '{section_id}' not found. Available: {available}"
        return msg[:max_chars]

    sec = sections.get(section_id, {}) or {}
    title = str(sec.get("title") or section_id)
    one_liner = str(sec.get("one_liner") or "")

    # Claims-style section
    claims = sec.get("claims", None)
    if isinstance(claims, list):
        out: List[str] = [f"# {title}", ""]
        if one_liner:
            out.extend([one_liner, ""])
        for claim in claims:
            kind = (claim or {}).get("kind", "?")
            stmt = (claim or {}).get("statement", "")
            out.append(f"- [{kind}] {stmt}")
            for ev in (claim or {}).get("evidence", []) or []:
                path = (ev or {}).get("path", "?")
                quote = (ev or {}).get("quote", "")
                quote_short = quote if len(quote) <= 200 else quote[:200] + "...(truncated)"
                out.append(f'  - evidence: {path}: "{quote_short}"')
        text = "\n".join(out)
        return text[:max_chars]

    # Content-style section
    content = sec.get("content", None)
    if content is not None:
        text = json.dumps(content, indent=2, ensure_ascii=False)
        return text[:max_chars]

    return f"(empty section: {section_id})"[:max_chars]


def _cmd_get_section(args: argparse.Namespace) -> int:
    doc = _load_doc(args.repo_root)
    sys.stdout.write(_render_section(doc, args.section_id, args.max_chars) + "\n")
    return 0


def _cmd_list_sections(args: argparse.Namespace) -> int:
    doc = _ensure_book_v2(_load_doc(args.repo_root))
    toc = (doc.get("book") or {}).get("toc") or _build_toc((doc.get("book") or {}).get("sections") or {})
    for item in toc:
        sid = (item or {}).get("id", "")
        title = (item or {}).get("title", "")
        if sid:
            sys.stdout.write(f"{sid}\t{title}\n")
    return 0


def _cmd_summary_toc(args: argparse.Namespace) -> int:
    doc = _load_doc(args.repo_root)
    sys.stdout.write(_render_summary_toc(doc, args.max_chars) + "\n")
    return 0


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RepoMemory CLI (standalone)")
    parser.add_argument("--repo-root", default=".", help="Repo root (default: .)")

    sub = parser.add_subparsers(dest="command", required=True)

    p_get = sub.add_parser("get-section", help="Print one section by id")
    p_get.add_argument("section_id", help="Section id (e.g., core.architecture)")
    p_get.add_argument("--max-chars", type=int, default=8000, help="Max output chars")
    p_get.set_defaults(func=_cmd_get_section)

    p_list = sub.add_parser("list-sections", help="List available section IDs (TOC)")
    p_list.set_defaults(func=_cmd_list_sections)

    p_toc = sub.add_parser("summary-toc", help="Print Summary + TOC")
    p_toc.add_argument("--max-chars", type=int, default=3000, help="Max output chars")
    p_toc.set_defaults(func=_cmd_summary_toc)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

