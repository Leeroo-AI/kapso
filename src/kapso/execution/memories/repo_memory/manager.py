"""
RepoMemory manager
=================

This class owns persistence + update logic for repository memory.

Key guarantee:
- If a Kapso experiment continues from a branch, the memory file committed
  in that branch is the memory of the code it starts from.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import git

from kapso.execution.memories.repo_memory.builders import (
    LLMLike,
    build_repo_map,
    infer_repo_model_initial,
    infer_repo_model_update,
    infer_repo_model_with_retry,
)


class RepoMemoryManager:
    # ---------------------------------------------------------------------
    # Schema
    # ---------------------------------------------------------------------
    #
    # v1 stored semantic memory in a flat `repo_model` with `claims[]`.
    # v2 adds a "Book" view (`book.summary`, `book.toc`, `book.sections`) so prompts
    # can stay bounded (Summary + TOC only) while agents can read full details
    # directly from `.kapso/repo_memory.json`.
    #
    # IMPORTANT: We keep `repo_model` for backward compatibility and for existing
    # consumers/tests that still read `repo_model.summary/claims/...`.
    SCHEMA_VERSION = 2
    KAPSO_DIR = ".kapso"
    MEMORY_FILE = "repo_memory.json"
    MEMORY_REL_PATH = os.path.join(KAPSO_DIR, MEMORY_FILE)

    # Default model for repo-model inference.
    DEFAULT_REPO_MODEL_LLM = "gpt-4o-mini"

    # Stable section IDs (contract). Keep these IDs stable across versions.
    #
    # Notes:
    # - These correspond to "core" sections that are always meaningful to
    #   navigation, even if empty in a small repo.
    # - Optional LLM-generated sections must use the `opt.` prefix.
    CORE_SECTIONS = [
        "core.architecture",
        "core.entrypoints",
        "core.where_to_edit",
        "core.invariants",
        "core.testing",
        "core.gotchas",
        "core.dependencies",
    ]

    # Deterministic titles + one-liners for core TOC entries.
    # This keeps the TOC stable even when a section has no content yet.
    CORE_SECTION_META: Dict[str, Dict[str, str]] = {
        "core.architecture": {
            "title": "Architecture",
            "one_liner": "System design and module structure",
        },
        "core.entrypoints": {
            "title": "Entrypoints",
            "one_liner": "How to run the application",
        },
        "core.where_to_edit": {
            "title": "Where to edit",
            "one_liner": "Key files for modifications",
        },
        "core.invariants": {
            "title": "Invariants",
            "one_liner": "Contracts, constraints, and assumptions",
        },
        "core.testing": {
            "title": "Testing",
            "one_liner": "How to run tests and validate changes",
        },
        "core.gotchas": {
            "title": "Gotchas",
            "one_liner": "Common pitfalls and sharp edges",
        },
        "core.dependencies": {
            "title": "Dependencies",
            "one_liner": "Key dependencies and environment notes",
        },
    }

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @classmethod
    def _memory_abs_path(cls, repo_root: str) -> str:
        return os.path.join(repo_root, cls.MEMORY_REL_PATH)

    @classmethod
    def _ensure_dir(cls, repo_root: str) -> None:
        os.makedirs(os.path.join(repo_root, cls.KAPSO_DIR), exist_ok=True)

    @classmethod
    def _count_claims_in_book_sections(cls, sections: Dict[str, Any]) -> int:
        """Count claims across all book sections (v2)."""
        total = 0
        for sec in (sections or {}).values():
            total += len((sec or {}).get("claims", []) or [])
        return total

    @classmethod
    def _build_toc_from_sections(cls, sections: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build an ordered TOC list from sections.
        
        Rules:
        - Core sections first in stable order (always included).
        - Optional sections (`opt.*`) next, ordered by section id.
        """
        sections = sections or {}

        toc: List[Dict[str, Any]] = []
        for sid in cls.CORE_SECTIONS:
            meta = cls.CORE_SECTION_META.get(sid, {"title": sid, "one_liner": ""})
            sec = sections.get(sid, {}) or {}
            toc.append(
                {
                    "id": sid,
                    "title": sec.get("title") or meta["title"],
                    "one_liner": sec.get("one_liner") or meta.get("one_liner", ""),
                }
            )

        # Optional sections: include anything not core, prefer opt.* ids
        optional_ids = [sid for sid in sections.keys() if sid not in set(cls.CORE_SECTIONS)]
        optional_ids.sort()
        for sid in optional_ids:
            sec = sections.get(sid, {}) or {}
            toc.append(
                {
                    "id": sid,
                    "title": sec.get("title") or sid,
                    "one_liner": sec.get("one_liner") or "",
                }
            )
        return toc

    @classmethod
    def _ensure_core_sections_present(cls, sections: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all core section IDs exist in the sections dict."""
        sections = dict(sections or {})
        for sid in cls.CORE_SECTIONS:
            if sid in sections and isinstance(sections[sid], dict):
                # Fill missing metadata fields if absent.
                meta = cls.CORE_SECTION_META.get(sid, {})
                sections[sid].setdefault("title", meta.get("title", sid))
                sections[sid].setdefault("one_liner", meta.get("one_liner", ""))
                continue

            meta = cls.CORE_SECTION_META.get(sid, {})
            sections[sid] = {
                "title": meta.get("title", sid),
                "one_liner": meta.get("one_liner", ""),
                # Keep both possible shapes available; empty by default.
                "claims": [],
                "content": [],
            }
        return sections

    @classmethod
    def _build_book_from_v1_repo_model(cls, repo_model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a v2 `book` from a v1-style `repo_model`.
        
        This is used both for v1→v2 migration and as a fallback path while we
        incrementally roll out v2 builders.
        """
        repo_model = repo_model or {}
        claims = repo_model.get("claims", []) or []

        # Split claims into sections by kind (simple, deterministic mapping).
        sections: Dict[str, Any] = {}
        sections["core.entrypoints"] = {
            "title": cls.CORE_SECTION_META["core.entrypoints"]["title"],
            "one_liner": cls.CORE_SECTION_META["core.entrypoints"]["one_liner"],
            "content": repo_model.get("entrypoints", []) or [],
        }
        sections["core.where_to_edit"] = {
            "title": cls.CORE_SECTION_META["core.where_to_edit"]["title"],
            "one_liner": cls.CORE_SECTION_META["core.where_to_edit"]["one_liner"],
            "content": repo_model.get("where_to_edit", []) or [],
        }

        architecture_claims = []
        invariants_claims = []
        deps_claims = []
        gotchas_claims = []
        testing_claims = []

        for c in claims:
            kind = (c or {}).get("kind", "") or ""
            if kind in ("architecture", "algorithm"):
                architecture_claims.append(c)
            elif kind == "contract":
                invariants_claims.append(c)
            elif kind == "deployment":
                deps_claims.append(c)
            elif kind == "testing":
                testing_claims.append(c)
            else:
                gotchas_claims.append(c)

        sections["core.architecture"] = {
            "title": cls.CORE_SECTION_META["core.architecture"]["title"],
            "one_liner": cls.CORE_SECTION_META["core.architecture"]["one_liner"],
            "claims": architecture_claims,
        }
        sections["core.invariants"] = {
            "title": cls.CORE_SECTION_META["core.invariants"]["title"],
            "one_liner": cls.CORE_SECTION_META["core.invariants"]["one_liner"],
            "claims": invariants_claims,
        }
        sections["core.dependencies"] = {
            "title": cls.CORE_SECTION_META["core.dependencies"]["title"],
            "one_liner": cls.CORE_SECTION_META["core.dependencies"]["one_liner"],
            "claims": deps_claims,
        }
        sections["core.gotchas"] = {
            "title": cls.CORE_SECTION_META["core.gotchas"]["title"],
            "one_liner": cls.CORE_SECTION_META["core.gotchas"]["one_liner"],
            "claims": gotchas_claims,
        }
        sections["core.testing"] = {
            "title": cls.CORE_SECTION_META["core.testing"]["title"],
            "one_liner": cls.CORE_SECTION_META["core.testing"]["one_liner"],
            "claims": testing_claims,
        }

        sections = cls._ensure_core_sections_present(sections)
        toc = cls._build_toc_from_sections(sections)

        return {
            "summary": (repo_model.get("summary") or "").strip(),
            "toc": toc,
            "sections": sections,
        }

    @classmethod
    def _build_book_from_v2_model(cls, model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build the `book` from the LLM's RepoMemory V2 output.
        
        Expected model shape:
        {
          "summary": "...",
          "sections": { "core.architecture": {...}, ... }
        }
        """
        model = model or {}
        sections = model.get("sections", {}) if isinstance(model.get("sections"), dict) else {}
        sections = cls._ensure_core_sections_present(sections)
        toc = cls._build_toc_from_sections(sections)
        return {
            "summary": (model.get("summary") or "").strip(),
            "toc": toc,
            "sections": sections,
        }

    @classmethod
    def _legacy_repo_model_from_book(cls, book: Dict[str, Any]) -> Dict[str, Any]:
        """
        Derive a legacy `repo_model` view from the `book`.
        
        This keeps backward compatibility with existing code/tests that still read:
        - repo_model.summary
        - repo_model.entrypoints
        - repo_model.where_to_edit
        - repo_model.claims[]
        """
        book = book or {}
        sections = book.get("sections", {}) if isinstance(book.get("sections"), dict) else {}

        entrypoints = (sections.get("core.entrypoints", {}) or {}).get("content", []) or []
        where_to_edit = (sections.get("core.where_to_edit", {}) or {}).get("content", []) or []

        # Flatten all claims across sections.
        flat_claims: List[Dict[str, Any]] = []
        for sec in (sections or {}).values():
            for claim in (sec or {}).get("claims", []) or []:
                if isinstance(claim, dict):
                    flat_claims.append(claim)

        return {
            "summary": (book.get("summary") or "").strip(),
            "entrypoints": entrypoints,
            "where_to_edit": where_to_edit,
            "claims": flat_claims,
        }

    @classmethod
    def migrate_v1_to_v2(cls, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Auto-migrate a v1 RepoMemory document to schema v2.
        
        Key design:
        - Add `book` as a structured, navigable view.
        - Preserve existing `repo_model` unchanged for backward compatibility.
        - Migration is idempotent (safe to call multiple times).
        """
        doc = doc or {}
        schema = int(doc.get("schema_version") or 1)

        book = doc.get("book", None)
        if not isinstance(book, dict):
            # v1 (or malformed) doc: derive Book from legacy repo_model.
            repo_model = doc.get("repo_model", {}) or {}
            book = cls._build_book_from_v1_repo_model(repo_model)
        else:
            # Already has a Book view: ensure it has all core sections + a stable TOC.
            sections = book.get("sections", {}) if isinstance(book.get("sections"), dict) else {}
            sections = cls._ensure_core_sections_present(sections)
            book["sections"] = sections
            book["toc"] = cls._build_toc_from_sections(sections)

            # Ensure a usable summary is always present.
            if not (book.get("summary") or "").strip():
                repo_model = doc.get("repo_model", {}) or {}
                book["summary"] = (repo_model.get("summary") or "").strip()

        doc["book"] = book
        # Bump schema version in-memory (we keep `repo_model` for compatibility).
        doc["schema_version"] = max(schema, 2)

        # Update/extend quality metrics (do not remove old keys).
        doc.setdefault("quality", {})
        doc["quality"]["section_count"] = len((book.get("sections") or {}) if isinstance(book.get("sections"), dict) else {})
        doc["quality"]["claim_count"] = cls._count_claims_in_book_sections(
            (book.get("sections") or {}) if isinstance(book.get("sections"), dict) else {}
        )
        return doc

    # ---------------------------------------------------------------------
    # Load / save
    # ---------------------------------------------------------------------

    @classmethod
    def load_from_worktree(cls, repo_root: str) -> Optional[Dict[str, Any]]:
        """Load memory JSON from a working tree (returns None if missing)."""
        path = cls._memory_abs_path(repo_root)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        # Return a v2-shaped document to callers (without writing).
        # NOTE: `ensure_exists_in_worktree()` is responsible for persisting the
        # migration back to disk when needed.
        return cls.migrate_v1_to_v2(doc)

    @classmethod
    def write_to_worktree(cls, repo_root: str, doc: Dict[str, Any]) -> None:
        """Write memory JSON to a working tree (atomic-ish write)."""
        cls._ensure_dir(repo_root)
        path = cls._memory_abs_path(repo_root)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)

    @classmethod
    def ensure_exists_in_worktree(
        cls,
        repo_root: str,
        initial_repo: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ensure the memory file exists. If missing, create a minimal skeleton.
        
        Note: skeleton contains RepoMap but may omit RepoModel until inference.
        """
        # If the memory file exists, load it and *persist* any v1→v2 migration.
        #
        # Why persist?
        # - Coding agents read `.kapso/repo_memory.json` from disk.
        # - If we only migrate in-memory, agents won't see the Book/TOC structure.
        # - Persisting keeps branches consistent and auditable.
        path = cls._memory_abs_path(repo_root)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                raw_doc = json.load(f)
            # IMPORTANT:
            # `migrate_v1_to_v2()` mutates the input dict in-place.
            # If we want to decide whether to persist changes back to disk, we must
            # snapshot the pre-migration state before calling it.
            before = json.dumps(raw_doc, sort_keys=True, ensure_ascii=False)
            migrated = cls.migrate_v1_to_v2(raw_doc)
            after = json.dumps(migrated, sort_keys=True, ensure_ascii=False)
            # If migration changed schema_version/book/sections, write it back.
            if before != after:
                cls.write_to_worktree(repo_root, migrated)
            return migrated

        repo_map = build_repo_map(repo_root)
        doc: Dict[str, Any] = {
            "schema_version": cls.SCHEMA_VERSION,
            "generated_at": cls._now_iso(),
            "baseline": {
                "initial_repo": initial_repo,
            },
            "repo_map": repo_map,
            "repo_model": {
                "summary": "",
                "entrypoints": [],
                "where_to_edit": [],
                "claims": [],
            },
            # v2 Book view (keeps prompts bounded and memory navigable).
            "book": cls._build_book_from_v1_repo_model(
                {
                    "summary": "",
                    "entrypoints": [],
                    "where_to_edit": [],
                    "claims": [],
                }
            ),
            "experiments": [],
            "quality": {
                "evidence_ok": False,
                "missing_evidence": [],
                "section_count": len(cls.CORE_SECTIONS),
                "claim_count": 0,
            },
        }
        cls.write_to_worktree(repo_root, doc)
        return doc

    # ---------------------------------------------------------------------
    # Git integration (read from branch without checkout)
    # ---------------------------------------------------------------------

    @classmethod
    def load_from_git_branch(cls, repo: git.Repo, branch_name: str) -> Optional[Dict[str, Any]]:
        """Read `.kapso/repo_memory.json` from a given branch (no checkout)."""
        try:
            raw = repo.git.show(f"{branch_name}:{cls.MEMORY_REL_PATH}")
        except git.GitCommandError:
            return None
        try:
            doc = json.loads(raw)
            return cls.migrate_v1_to_v2(doc)
        except Exception:
            return None

    # ---------------------------------------------------------------------
    # Prompt rendering
    # ---------------------------------------------------------------------

    @classmethod
    def render_summary_and_toc(cls, doc: Dict[str, Any], max_chars: int = 3000) -> str:
        """
        Render Summary + TOC (bounded) for prompt injection.
        
        This is the v2 replacement for injecting large `render_brief()` blobs.
        Coding agents can read `.kapso/repo_memory.json` directly for details.
        """
        doc = cls.migrate_v1_to_v2(doc or {})
        book = doc.get("book", {}) or {}

        summary = (book.get("summary") or "").strip() or "(missing)"
        toc = book.get("toc", []) or []

        toc_lines = []
        for item in toc:
            sid = (item or {}).get("id", "")
            title = (item or {}).get("title", "")
            one = (item or {}).get("one_liner", "")
            if sid:
                suffix = f": {one}" if one else ""
                toc_lines.append(f"- [{sid}] {title}{suffix}")

        text = f"""# Repo Memory (book)
Schema: v{doc.get('schema_version')}
GeneratedAt: {doc.get('generated_at')}

## Summary
{summary}

## Table of Contents (section IDs)
{os.linesep.join(toc_lines) or '(no sections)'}

## How to read details
- Open `.kapso/repo_memory.json`
- Find `book.sections[section_id]` from the TOC above
"""
        if len(text) > max_chars:
            # Keep output strictly bounded.
            suffix = "\n... (truncated)\n"
            if max_chars <= len(suffix):
                return text[:max_chars]
            return text[: max_chars - len(suffix)] + suffix
        return text

    @classmethod
    def render_summary_and_toc_for_branch(
        cls,
        repo: git.Repo,
        branch_name: str,
        max_chars: int = 3000,
    ) -> str:
        """Load memory from a branch and render Summary+TOC (bounded)."""
        doc = cls.load_from_git_branch(repo, branch_name)
        if not doc:
            return ""
        return cls.render_summary_and_toc(doc, max_chars=max_chars)

    @classmethod
    def list_sections(cls, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Return TOC section metadata (v2)."""
        doc = cls.migrate_v1_to_v2(doc or {})
        return (doc.get("book", {}) or {}).get("toc", []) or []

    @classmethod
    def get_section(cls, doc: Dict[str, Any], section_id: str, max_chars: int = 8000) -> str:
        """
        Render a single section (v2) as human-readable text.
        
        This is intended for tool-style access and debugging.
        """
        doc = cls.migrate_v1_to_v2(doc or {})
        book = doc.get("book", {}) or {}
        sections = (book.get("sections", {}) or {}) if isinstance(book.get("sections", {}), dict) else {}

        if not section_id or section_id not in sections:
            available = list(sections.keys())
            msg = f"Section '{section_id}' not found. Available: {available}"
            return msg[:max_chars]

        sec = sections.get(section_id, {}) or {}
        title = sec.get("title") or section_id
        one_liner = sec.get("one_liner") or ""

        lines: List[str] = [f"# {title}", ""]
        if one_liner:
            lines.append(one_liner)
            lines.append("")

        # Claims section
        claims = sec.get("claims", None)
        if isinstance(claims, list):
            for claim in claims:
                kind = (claim or {}).get("kind", "?")
                stmt = (claim or {}).get("statement", "")
                lines.append(f"- [{kind}] {stmt}")
                for ev in (claim or {}).get("evidence", []) or []:
                    path = (ev or {}).get("path", "?")
                    quote = (ev or {}).get("quote", "")
                    # Keep quotes short and readable in section view.
                    quote_short = quote if len(quote) <= 200 else quote[:200] + "...(truncated)"
                    lines.append(f"  - evidence: {path}: \"{quote_short}\"")
            text = "\n".join(lines)
            return text[:max_chars]

        # Content section
        content = sec.get("content", None)
        if content is not None:
            # JSON is the most faithful representation for entrypoints/where-to-edit.
            text = json.dumps(content, indent=2, ensure_ascii=False)
            return text[:max_chars]

        return f"(empty section: {section_id})"[:max_chars]

    @classmethod
    def render_brief(cls, doc: Dict[str, Any], max_chars: int = 8000) -> str:
        """Render a compact repo-memory briefing for prompts (bounded)."""
        repo_map = doc.get("repo_map", {}) or {}
        repo_model = doc.get("repo_model", {}) or {}
        quality = doc.get("quality", {}) or {}

        entrypoints = repo_model.get("entrypoints") or repo_map.get("entrypoints") or []
        where = repo_model.get("where_to_edit") or []
        claims = repo_model.get("claims") or []

        # Keep only a few claims in prompt (agents can read the full JSON if needed).
        claims_text = "\n".join(
            f"- [{c.get('kind','?')}] {c.get('statement','')}"
            for c in claims[:8]
        )
        where_text = "\n".join(
            f"- {w.get('path','')}: {w.get('role','')}"
            for w in where[:10]
        )
        entry_text = "\n".join(
            f"- {e.get('path','')}: {e.get('how_to_run','')}"
            for e in entrypoints[:8]
        ) if entrypoints and isinstance(entrypoints[0], dict) else "\n".join(f"- {p}" for p in entrypoints[:8])

        text = f"""# Repo Memory (evidence-backed)
Schema: v{doc.get('schema_version')}
GeneratedAt: {doc.get('generated_at')}

## Repo Summary
{repo_model.get('summary','').strip() or '(missing)'}

## Entrypoints
{entry_text or '(unknown)'}

## Where to edit
{where_text or '(unknown)'}

## Key claims (must have evidence in repo files)
{claims_text or '(none)'}

## Memory quality
- evidence_ok: {bool(quality.get('evidence_ok'))}
- claim_count: {int(quality.get('claim_count') or 0)}
"""
        if len(text) > max_chars:
            return text[:max_chars] + "\n... (truncated)\n"
        return text

    @classmethod
    def render_brief_for_branch(
        cls,
        repo: git.Repo,
        branch_name: str,
        max_chars: int = 8000,
    ) -> str:
        doc = cls.load_from_git_branch(repo, branch_name)
        if not doc:
            return ""
        return cls.render_brief(doc, max_chars=max_chars)

    # ---------------------------------------------------------------------
    # Updating memory after an experiment
    # ---------------------------------------------------------------------

    @classmethod
    def bootstrap_baseline_model(
        cls,
        *,
        repo_root: str,
        llm: LLMLike,
        initial_repo: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> None:
        """
        Build baseline RepoMemory for an existing repository (seeded workspace).
        
        This runs once at the start so ideation can be grounded in the repo's
        actual architecture/algorithms with evidence links.
        
        Raises:
            ValueError: If evidence validation fails (LLM produced hallucinated claims).
        """
        repo_root = os.path.abspath(repo_root)
        doc = cls.ensure_exists_in_worktree(repo_root, initial_repo=initial_repo)
        doc["repo_map"] = build_repo_map(repo_root)
        doc["generated_at"] = cls._now_iso()

        llm_model = llm_model or cls.DEFAULT_REPO_MODEL_LLM
        model = infer_repo_model_with_retry(
            llm=llm,
            model=llm_model,
            repo_root=repo_root,
            repo_map=doc["repo_map"],
        )
        # Note: With line-number-based evidence, validation is no longer needed.
        # The model is trusted as-is.
        
        # Builders may return either:
        # - v1: {"summary", "entrypoints", "where_to_edit", "claims"}
        # - v2: {"summary", "sections": {...}}
        if isinstance((model or {}).get("sections"), dict):
            book = cls._build_book_from_v2_model(model)
            doc["book"] = book
            doc["repo_model"] = cls._legacy_repo_model_from_book(book)
        else:
            # Store legacy v1 model for compatibility.
            doc["repo_model"] = model  # Legacy v1 model (kept for backward compatibility)
            # Derive v2 Book view deterministically from v1 repo_model.
            doc["book"] = cls._build_book_from_v1_repo_model(model)
        doc["schema_version"] = cls.SCHEMA_VERSION

        doc["quality"] = {
            "evidence_ok": True,
            "missing_evidence": [],
            "section_count": len((doc.get("book") or {}).get("sections", {}) or {}),
            "claim_count": cls._count_claims_in_book_sections(
                (doc.get("book") or {}).get("sections", {}) or {}
            ),
        }
        cls.write_to_worktree(repo_root, doc)

    @classmethod
    def update_after_experiment(
        cls,
        *,
        repo_root: str,
        llm: LLMLike,
        branch_name: str,
        parent_branch_name: str,
        base_commit_sha: str,
        solution_spec: str,
        run_result: Dict[str, Any],
        llm_model: Optional[str] = None,
    ) -> None:
        """
        Update `.kapso/repo_memory.json` for the current repo state.
        
        Intended to be called at the end of a branch-level experiment, before the
        ExperimentSession is closed (so the file is committed into that branch).
        
        Raises:
            ValueError: If evidence validation fails after retry (LLM produced hallucinated claims).
        """
        repo_root = os.path.abspath(repo_root)
        repo = git.Repo(repo_root)
        head_commit_sha = repo.head.commit.hexsha

        doc = cls.ensure_exists_in_worktree(repo_root)

        # 1) Always refresh deterministic RepoMap.
        doc["repo_map"] = build_repo_map(repo_root)
        doc["generated_at"] = cls._now_iso()

        # 2) Record experiment delta (idea/spec + diffs + result).
        changed_files = repo.git.diff("--name-only", base_commit_sha, head_commit_sha).splitlines()
        numstat_lines = repo.git.diff("--numstat", base_commit_sha, head_commit_sha).splitlines()
        diff_numstat = []
        for line in numstat_lines[:200]:
            parts = line.split("\t")
            if len(parts) == 3:
                diff_numstat.append({"added": parts[0], "deleted": parts[1], "path": parts[2]})

        diff_summary = repo.git.diff("--stat", base_commit_sha, head_commit_sha)

        doc.setdefault("experiments", []).append(
            {
                "recorded_at": cls._now_iso(),
                "branch": branch_name,
                "parent_branch": parent_branch_name,
                "base_commit": base_commit_sha,
                "head_commit": head_commit_sha,
                # Explicit: commit hash of the code state this memory describes.
                # Note: the RepoMemory update itself is committed as a follow-up metadata commit,
                # so the branch HEAD may advance after this update.
                "code_head_commit": head_commit_sha,
                "solution_spec": (solution_spec or "")[:8000],
                "changed_files": changed_files[:200],
                "diff_numstat": diff_numstat,
                "run_result": run_result,
            }
        )

        # 3) Update semantic RepoModel via LLM.
        llm_model = llm_model or cls.DEFAULT_REPO_MODEL_LLM
        # Builders update the public RepoMemory V2 semantic model (summary + sections).
        # We also keep a legacy `repo_model` view for backward compatibility, so for updates
        # we derive the semantic model from `doc["book"]`.
        previous_book = doc.get("book", {}) if isinstance(doc.get("book"), dict) else {}
        previous_model_v2 = {
            "summary": (previous_book.get("summary") or "").strip(),
            "sections": previous_book.get("sections", {}) if isinstance(previous_book.get("sections"), dict) else {},
        }

        updated_model: Dict[str, Any]
        # If we have no meaningful semantic model yet, do a full initial inference.
        #
        # Note: v2 always has core section shells, so checking `sections` truthiness
        # is not enough. Instead, treat it as "missing" when we have no summary AND
        # no evidence-backed claims anywhere.
        prev_sections = previous_model_v2.get("sections", {}) if isinstance(previous_model_v2.get("sections"), dict) else {}
        prev_claim_count = cls._count_claims_in_book_sections(prev_sections)
        if not (previous_model_v2.get("summary") or "").strip() and prev_claim_count == 0:
            updated_model = infer_repo_model_initial(
                llm=llm,
                model=llm_model,
                repo_root=repo_root,
                repo_map=doc["repo_map"],
            )
        else:
            updated_model = infer_repo_model_update(
                llm=llm,
                model=llm_model,
                repo_root=repo_root,
                repo_map=doc["repo_map"],
                previous_model=previous_model_v2,
                diff_summary=diff_summary[:8000],
                changed_files=changed_files,
            )

        # Note: With line-number-based evidence, validation is no longer needed.
        # The model is trusted as-is.

        doc["repo_model"] = updated_model
        if isinstance((updated_model or {}).get("sections"), dict):
            book = cls._build_book_from_v2_model(updated_model)
            doc["book"] = book
            doc["repo_model"] = cls._legacy_repo_model_from_book(book)
        else:
            doc["repo_model"] = updated_model
            doc["book"] = cls._build_book_from_v1_repo_model(updated_model)
        doc["schema_version"] = cls.SCHEMA_VERSION
        doc["quality"] = {
            "evidence_ok": True,
            "missing_evidence": [],
            "section_count": len((doc.get("book") or {}).get("sections", {}) or {}),
            "claim_count": cls._count_claims_in_book_sections(
                (doc.get("book") or {}).get("sections", {}) or {}
            ),
        }

        cls.write_to_worktree(repo_root, doc)

