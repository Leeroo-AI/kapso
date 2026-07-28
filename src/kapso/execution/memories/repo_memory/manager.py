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
    infer_repo_model_update,
    infer_repo_model_with_retry,
)


class RepoMemoryManager:
    """Own the single Book-shaped repository-memory document."""

    SCHEMA_VERSION = 2
    KAPSO_DIR = ".kapso"
    MEMORY_FILE = "repo_memory.json"
    MEMORY_REL_PATH = os.path.join(KAPSO_DIR, MEMORY_FILE)

    # Default model for repo-model inference.
    DEFAULT_REPO_MODEL_LLM = "utility"
    DEFAULT_FAILURE_POLICY = "warn"
    DEFAULT_MAX_RETRIES = 2
    FAILURE_POLICIES = {"warn", "fail"}

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
    def normalize_failure_policy(cls, policy: str) -> str:
        """Validate and normalize the optional-enrichment failure policy."""
        if not isinstance(policy, str):
            raise ValueError("RepoMemory failure policy must be 'warn' or 'fail'")
        normalized = policy.strip().lower()
        if normalized not in cls.FAILURE_POLICIES:
            raise ValueError("RepoMemory failure policy must be 'warn' or 'fail'")
        return normalized

    @classmethod
    def normalize_max_retries(cls, max_retries: int) -> int:
        """Validate the number of structured-response repair attempts."""
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise ValueError("RepoMemory max retries must be a non-negative integer")
        if max_retries < 0:
            raise ValueError("RepoMemory max retries must be a non-negative integer")
        return max_retries

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
        optional_ids = [
            sid for sid in sections.keys() if sid not in set(cls.CORE_SECTIONS)
        ]
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
    def _build_book_from_model(cls, model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build the canonical `book` from the LLM's repository-memory output.

        Expected model shape:
        {
          "summary": "...",
          "sections": { "core.architecture": {...}, ... }
        }
        """
        model = model or {}
        sections = (
            model.get("sections", {}) if isinstance(model.get("sections"), dict) else {}
        )
        sections = cls._ensure_core_sections_present(sections)
        toc = cls._build_toc_from_sections(sections)
        return {
            "summary": (model.get("summary") or "").strip(),
            "toc": toc,
            "sections": sections,
        }

    @classmethod
    def _require_document(cls, doc: Any) -> Dict[str, Any]:
        expected = {
            "schema_version",
            "generated_at",
            "repo_map",
            "book",
            "experiments",
            "quality",
        }
        if not isinstance(doc, dict) or set(doc) != expected:
            raise ValueError("repository memory fields are invalid")
        if doc["schema_version"] != cls.SCHEMA_VERSION:
            raise ValueError("repository memory schema is unsupported")
        if (
            not isinstance(doc["generated_at"], str)
            or not isinstance(doc["repo_map"], dict)
            or not isinstance(doc["experiments"], list)
            or not isinstance(doc["quality"], dict)
        ):
            raise ValueError("repository memory document is invalid")
        cls._require_book(doc)
        return doc

    @classmethod
    def _require_book(cls, doc: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(doc, dict) or doc.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError("repository memory schema is unsupported")
        book = doc.get("book")
        if (
            not isinstance(book, dict)
            or set(book) != {"summary", "toc", "sections"}
            or not isinstance(book["summary"], str)
            or not isinstance(book["toc"], list)
            or not isinstance(book["sections"], dict)
        ):
            raise ValueError("repository memory book is invalid")
        if book["toc"] != cls._build_toc_from_sections(book["sections"]):
            raise ValueError("repository memory table of contents is not canonical")
        return book

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
        return cls._require_document(doc)

    @classmethod
    def write_to_worktree(cls, repo_root: str, doc: Dict[str, Any]) -> None:
        """Write one validated memory document to a working tree."""
        cls._require_document(doc)
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
    ) -> Dict[str, Any]:
        """Ensure the canonical memory file exists, creating it when absent."""
        path = cls._memory_abs_path(repo_root)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return cls._require_document(json.load(f))

        repo_map = build_repo_map(repo_root)
        doc: Dict[str, Any] = {
            "schema_version": cls.SCHEMA_VERSION,
            "generated_at": cls._now_iso(),
            "repo_map": repo_map,
            "book": cls._build_book_from_model({"summary": "", "sections": {}}),
            "experiments": [],
            "quality": {
                "evidence_ok": False,
                "missing_evidence": [],
                "section_count": len(cls.CORE_SECTIONS),
                "claim_count": 0,
            },
        }
        cls._require_document(doc)
        cls.write_to_worktree(repo_root, doc)
        return doc

    # ---------------------------------------------------------------------
    # Git integration (read from branch without checkout)
    # ---------------------------------------------------------------------

    @classmethod
    def load_from_git_branch(
        cls, repo: git.Repo, branch_name: str
    ) -> Optional[Dict[str, Any]]:
        """Read `.kapso/repo_memory.json` from a given branch (no checkout)."""
        status, raw, error = repo.git.execute(
            ["git", "show", f"{branch_name}:{cls.MEMORY_REL_PATH}"],
            with_extended_output=True,
            with_exceptions=False,
        )
        if status == 128 and "does not exist" in error:
            return None
        if status != 0:
            raise ValueError(error.strip() or "could not read repository memory")
        return cls._require_document(json.loads(raw))

    # ---------------------------------------------------------------------
    # Prompt rendering
    # ---------------------------------------------------------------------

    @classmethod
    def render_summary_and_toc(cls, doc: Dict[str, Any], max_chars: int = 3000) -> str:
        """
        Render Summary + TOC (bounded) for prompt injection.

        Coding agents can read `.kapso/repo_memory.json` directly for details.
        """
        book = cls._require_book(doc)

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
        """Return canonical TOC section metadata."""
        return list(cls._require_book(doc)["toc"])

    @classmethod
    def get_section(
        cls, doc: Dict[str, Any], section_id: str, max_chars: int = 8000
    ) -> str:
        """
        Render a single section as human-readable text.

        This is intended for tool-style access and debugging.
        """
        book = cls._require_book(doc)
        sections = book["sections"]

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
                    quote_short = (
                        quote if len(quote) <= 200 else quote[:200] + "...(truncated)"
                    )
                    lines.append(f'  - evidence: {path}: "{quote_short}"')
            text = "\n".join(lines)
            return text[:max_chars]

        # Content section
        content = sec.get("content", None)
        if content is not None:
            # JSON is the most faithful representation for entrypoints/where-to-edit.
            text = json.dumps(content, indent=2, ensure_ascii=False)
            return text[:max_chars]

        return f"(empty section: {section_id})"[:max_chars]

    # ---------------------------------------------------------------------
    # Updating memory after an experiment
    # ---------------------------------------------------------------------

    @classmethod
    def bootstrap_baseline_model(
        cls,
        *,
        repo_root: str,
        llm: LLMLike,
        llm_model: Optional[str] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        """
        Build baseline RepoMemory for an existing repository (seeded workspace).

        This runs once at the start so ideation can be grounded in the repo's
        actual architecture/algorithms with evidence links.

        Raises:
            ValueError: If configuration is invalid or structured-response
                repair is exhausted.
        """
        repo_root = os.path.abspath(repo_root)
        max_retries = cls.normalize_max_retries(max_retries)
        doc = cls.ensure_exists_in_worktree(repo_root)
        doc["repo_map"] = build_repo_map(repo_root)
        doc["generated_at"] = cls._now_iso()

        llm_model = llm_model or cls.DEFAULT_REPO_MODEL_LLM
        model = infer_repo_model_with_retry(
            llm=llm,
            model=llm_model,
            repo_root=repo_root,
            repo_map=doc["repo_map"],
            max_retries=max_retries,
        )
        doc["book"] = cls._build_book_from_model(model)

        doc["quality"] = {
            "evidence_ok": True,
            "missing_evidence": [],
            "section_count": len((doc.get("book") or {}).get("sections", {}) or {}),
            "claim_count": cls._count_claims_in_book_sections(
                (doc.get("book") or {}).get("sections", {}) or {}
            ),
        }
        cls._require_document(doc)
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
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        """
        Update `.kapso/repo_memory.json` for the current repo state.

        Intended to be called at the end of a branch-level experiment, before the
        ExperimentSession is closed (so the file is committed into that branch).

        Raises:
            ValueError: If configuration is invalid or structured-response
                repair is exhausted.
        """
        repo_root = os.path.abspath(repo_root)
        max_retries = cls.normalize_max_retries(max_retries)
        repo = git.Repo(repo_root)
        head_commit_sha = repo.head.commit.hexsha

        doc = cls.ensure_exists_in_worktree(repo_root)

        # 1) Always refresh deterministic RepoMap.
        doc["repo_map"] = build_repo_map(repo_root)
        doc["generated_at"] = cls._now_iso()

        # 2) Record experiment delta (idea/spec + diffs + result).
        changed_files = repo.git.diff(
            "--name-only", base_commit_sha, head_commit_sha
        ).splitlines()
        numstat_lines = repo.git.diff(
            "--numstat", base_commit_sha, head_commit_sha
        ).splitlines()
        diff_numstat = []
        for line in numstat_lines[:200]:
            parts = line.split("\t")
            if len(parts) == 3:
                diff_numstat.append(
                    {"added": parts[0], "deleted": parts[1], "path": parts[2]}
                )

        diff_summary = repo.git.diff("--stat", base_commit_sha, head_commit_sha)

        doc["experiments"].append(
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

        # 3) Update the semantic Book through the configured model.
        llm_model = llm_model or cls.DEFAULT_REPO_MODEL_LLM
        previous_book = cls._require_book(doc)
        previous_model = {
            "summary": previous_book["summary"].strip(),
            "sections": previous_book["sections"],
        }

        updated_model: Dict[str, Any]
        # If we have no meaningful semantic model yet, do a full initial inference.
        #
        # The Book always has core section shells, so checking `sections` truthiness
        # is not enough. Instead, treat it as "missing" when we have no summary AND
        # no evidence-backed claims anywhere.
        previous_sections = previous_model["sections"]
        previous_claim_count = cls._count_claims_in_book_sections(previous_sections)
        if (
            not previous_model["summary"]
            and previous_claim_count == 0
        ):
            updated_model = infer_repo_model_with_retry(
                llm=llm,
                model=llm_model,
                repo_root=repo_root,
                repo_map=doc["repo_map"],
                max_retries=max_retries,
            )
        else:
            updated_model = infer_repo_model_update(
                llm=llm,
                model=llm_model,
                repo_root=repo_root,
                repo_map=doc["repo_map"],
                previous_model=previous_model,
                diff_summary=diff_summary[:8000],
                changed_files=changed_files,
                max_retries=max_retries,
            )

        doc["book"] = cls._build_book_from_model(updated_model)
        doc["quality"] = {
            "evidence_ok": True,
            "missing_evidence": [],
            "section_count": len((doc.get("book") or {}).get("sections", {}) or {}),
            "claim_count": cls._count_claims_in_book_sections(
                (doc.get("book") or {}).get("sections", {}) or {}
            ),
        }

        cls._require_document(doc)
        cls.write_to_worktree(repo_root, doc)
