"""
RepoMemory manager
=================

This class owns persistence + update logic for repository memory.

Key guarantee:
- If a Praxium experiment continues from a branch, the memory file committed
  in that branch is the memory of the code it starts from.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import git

from src.repo_memory.builders import (
    LLMLike,
    build_repo_map,
    infer_repo_model_initial,
    infer_repo_model_update,
    infer_repo_model_with_retry,
    validate_evidence,
)


class RepoMemoryManager:
    SCHEMA_VERSION = 1
    PRAXIUM_DIR = ".praxium"
    MEMORY_FILE = "repo_memory.json"
    MEMORY_REL_PATH = os.path.join(PRAXIUM_DIR, MEMORY_FILE)

    # Default model for repo-model inference.
    DEFAULT_REPO_MODEL_LLM = "gpt-4o-mini"

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
        os.makedirs(os.path.join(repo_root, cls.PRAXIUM_DIR), exist_ok=True)

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
            return json.load(f)

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
        seed_repo_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ensure the memory file exists. If missing, create a minimal skeleton.
        
        Note: skeleton contains RepoMap but may omit RepoModel until inference.
        """
        existing = cls.load_from_worktree(repo_root)
        if existing is not None:
            return existing

        repo_map = build_repo_map(repo_root)
        doc: Dict[str, Any] = {
            "schema_version": cls.SCHEMA_VERSION,
            "generated_at": cls._now_iso(),
            "baseline": {
                "seed_repo_path": seed_repo_path,
            },
            "repo_map": repo_map,
            "repo_model": {
                "summary": "",
                "entrypoints": [],
                "where_to_edit": [],
                "claims": [],
            },
            "experiments": [],
            "quality": {
                "evidence_ok": False,
                "missing_evidence": [],
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
        """Read `.praxium/repo_memory.json` from a given branch (no checkout)."""
        try:
            raw = repo.git.show(f"{branch_name}:{cls.MEMORY_REL_PATH}")
        except git.GitCommandError:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    # ---------------------------------------------------------------------
    # Prompt rendering
    # ---------------------------------------------------------------------

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
        seed_repo_path: Optional[str] = None,
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
        doc = cls.ensure_exists_in_worktree(repo_root, seed_repo_path=seed_repo_path)
        doc["repo_map"] = build_repo_map(repo_root)
        doc["generated_at"] = cls._now_iso()

        llm_model = llm_model or cls.DEFAULT_REPO_MODEL_LLM
        model = infer_repo_model_with_retry(
            llm=llm,
            model=llm_model,
            repo_root=repo_root,
            repo_map=doc["repo_map"],
        )
        check = validate_evidence(repo_root, model)
        if not check.ok:
            raise ValueError(
                f"RepoMemory bootstrap failed: evidence validation failed after retries.\n"
                f"Missing evidence: {check.missing[:10]}"
            )
        
        doc["repo_model"] = model
        doc["quality"] = {
            "evidence_ok": True,
            "missing_evidence": [],
            "claim_count": len((model or {}).get("claims", []) or []),
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
        Update `.praxium/repo_memory.json` for the current repo state.
        
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
                "solution_spec": (solution_spec or "")[:8000],
                "changed_files": changed_files[:200],
                "diff_numstat": diff_numstat,
                "run_result": run_result,
            }
        )

        # 3) Update semantic RepoModel via LLM.
        llm_model = llm_model or cls.DEFAULT_REPO_MODEL_LLM
        previous_model = (doc.get("repo_model") or {}) if isinstance(doc.get("repo_model"), dict) else {}

        updated_model: Dict[str, Any]
        # If we have no model yet, do a full initial inference.
        if not previous_model.get("summary") and not previous_model.get("claims"):
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
                previous_model=previous_model,
                diff_summary=diff_summary[:8000],
                changed_files=changed_files,
            )

        # Evidence validation is the hard quality gate.
        check = validate_evidence(repo_root, updated_model)
        if not check.ok:
            # Retry once with a full rebuild (more robust than delta update).
            rebuilt = infer_repo_model_initial(
                llm=llm,
                model=llm_model,
                repo_root=repo_root,
                repo_map=doc["repo_map"],
            )
            check = validate_evidence(repo_root, rebuilt)
            if not check.ok:
                raise ValueError(
                    f"RepoMemory update failed: evidence validation failed after retry.\n"
                    f"Missing evidence: {check.missing[:10]}"
                )
            updated_model = rebuilt

        doc["repo_model"] = updated_model
        doc["quality"] = {
            "evidence_ok": True,
            "missing_evidence": [],
            "claim_count": len((updated_model or {}).get("claims", []) or []),
        }

        cls.write_to_worktree(repo_root, doc)

