"""The repo ingestor's phase plan: what runs, and what stops it.

Pins docs/plans/leeroopedia-learning-findings.md L6: the orphan-mining agent
sessions run only when the deterministic triage left them work.
"""

from pathlib import Path

import pytest

from kapso.knowledge_base.learners.ingestors import repo_ingestor as ingestor_module
from kapso.knowledge_base.learners.ingestors.repo_ingestor.context_builder import (
    orphan_candidate_counts,
)

CANDIDATES = """# Orphan Candidates: Repo

## AUTO_KEEP (Must Document)

| # | File | Lines | Rule | Status |
|---|------|-------|------|--------|
{auto_keep}

## MANUAL_REVIEW (Agent Evaluates)

| # | File | Lines | Purpose | Decision | Reasoning |
|---|------|-------|---------|----------|-----------|
{manual_review}
"""
NONE_KEEP = "| — | (none) | — | — | — |"
NONE_REVIEW = "| — | (none) | — | — | — | — |"


def candidates_file(path: Path, auto_keep=NONE_KEEP, manual_review=NONE_REVIEW) -> Path:
    path.write_text(CANDIDATES.format(auto_keep=auto_keep, manual_review=manual_review))
    return path


def test_orphan_candidate_counts(tmp_path):
    path = tmp_path / "_orphan_candidates.md"
    assert orphan_candidate_counts(candidates_file(path)) == (0, 0)
    assert orphan_candidate_counts(candidates_file(
        path, manual_review="| 1 | `src/a.py` | 120 | helpers | ⬜ PENDING | — |",
    )) == (1, 0)
    assert orphan_candidate_counts(candidates_file(
        path,
        auto_keep="| 1 | `src/big.py` | 400 | K1 | ⬜ TODO |",
        manual_review="| 1 | `src/a.py` | 120 | helpers | ✅ APPROVED | useful |\n"
                      "| 2 | `src/b.py` | 90 | glue | ❌ REJECTED | trivial |",
    )) == (0, 2)


@pytest.fixture
def mining(tmp_path, monkeypatch):
    """A RepoIngestor whose triage writes a given candidates file and whose
    agent phases are recorded instead of run."""
    ingestor = ingestor_module.RepoIngestor(params={"wiki_dir": tmp_path})
    ingestor._wiki_dir = tmp_path
    phases = []
    monkeypatch.setattr(ingestor, "_run_phase", lambda phase, *args: phases.append(phase))

    def run(auto_keep=NONE_KEEP, manual_review=NONE_REVIEW, on_review=None):
        path = tmp_path / "_orphan_candidates.md"
        monkeypatch.setattr(
            ingestor_module, "generate_orphan_candidates",
            lambda **kwargs: candidates_file(path, auto_keep, manual_review),
        )
        if on_review:
            monkeypatch.setattr(ingestor, "_run_phase", lambda phase, *args: (
                phases.append(phase), phase == "orphan_review" and on_review(path)))
        phases.clear()
        ingestor._run_orphan_mining("Repo", tmp_path, "https://example.test/repo", "main")
        return list(phases)

    return run


def test_orphan_phases_skip_when_triage_finds_nothing(mining):
    assert mining() == []


def test_orphan_review_runs_only_for_pending_files_and_create_only_for_approved(mining):
    # Review pending, agent rejects: no page work follows.
    assert mining(
        manual_review="| 1 | `src/a.py` | 120 | helpers | ⬜ PENDING | — |",
        on_review=lambda path: candidates_file(
            path, manual_review="| 1 | `src/a.py` | 120 | helpers | ❌ REJECTED | trivial |"),
    ) == ["orphan_review"]
    # AUTO_KEEP files need pages without any review.
    assert mining(auto_keep="| 1 | `src/big.py` | 400 | K1 | ⬜ TODO |") == [
        "orphan_create", "orphan_audit",
    ]
