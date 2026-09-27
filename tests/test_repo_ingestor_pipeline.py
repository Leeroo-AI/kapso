"""The repo ingestor's phase plan: what runs, and what stops it.

Pins docs/plans/leeroopedia-learning-findings.md L4–L6: the validator checks
what serving needs from every page (wikitext, an Overview with text, no
temporary clone paths, namespaced links) and reads index entries from their
file links; publishing the workflows as repositories is a config switch that
runs after validation and fails loud when a repository is not created, while
a Workflow without a published repository is a warning; and the orphan-mining
agent sessions run only when the deterministic triage left them work.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from kapso.core.config import PLATFORM_CONFIG_PATH, load_config
from kapso.core.preflight import learn_knowledge_requirements
from kapso.knowledge_base.learners.ingestors import repo_ingestor as ingestor_module
from kapso.knowledge_base.learners.ingestors.repo_ingestor.context_builder import (
    orphan_candidate_counts,
)
from kapso.knowledge_base.learners.ingestors.repo_ingestor.wiki_validator import (
    validate_wiki_directory,
)
from kapso.knowledge_base.search.kg_graph_search import _extract_overview

PACKAGED = load_config(str(PLATFORM_CONFIG_PATH))

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


# --------------------------------------------------------------------------
# Publishing workflows as repositories (L5)
# --------------------------------------------------------------------------

WORKFLOW_PAGE = """{{{{PageInfo|type=Workflow|title=Repo_Foo}}}}
== Overview ==
Runs foo.

== GitHub URL ==
{url}
"""


def workflow_wiki(tmp_path, url="[https://github.com/PENDING Pending Repository Build]"):
    (tmp_path / "workflows").mkdir()
    (tmp_path / "workflows" / "Repo_Foo.md").write_text(WORKFLOW_PAGE.format(url=url))
    (tmp_path / "_WorkflowIndex.md").write_text(
        "## Workflow: Repo_Foo\n\n**File:** [→](./workflows/Repo_Foo.md)\n"
    )
    return tmp_path


class PublishingAgent:
    """Stands in for the builder session: writes the result file it is given, or not."""

    def __init__(self, url=None):
        self.url = url
        self.prompts = []

    def generate_code(self, prompt, **kwargs):
        self.prompts.append(prompt)
        result_file = Path(prompt.split("**Result File**: `", 1)[1].split("`", 1)[0])
        if self.url:
            result_file.write_text(self.url)
        return SimpleNamespace(success=True, output="", error=None, metadata={})


def test_publishing_is_off_by_default_and_a_placeholder_url_is_a_warning(tmp_path):
    assert PACKAGED["modes"][PACKAGED["default_mode"]]["learner"]["ingestor"]["publish_workflows"] is False
    assert ingestor_module.RepoIngestor(params={"wiki_dir": tmp_path})._publish_workflows is False
    report = validate_wiki_directory(workflow_wiki(tmp_path))
    assert report.errors == []
    assert any("no published repository URL" in warning for warning in report.warnings)


def test_a_repository_that_is_not_created_fails_the_publishing_phase(tmp_path):
    ingestor = ingestor_module.RepoIngestor(params={"wiki_dir": tmp_path, "publish_workflows": True})
    ingestor._wiki_dir = workflow_wiki(tmp_path)
    ingestor._agent = PublishingAgent(url=None)
    with pytest.raises(RuntimeError, match="publishing failed for 1 of 1"):
        ingestor._run_repo_builder_phase("Repo", "https://example.test/repo")
    # The prompt carries no token: gh reads GH_TOKEN itself.
    assert "GH_TOKEN=" not in ingestor._agent.prompts[0]
    assert "PENDING" in (tmp_path / "workflows" / "Repo_Foo.md").read_text()


def test_a_created_repository_is_written_into_the_workflow_page(tmp_path):
    ingestor = ingestor_module.RepoIngestor(params={"wiki_dir": tmp_path, "publish_workflows": True})
    ingestor._wiki_dir = workflow_wiki(tmp_path)
    ingestor._agent = PublishingAgent(url="https://github.com/org/repo-foo")
    ingestor._run_repo_builder_phase("Repo", "https://example.test/repo")
    page = (tmp_path / "workflows" / "Repo_Foo.md").read_text()
    assert "[https://github.com/org/repo-foo Workflow Repository]" in page
    assert validate_wiki_directory(tmp_path).warnings == []


def test_preflight_requires_the_gh_token_only_when_publishing(monkeypatch):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    labels = lambda publishing: {
        item.label: item.required
        for item in learn_knowledge_requirements(PACKAGED, skip_merge=True, publish_workflows=publishing)
    }
    assert "GH_TOKEN" not in labels(False)
    assert labels(True)["GH_TOKEN"] is True


# --------------------------------------------------------------------------
# What the validator requires of every page (L4)
# --------------------------------------------------------------------------

GOOD_PRINCIPLE = """{{PageInfo|type=Principle|title=Repo_Rule}}
== Overview ==
A rule with text.

=== Description ===
More text.

== Related Pages ==
* [[implemented_by::Implementation:Repo_Rule_Impl]]
"""
GOOD_IMPLEMENTATION = """{{PageInfo|type=Implementation|title=Repo_Rule_Impl}}
== Overview ==
The code for the rule, at src/rule.py.
"""


def wiki(tmp_path, principle=GOOD_PRINCIPLE):
    for subdir in ("principles", "implementations"):
        (tmp_path / subdir).mkdir(exist_ok=True)
    (tmp_path / "principles" / "Repo_Rule.md").write_text(principle)
    (tmp_path / "implementations" / "Repo_Rule_Impl.md").write_text(GOOD_IMPLEMENTATION)
    (tmp_path / "_PrincipleIndex.md").write_text("| Page | File |\n|---|---|\n| Repo_Rule | [→](./principles/Repo_Rule.md) |\n")
    (tmp_path / "_ImplementationIndex.md").write_text("| 1 | Repo_Rule_Impl | [→](./implementations/Repo_Rule_Impl.md) |\n")
    return validate_wiki_directory(tmp_path)


def test_a_well_formed_wiki_passes_with_no_warnings(tmp_path):
    report = wiki(tmp_path)
    assert report.errors == [] and report.warnings == []


@pytest.mark.parametrize("page, complaint", [
    ("## Overview\nMarkdown page.\n", "no wikitext section headings"),
    ("== Summary ==\nOld layout.\n\n== Related Pages ==\n[[implemented_by::Implementation:Repo_Rule_Impl]]\n", "missing the == Overview =="),
    ("== Overview ==\n\n== Related Pages ==\n[[implemented_by::Implementation:Repo_Rule_Impl]]\n", "nothing to embed"),
    (GOOD_PRINCIPLE.replace("More text.", "See /tmp/kapso_repo_ab12cd/src/rule.py."), "temporary clone path"),
    (GOOD_PRINCIPLE.replace("More text.", "See [[Repo_Rule_Impl]] and [[Other|the other]]."), "2 link(s) without a namespace ([[Repo_Rule_Impl]], [[Other]])"),
], ids=["markdown", "no-overview", "empty-overview", "temp-path", "plain-links"])
def test_the_validator_rejects_pages_serving_cannot_use(tmp_path, page, complaint):
    report = wiki(tmp_path, principle=page)
    assert any(complaint in error for error in report.errors), report.errors


def test_index_entries_come_from_file_links_not_table_columns(tmp_path):
    # A steps table whose first column is a number, and a summary table of
    # short names, used to produce false "missing from index" warnings.
    (tmp_path / "workflows").mkdir()
    (tmp_path / "workflows" / "Repo_Foo.md").write_text(WORKFLOW_PAGE.format(url="[https://github.com/o/r Workflow Repository]"))
    (tmp_path / "_WorkflowIndex.md").write_text(
        "| Workflow | Steps |\n|---|---|\n| Foo | 3 |\n\n## Workflow: Repo_Foo\n\n**File:** [→](./workflows/Repo_Foo.md)\n\n"
        "| # | Step | API |\n|---|---|---|\n| 1 | Load | load() |\n"
    )
    assert validate_wiki_directory(tmp_path).warnings == []


def test_the_parser_reads_the_older_layouts_first_section_as_the_overview():
    text = "{{PageInfo|type=Principle|title=X}}\n== Metadata ==\n{| table |}\n\n== Summary ==\nThe first real section.\n\n== Usage ==\nLater.\n"
    assert _extract_overview(text) == "The first real section."
    assert _extract_overview("== Overview ==\nProper.\n\n== Summary ==\nNot this.\n") == "Proper."
