"""Learner sessions take their settings from config, and failures raise.

Pins the fixes for docs/plans/leeroopedia-learning-findings.md L1–L3 and L7: a
learner session carries the config's effort and no deadline unless one is
configured, and cannot execute the repository it reads or reach the network; a
failed phase stops the run instead of being logged and skipped, a prompt with a
missing variable is never sent, and the pipeline propagates an ingestor failure
instead of counting the pages that happened to come out as a success.
"""

import json
from types import SimpleNamespace

import pytest

from kapso.core.config import PLATFORM_CONFIG_PATH, load_config
from kapso.knowledge_base.learners import knowledge_learner_pipeline as pipeline_module
from kapso.knowledge_base.learners.ingestors import repo_ingestor as ingestor_module
from kapso.knowledge_base.learners.ingestors.research_ingestor import idea_ingestor
from kapso.knowledge_base.learners.merger import knowledge_merger as merger_module
from kapso.knowledge_base.search.base import WikiPage

PACKAGED = load_config(str(PLATFORM_CONFIG_PATH))
LEARNER = PACKAGED["modes"][PACKAGED["default_mode"]]["learner"]
# The whole built-in set of a learning session: file tools, nothing that
# runs code, reaches the network or delegates (pinned with --tools).
TOOLS = ["Read", "Write", "Edit", "Glob", "Grep"]


class FakeAgent:
    """Stands in for the Claude Code adapter; reports a killed session."""

    def __init__(self, config):
        self.config = config

    def initialize(self, workspace):
        self.workspace = workspace

    def generate_code(self, prompt, **kwargs):
        return SimpleNamespace(success=False, output="", error="terminated", metadata={})


@pytest.fixture
def agents(monkeypatch):
    created = []
    monkeypatch.setattr(
        ingestor_module.CodingAgentFactory, "create",
        staticmethod(lambda config: created.append(FakeAgent(config)) or created[-1]),
    )
    return created


def ingestor(tmp_path, **params):
    return ingestor_module.RepoIngestor(params={"wiki_dir": tmp_path, **params})


def test_ingestor_session_carries_config_effort_and_no_deadline(tmp_path, agents):
    ingestor(tmp_path)._initialize_agent(str(tmp_path))
    config = agents[0].config
    assert config.model == LEARNER["ingestor"]["model"]
    assert config.agent_specific["auth_mode"] == LEARNER["ingestor"]["auth_mode"]
    assert config.agent_specific["effort"] == LEARNER["ingestor"]["effort"]
    assert config.agent_specific["timeout"] is None
    assert config.agent_specific["builtin_tools"] == TOOLS
    assert config.agent_specific["strict_mcp_config"] is True

    ingestor(tmp_path, effort="low", timeout=60)._initialize_agent(str(tmp_path))
    assert agents[1].config.agent_specific["effort"] == "low"
    assert agents[1].config.agent_specific["timeout"] == 60


def test_research_ingestor_session_carries_config_effort_and_no_deadline(tmp_path, agents):
    idea_ingestor.IdeaIngestor(params={"wiki_dir": tmp_path})._initialize_agent(str(tmp_path))
    config = agents[0].config
    assert config.model == LEARNER["ingestor"]["model"]
    assert config.agent_specific["auth_mode"] == LEARNER["ingestor"]["auth_mode"]
    assert config.agent_specific["effort"] == LEARNER["ingestor"]["effort"]
    assert config.agent_specific["timeout"] is None
    assert config.agent_specific["builtin_tools"] == TOOLS
    assert config.agent_specific["strict_mcp_config"] is True


def test_research_phase_failure_stops_the_run(tmp_path, agents, monkeypatch):
    monkeypatch.setattr(idea_ingestor.IdeaIngestor, "_build_phase_prompt", lambda self, phase, **kw: "plan")
    ingestor = idea_ingestor.IdeaIngestor(params={"wiki_dir": tmp_path})
    ingestor._initialize_agent(str(tmp_path))
    with pytest.raises(RuntimeError, match="planning phase failed .* terminated"):
        ingestor._run_planning_phase("q", "https://example.test", "content")


def test_merger_session_carries_config_effort_and_no_deadline(tmp_path, agents):
    merger_module.KnowledgeMerger()._initialize_agent(tmp_path)
    config = agents[0].config
    assert config.model == LEARNER["merger"]["model"]
    assert config.agent_specific["auth_mode"] == LEARNER["merger"]["auth_mode"]
    assert config.agent_specific["effort"] == LEARNER["merger"]["effort"]
    assert config.agent_specific["timeout"] is None
    assert config.agent_specific["builtin_tools"] == TOOLS
    assert config.agent_specific["strict_mcp_config"] is True


def test_failed_phase_stops_the_run(tmp_path, agents, monkeypatch):
    monkeypatch.setattr(ingestor_module, "_load_prompt", lambda name: "map {repo_name}")
    repo = ingestor(tmp_path)
    repo._initialize_agent(str(tmp_path))
    with pytest.raises(RuntimeError, match="anchoring phase failed .* terminated"):
        repo._run_phase("anchoring", "Some_Repo", str(tmp_path))


def test_prompt_with_a_missing_variable_is_never_sent(tmp_path, monkeypatch):
    monkeypatch.setattr(ingestor_module, "_load_prompt", lambda name: "{repo_name} {no_such_variable}")
    with pytest.raises(KeyError, match="no_such_variable"):
        ingestor(tmp_path)._build_phase_prompt("anchoring_context", "Some_Repo", str(tmp_path))


def test_prompt_whose_wiki_structure_is_missing_is_never_sent(tmp_path, monkeypatch):
    def missing(page_type):
        raise FileNotFoundError(page_type)

    monkeypatch.setattr(ingestor_module, "load_wiki_structure", missing)
    with pytest.raises(FileNotFoundError, match="workflow"):
        ingestor(tmp_path)._build_phase_prompt("anchoring", "Some_Repo", str(tmp_path))


def test_pipeline_propagates_an_ingest_failure(tmp_path, monkeypatch):
    class FailingIngestor:
        def ingest(self, source):
            raise RuntimeError("git clone failed")

    monkeypatch.setattr(
        pipeline_module.IngestorFactory, "for_source",
        staticmethod(lambda source, **params: FailingIngestor()),
    )
    pipeline = pipeline_module.KnowledgePipeline(wiki_dir=tmp_path)
    with pytest.raises(RuntimeError, match="git clone failed"):
        pipeline.run(object())
    with pytest.raises(ValueError, match="No sources"):
        pipeline.run()


def test_pipeline_result_is_not_a_success_with_errors():
    # The old rule — success whenever any page came out — hid lost sources.
    assert pipeline_module.PipelineResult(total_pages_extracted=60, errors=["x"]).success is False
    assert pipeline_module.PipelineResult(total_pages_extracted=60).success is True


def merge_setup(tmp_path, monkeypatch, tool_calls, in_store):
    """A merger over an index whose store holds `in_store`, with a session
    that reports `tool_calls`; the proposed pages are already published into
    wiki_dir, as the ingestor does before the merge."""
    wiki = tmp_path / "wiki"
    (wiki / "principles").mkdir(parents=True, exist_ok=True)
    index = wiki / ".index"
    index.write_text(json.dumps({
        "version": "1.0", "created_at": "now", "data_source": str(wiki),
        "search_backend": "kg_graph_search", "backend_refs": {"weaviate_collection": "T"}, "page_count": 1,
    }))
    pages = [WikiPage(id=f"Principle/{name}", page_type="Principle", overview=name, content=f"== Overview ==\n{name}")
             for name in ("New", "Folded")]
    for page in pages:
        (wiki / "principles" / f"{page.id.split('/')[1]}.md").write_text(page.content)

    class FakeStore:
        def page_exists(self, page_id): return page_id in in_store
        def get_indexed_count(self): return len(in_store)
        def close(self): pass

    class MergeAgent:
        def __init__(self, config): pass
        def initialize(self, workspace): pass
        def generate_code(self, prompt, **kwargs):
            return SimpleNamespace(success=True, output="", error=None, metadata={"tool_calls": tool_calls})

    monkeypatch.setattr(merger_module.KnowledgeSearchFactory, "create", staticmethod(lambda *a, **k: FakeStore()))
    monkeypatch.setattr(merger_module.CodingAgentFactory, "create", staticmethod(lambda config: MergeAgent(config)))
    merger = merger_module.KnowledgeMerger(agent_config={"kg_index_path": str(index)})
    return merger, pages, wiki, index


def test_merge_result_comes_from_the_store_not_the_plan(tmp_path, monkeypatch):
    # Live run 2026-09-28: 41 pages indexed and 6 merged, reported as 0 and 0
    # because the plan file's headings differed from what a parser expected.
    calls = [
        {"name": "mcp__kg-graph-search__kg_index", "input": {"page_data": {"page_id": "Principle/New"}}},
        {"name": "mcp__kg-graph-search__kg_edit", "input": {"page_id": "Principle/Old", "updates": {}}},
    ]
    merger, pages, wiki, index = merge_setup(tmp_path, monkeypatch, calls, in_store={"Principle/Old", "Principle/New"})
    result = merger.merge(pages, wiki_dir=wiki, staging_dir=wiki)
    assert result.created == ["Principle/New"] and result.edited == ["Principle/Old"]
    # The folded page lives on in the page it was merged into: its file goes.
    assert (wiki / "principles" / "New.md").exists()
    assert not (wiki / "principles" / "Folded.md").exists()
    assert json.loads(index.read_text())["page_count"] == 2


def test_merge_fails_when_an_indexed_page_is_not_in_the_store(tmp_path, monkeypatch):
    calls = [{"name": "mcp__kg-graph-search__kg_index", "input": {"page_data": {"page_id": "Principle/New"}}}]
    merger, pages, wiki, _ = merge_setup(tmp_path, monkeypatch, calls, in_store=set())
    with pytest.raises(RuntimeError, match="does not hold"):
        merger.merge(pages, wiki_dir=wiki, staging_dir=wiki)
    # And a page that simply vanished, with nothing edited, is not "merged".
    merger, pages, wiki, _ = merge_setup(tmp_path, monkeypatch, [], in_store={"Principle/New"})
    with pytest.raises(RuntimeError, match="neither in the store nor merged"):
        merger.merge(pages, wiki_dir=wiki, staging_dir=wiki)


def test_merge_without_an_index_fails_when_the_index_cannot_be_built(tmp_path):
    def failing_builder(wiki_dir, save_to):
        raise RuntimeError("Weaviate is down")

    page = WikiPage(id="Principle/A", page_type="Principle", overview="a", content="== Overview ==\na")
    pipeline = pipeline_module.KnowledgePipeline(wiki_dir=tmp_path, index_builder=failing_builder)
    with pytest.raises(RuntimeError, match="Weaviate is down"):
        pipeline.merge_pages([page])
    # The page was written before the index failed: the failure is not silent
    # and the file is there for the retry.
    assert (tmp_path / "principles" / "A.md").exists()
    # Without a builder there is no way to make the pages findable.
    with pytest.raises(RuntimeError, match="no index_builder"):
        merger_module.KnowledgeMerger().merge([page], wiki_dir=tmp_path)
