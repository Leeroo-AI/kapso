"""Learner sessions take their settings from config, and failures raise.

Pins the fixes for docs/plans/leeroopedia-learning-findings.md L1–L3: a learner
session carries the config's effort and no deadline unless one is configured, a
failed phase stops the run instead of being logged and skipped, a prompt with a
missing variable is never sent, and the pipeline propagates an ingestor failure
instead of counting the pages that happened to come out as a success.
"""

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
    return ingestor_module.RepoIngestor(params={"wiki_dir": tmp_path, "github_pat": "x", **params})


def test_ingestor_session_carries_config_effort_and_no_deadline(tmp_path, agents):
    ingestor(tmp_path)._initialize_agent(str(tmp_path))
    config = agents[0].config
    assert config.model == LEARNER["ingestor"]["model"]
    assert config.agent_specific["auth_mode"] == LEARNER["ingestor"]["auth_mode"]
    assert config.agent_specific["effort"] == LEARNER["ingestor"]["effort"]
    assert config.agent_specific["timeout"] is None

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


def test_merge_without_an_index_fails_when_the_index_cannot_be_built(tmp_path, monkeypatch):
    class FakeKapso:
        def index_kg(self, wiki_dir, save_to):
            raise RuntimeError("Weaviate is down")

    monkeypatch.setattr("kapso.kapso.Kapso", FakeKapso)
    page = WikiPage(id="Principle/A", page_type="Principle", overview="a", content="== Overview ==\na")
    with pytest.raises(RuntimeError, match="Weaviate is down"):
        merger_module.KnowledgeMerger().merge([page], wiki_dir=tmp_path)
    # The page was written before the index failed: the failure is not silent
    # and the file is there for the retry.
    assert (tmp_path / "principles" / "A.md").exists()
