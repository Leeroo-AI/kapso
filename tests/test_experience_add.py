"""`Kapso.add_experience` (wp3-experience.md, change 7): `mine` creates
the bank, `leeroopedia` is the key's presence, a repository or a question
is ingested into a new index of the experience home with its own backend
namespace and its record; an existing or reserved name is refused; a
build with an index loaded connects the search from the record and mounts
the search gates over it; nothing loaded means the null search and no
backend built anywhere."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import kapso.experience.ingest as ingest_module
from kapso.experience import ExperienceError, IngestError
from kapso.experience.ingest import collection_name
from kapso.kapso import Kapso
from kapso.knowledge_base.types import ResearchFindings
from tests.handles_support import facade
from tests.test_experience_reference import settings_for
from tests.test_kapso_facade_learn import _git_workspace, facade_config


class FakeSearch:
    def __init__(self, params, valid=True):
        self.params = params
        self.indexed = None
        self.valid = valid
        self.closed = False

    def index(self, data):
        self.indexed = data

    def get_backend_refs(self):
        return {"weaviate_collection": self.params["weaviate_collection"], "corpus": self.params["corpus"]}

    def get_indexed_count(self):
        return 2

    def validate_backend_data(self):
        return self.valid

    def is_enabled(self):
        return True

    def close(self):
        self.closed = True


@pytest.fixture
def ingestion(monkeypatch, tmp_path):
    """The pipeline and the search backend stubbed at their boundaries:
    the pipeline writes one page and calls the index builder, the factory
    hands back a fake that records its parameters."""
    captured = {"searches": []}

    class FakePipeline:
        def __init__(self, wiki_dir, ingestor_params, merger_params, index_builder):
            captured.update(wiki_dir=Path(wiki_dir), ingestor_params=ingestor_params, index_builder=index_builder)

        def run(self, *sources, status=None):
            captured["sources"] = sources
            wiki = captured["wiki_dir"]
            (wiki / "Workflow").mkdir(parents=True)
            (wiki / "Workflow" / "page.md").write_text("# a page\n")
            captured["index_builder"](wiki_dir=str(wiki), save_to=str(wiki / ".index"))
            return SimpleNamespace(total_pages_extracted=2, created=2, edited=0, errors=[])

    def create(search_type, params, preset=None):
        search = FakeSearch(params)
        captured["searches"].append((search_type, search))
        return search

    monkeypatch.setattr(ingest_module, "KnowledgePipeline", FakePipeline)
    monkeypatch.setattr(ingest_module, "KnowledgeSearchFactory", SimpleNamespace(create=create))
    monkeypatch.setattr("kapso.kapso.run_preflight", lambda *a, **k: captured.setdefault("preflight", []).append(k))
    monkeypatch.chdir(tmp_path)
    return captured


def _kapso(tmp_path):
    kapso = facade()
    kapso._experience_settings = settings_for(tmp_path)
    return kapso


def test_a_repository_is_ingested_into_a_new_index_with_its_own_namespace(tmp_path, ingestion, capsys):
    kapso = _kapso(tmp_path)
    assert kapso.add_experience(repo="https://github.com/org/x", name="org-x") == "org-x"
    record_path = Path(settings_for(tmp_path)["indexes"]) / "org-x" / "index.json"
    record = json.loads(record_path.read_text())
    assert record["name"] == "org-x" and record["source"] == "https://github.com/org/x" and record["page_count"] == 2
    assert record["search_backend"] == "kg_graph_search" and len(record["id"]) == 12
    assert record["backend_refs"] == {"weaviate_collection": collection_name("org-x", record["id"]), "corpus": record["id"]}
    assert collection_name("org-x", record["id"]) == f"KapsoOrgX{record['id']}"
    # the pages live under the index, the backend was built with the namespace and closed
    assert ingestion["wiki_dir"] == record_path.parent / "wiki"
    [(search_type, search)] = ingestion["searches"]
    assert search_type == "kg_graph_search" and search.params["corpus"] == record["id"] and search.closed
    assert str(search.indexed.wiki_dir) == str(ingestion["wiki_dir"])
    assert [type(s).__name__ for s in ingestion["sources"]] == ["Repo"]
    assert ingestion["preflight"] == [{"research": False}]
    assert "Experience added: org-x" in capsys.readouterr().out
    # the index loads at its id and an existing name is refused
    source = kapso.load_experience("org-x")
    assert (source.kind, source.head, source.cards) == ("index", record["id"], 2)
    with pytest.raises(IngestError, match="the index `org-x` exists"):
        kapso.add_experience(repo="https://github.com/org/y", name="org-x")


def test_a_question_is_researched_then_ingested(tmp_path, ingestion, monkeypatch):
    asked = []

    class FakeResearcher:
        def __init__(self, config_path=None):
            asked.append(("init", config_path))

        def research(self, question, *, mode, depth):
            asked.append((question, mode, depth))
            return ResearchFindings(query=question, ideas=["idea-a", "idea-b"], implementations=["implementation-c"])

    monkeypatch.setattr(ingest_module, "Researcher", FakeResearcher)
    kapso = _kapso(tmp_path)
    assert kapso.add_experience(research="QLoRA memory", name="qlora") == "qlora"
    # the modes and the depth are the config's; the findings' typed
    # sources, not the wrapper, reach the pipeline
    assert asked == [("init", kapso.config_path), ("QLoRA memory", ["idea", "implementation"], "deep")]
    assert ingestion["sources"] == ("idea-a", "idea-b", "implementation-c")
    assert ingestion["preflight"] == [{"research": True}]
    record = json.loads((Path(settings_for(tmp_path)["indexes"]) / "qlora" / "index.json").read_text())
    assert record["source"] == "QLoRA memory"


def test_a_paper_is_read_whole_then_ingested(tmp_path, ingestion):
    paper = tmp_path / "paper.md"
    paper.write_text("# Adapters\nthe whole paper\n")
    kapso = _kapso(tmp_path)
    assert kapso.add_experience(paper=str(paper), name="adapters") == "adapters"
    [report] = ingestion["sources"]
    assert report.content == "# Adapters\nthe whole paper\n"
    assert ingestion["preflight"] == [{"research": False}]
    record = json.loads((Path(settings_for(tmp_path)["indexes"]) / "adapters" / "index.json").read_text())
    assert record["source"] == str(paper)
    with pytest.raises(ValueError, match="exactly one of"):
        kapso.add_experience(paper=str(paper), repo="https://github.com/org/x", name="again")


def test_the_bank_and_the_hosted_gate_are_added_by_reference(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)   # no .env in reach: the key's presence is the environment's
    kapso = _kapso(tmp_path)
    assert kapso.add_experience("mine") == "mine"
    home = Path(settings_for(tmp_path)["bank_home"])
    assert (home / "HEAD").is_file() and "created your bank" in capsys.readouterr().out
    assert kapso.add_experience("mine") == "mine" and "your bank exists" in capsys.readouterr().out
    monkeypatch.delenv("LEEROOPEDIA_API_KEY", raising=False)
    with pytest.raises(ExperienceError, match="LEEROOPEDIA_API_KEY"):
        kapso.add_experience("leeroopedia")
    monkeypatch.setenv("LEEROOPEDIA_API_KEY", "kpsk_test")
    monkeypatch.setattr("kapso.kapso.probe_hosted_gate", lambda settings: "Leeroopedia")
    assert kapso.add_experience("leeroopedia") == "leeroopedia"
    assert "the gate answers as Leeroopedia" in capsys.readouterr().out


def test_what_cannot_be_added_is_refused_by_name(tmp_path, ingestion):
    kapso = _kapso(tmp_path)
    with pytest.raises(ValueError, match="exactly one of"):
        kapso.add_experience()
    with pytest.raises(ValueError, match="exactly one of"):
        kapso.add_experience("mine", repo="u")
    with pytest.raises(ValueError, match="needs no add"):
        kapso.add_experience("acme/kapso-bank")
    with pytest.raises(IngestError, match="an index needs a name"):
        kapso.add_experience(repo="https://github.com/org/x")
    with pytest.raises(IngestError, match="reserved reference"):
        kapso.add_experience(repo="https://github.com/org/x", name="mine")
    with pytest.raises(IngestError, match="not an index name"):
        kapso.add_experience(repo="https://github.com/org/x", name="Org X")
    assert ingestion["searches"] == []


def _build_with(tmp_path, monkeypatch, search):
    """A build through a fake orchestrator that captures what it was
    given, with the index's search stubbed where the facade connects it."""
    config_path = facade_config(tmp_path)
    _git_workspace(tmp_path / "ws")
    captured = {}

    class FakeOrchestrator:
        def __init__(self, handler, **kwargs):
            captured.update(kwargs)
            self.search_strategy = SimpleNamespace(
                workspace=SimpleNamespace(workspace_dir=str(tmp_path / "ws")),
                checkout_to_best_experiment_branch=lambda: "main",
                get_deliverable_experiment=lambda: None,
                get_deliverable_score=lambda: None,
                get_experiment_history=lambda best_last=True: [],
                iteration_evaluations=[],
                ideation_gates=["research", "idea", "code"],
            )
            self.operation_status = SimpleNamespace(path=str(tmp_path / "ws" / ".kapso" / "status.json"))

        def solve(self, **kwargs):
            result = SimpleNamespace(
                final_feedback=None, iterations_run=0, cumulative_iterations=0, total_cost=0.0,
                stopped_reason="max_iterations", stop_detail=None, best_experiment=None,
                platform_budget=None, requests=[],
            )
            # the delivery runs inside solve(), as the real orchestrator runs it
            kwargs["finish"](result)
            return result

    monkeypatch.setattr("kapso.kapso.OrchestratorAgent", FakeOrchestrator)
    monkeypatch.setattr("kapso.kapso.search_for", lambda record, config: search)
    monkeypatch.setattr("kapso.kapso.run_preflight", lambda *a, **k: None)
    monkeypatch.chdir(tmp_path)
    kapso = Kapso(config=config_path)
    kapso._experience_settings = {**settings_for(tmp_path), "bank_home": str(tmp_path / "bank-home.git")}
    return kapso, captured


def test_a_build_with_an_index_loaded_connects_its_search_from_the_record(tmp_path, monkeypatch, ingestion):
    search = FakeSearch({"weaviate_collection": "c", "corpus": "i"})
    kapso, captured = _build_with(tmp_path, monkeypatch, search)
    kapso.add_experience(repo="https://github.com/org/x", name="org-x")
    kapso.load_experience("org-x")
    kapso.build("test goal", output=str(tmp_path / "ws"))
    assert captured["knowledge_search"] is search
    overrides = captured["strategy_params_overrides"]
    assert overrides["kg_index_path"] == str(Path(settings_for(tmp_path)["indexes"]) / "org-x" / "index.json")
    assert "bank_serving" not in overrides and "hosted" not in overrides


def test_a_build_refuses_an_index_whose_stores_were_wiped(tmp_path, monkeypatch, ingestion):
    kapso, _captured = _build_with(tmp_path, monkeypatch, FakeSearch({"weaviate_collection": "c", "corpus": "i"}, valid=False))
    kapso.add_experience(repo="https://github.com/org/x", name="org-x")
    kapso.load_experience("org-x")
    with pytest.raises(ExperienceError, match="has no pages in its stores"):
        kapso.build("test goal", output=str(tmp_path / "ws"))


def test_a_build_refuses_an_index_re_ingested_since_it_was_loaded(tmp_path, monkeypatch, ingestion):
    # Regression: the handle pinned index A, the name was removed and
    # ingested again as B before the build, and the build connected B
    # while recording A. The pin holds at the build: the record on disk
    # must be the index that was loaded.
    kapso, _captured = _build_with(tmp_path, monkeypatch, FakeSearch({"weaviate_collection": "c", "corpus": "i"}))
    kapso.add_experience(repo="https://github.com/org/x", name="org-x")
    loaded = kapso.load_experience("org-x")
    record_path = Path(settings_for(tmp_path)["indexes"]) / "org-x" / "index.json"
    record = json.loads(record_path.read_text())
    record_path.write_text(json.dumps({**record, "id": "b" * 12}))
    with pytest.raises(ExperienceError, match=f"org-x was re-ingested since it was loaded \\(index {loaded.head} then, bbbbbbbbbbbb now\\)"):
        kapso.build("test goal", output=str(tmp_path / "ws"))


def test_nothing_loaded_is_the_null_search_and_no_backend(tmp_path, monkeypatch):
    kapso, captured = _build_with(tmp_path, monkeypatch, None)
    monkeypatch.setattr(
        "kapso.kapso.KnowledgeSearchFactory",
        SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(AssertionError("no backend without an index")),
                        create_null=lambda: SimpleNamespace(is_enabled=lambda: False, close=lambda: None)),
    )
    kapso.build("test goal", output=str(tmp_path / "ws"))
    assert captured["knowledge_search"] is None and captured["strategy_params_overrides"] is None
