"""`search_experience` and `show_experience`: an ingested index is read
back, not rebuilt. The stores are faked at the backend boundary; the wiki
files and the pages parsed from them are real."""

import pytest

import kapso.experience.browse as browse_module
from kapso.experience import ExperienceError, ExperienceReferenceError, ExperienceStore, IndexRecord
from kapso.experience.browse import index_page, open_search, render_hits, render_page
from kapso.knowledge_base.search import KGOutput, KGResultItem
from tests.handles_support import facade
from tests.test_experience_reference import settings_for

PAGE = """= Paged Optimizers =

== Overview ==
Optimizer state that spills to CPU memory when the GPU runs out.

=== Description ===
Paged optimizers keep optimizer state in unified memory.

== Metadata ==
| [[domain::LLMs]], [[domain::Fine_Tuning]]

== Related ==
* [[realized_by::Implementation:acme/Bnb_Paged_AdamW]]
"""


class FakeSearch:
    def __init__(self, hits=(), valid=True):
        self.hits = list(hits)
        self.valid = valid
        self.queries = []
        self.closed = False

    def validate_backend_data(self):
        return self.valid

    def search(self, query, filters):
        self.queries.append((query, filters.top_k))
        return KGOutput(query=query, filters=filters, results=self.hits)

    def close(self):
        self.closed = True


def make_index(tmp_path, name="qlora", with_staging=False):
    store = ExperienceStore(settings_for(tmp_path)["banks"], settings_for(tmp_path)["indexes"])
    directory = store.index_dir(name)
    (directory / "wiki" / "principles").mkdir(parents=True)
    (directory / "wiki" / "principles" / "Paged_Optimizers.md").write_text(PAGE)
    if with_staging:
        (directory / "wiki" / "_staging" / "principles").mkdir(parents=True)
        (directory / "wiki" / "_staging" / "principles" / "Half_Written.md").write_text(PAGE)
    record = IndexRecord(
        name=name, id="abc123def456", created_at="2026-11-05T00:00:00+00:00", source="paper.pdf",
        search_backend="kg_graph_search",
        backend_refs={"weaviate_collection": "KapsoQloraabc123def456", "weaviate_url": "http://localhost:8080"},
        page_count=1, path=directory,
    )
    store.write_index_record(record)
    return store.index_record(name)


def hit(page_id="Principle/Paged_Optimizers", score=0.87):
    return KGResultItem(id=page_id, score=score, page_type="Principle", overview="Optimizer state that spills to CPU.")


def kapso_for(tmp_path):
    kapso = facade()
    kapso._experience_settings = settings_for(tmp_path)
    return kapso


def test_a_down_or_wiped_store_is_refused_not_read_as_no_match(tmp_path, monkeypatch):
    record = make_index(tmp_path)
    wiped = FakeSearch(valid=False)
    monkeypatch.setattr(browse_module, "search_for", lambda record, config: wiped)
    with pytest.raises(ExperienceError, match="no pages in its stores.*KapsoQloraabc123def456.*localhost:8080"):
        open_search(record, {})
    assert wiped.closed
    live = FakeSearch()
    monkeypatch.setattr(browse_module, "search_for", lambda record, config: live)
    assert open_search(record, {}) is live and not live.closed


def test_a_page_is_read_from_the_wiki_files_with_its_links_and_never_from_staging(tmp_path):
    record = make_index(tmp_path, with_staging=True)
    page = index_page(record, "Principle/Paged_Optimizers")
    assert page.domains == ["LLMs", "Fine Tuning"] and page.content == PAGE
    assert page.outgoing_links == [
        {"edge_type": "realized_by", "target_type": "Implementation", "target_id": "acme/Bnb_Paged_AdamW"}
    ]
    with pytest.raises(ExperienceError, match="`qlora` has no page `Principle/Half_Written`"):
        index_page(record, "Principle/Half_Written")
    with pytest.raises(ExperienceError, match="`qlora` has no page `Principle/Nope`.*kapso experience search qlora"):
        index_page(record, "Principle/Nope")


def test_hits_and_a_page_are_rendered_whole(tmp_path):
    output = KGOutput(query="paged", filters=None, results=[hit(), hit("Workflow/Qlora", 0.5)])
    assert render_hits(output) == (
        "[1] Principle/Paged_Optimizers   Principle   score 0.87\nOptimizer state that spills to CPU.\n\n"
        "[2] Workflow/Qlora   Principle   score 0.50\nOptimizer state that spills to CPU."
    )
    assert render_hits(KGOutput(query="zzz", filters=None, results=[])) == "no pages matched 'zzz'"
    text = render_page(index_page(make_index(tmp_path), "Principle/Paged_Optimizers"))
    assert text.startswith("Principle/Paged_Optimizers   Principle\ndomains: LLMs, Fine Tuning\n\n= Paged Optimizers =")
    assert PAGE.rstrip() in text and text.endswith("links\n  realized_by -> Implementation: acme/Bnb_Paged_AdamW")


def test_search_defaults_to_the_configured_top_k_and_closes_the_backend(tmp_path, monkeypatch):
    make_index(tmp_path)
    search = FakeSearch([hit()])
    monkeypatch.setattr(browse_module, "search_for", lambda record, config: search)
    checked = []
    monkeypatch.setattr("kapso.kapso.run_preflight", lambda verb, config, **context: checked.append((verb, context)))
    kapso = kapso_for(tmp_path)
    kapso._experience_settings["search_top_k"] = 4
    assert kapso.search_experience("qlora", "paged optimizers").results == [hit()]
    assert kapso.search_experience("qlora", "paged optimizers", top_k=2).results == [hit()]
    assert search.queries == [("paged optimizers", 4), ("paged optimizers", 2)] and search.closed
    assert checked == [("experience", {"search": True})] * 2


def test_only_an_index_is_searched_or_shown(tmp_path):
    make_index(tmp_path)
    kapso = kapso_for(tmp_path)
    with pytest.raises(ExperienceError, match="`mine` is not an index: search reads an index"):
        kapso.search_experience("mine", "x")
    with pytest.raises(ExperienceError, match="`leeroopedia` is not an index: show reads an index"):
        kapso.show_experience("leeroopedia", "Principle/X")
    with pytest.raises(ExperienceReferenceError, match="`absent`"):
        kapso.search_experience("absent", "x")
    assert kapso.show_experience("qlora", "Principle/Paged_Optimizers").page_type == "Principle"
