"""The hybrid query: its vector/keyword weight comes from the search's
params, and its failure reaches the caller instead of reading as no
results (Rule 2)."""

from types import SimpleNamespace

import pytest

from kapso.knowledge_base.search import kg_graph_search as module
from kapso.knowledge_base.search.base import KGSearchFilters
from kapso.knowledge_base.search.factory import KnowledgeSearchFactory

pytest.importorskip("weaviate")


class RecordingCollection:
    def __init__(self):
        self.hybrid_kwargs = None
        self.query = self

    def hybrid(self, **kwargs):
        self.hybrid_kwargs = kwargs
        return SimpleNamespace(objects=[])


class FailingCollection:
    def __init__(self):
        self.query = self

    def hybrid(self, **kwargs):
        raise RuntimeError("weaviate hybrid query failed")


def search_over(monkeypatch, collection, params):
    """A search built from `params` whose Weaviate collection is `collection`."""
    cls = module.KGGraphSearch
    monkeypatch.setattr(cls, "_initialize_neo4j", lambda self: None)
    monkeypatch.setattr(cls, "_initialize_weaviate", lambda self: None)
    monkeypatch.setattr(module, "make_embedder", lambda params: None)
    search = KnowledgeSearchFactory.create(
        "kg_graph_search", params={"corpus": "index-id", **params}
    )
    search._embedder = SimpleNamespace(embed_query=lambda query: [0.0])
    search._weaviate_client = SimpleNamespace(
        collections=SimpleNamespace(get=lambda name: collection)
    )
    return search


def test_the_configured_hybrid_alpha_reaches_the_hybrid_query(monkeypatch):
    collection = RecordingCollection()
    search = search_over(monkeypatch, collection, {"hybrid_alpha": 0.2})
    search._semantic_search("q", KGSearchFilters(top_k=3))
    # The stubs have no close() for the destructor to call.
    search._embedder = search._weaviate_client = None
    assert collection.hybrid_kwargs["alpha"] == 0.2


def test_a_failed_hybrid_query_raises_instead_of_returning_no_results(monkeypatch):
    search = search_over(monkeypatch, FailingCollection(), {})
    with pytest.raises(RuntimeError, match="hybrid query failed"):
        search.search("q", KGSearchFilters(top_k=3))
    search._embedder = search._weaviate_client = None
