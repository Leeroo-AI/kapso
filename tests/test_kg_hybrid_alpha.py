"""The hybrid query's vector/keyword weight comes from the search's params."""

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


def hybrid_alpha_sent(monkeypatch, params):
    """The alpha a search built from `params` sends to hybrid()."""
    collection = RecordingCollection()
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
    search._semantic_search("q", KGSearchFilters(top_k=3))
    # The stubs have no close() for the destructor to call.
    search._embedder = search._weaviate_client = None
    return collection.hybrid_kwargs["alpha"]


def test_the_configured_hybrid_alpha_reaches_the_hybrid_query(monkeypatch):
    assert hybrid_alpha_sent(monkeypatch, {"hybrid_alpha": 0.2}) == 0.2
