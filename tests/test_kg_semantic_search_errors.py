"""A failed Weaviate query reaches the caller of search() instead of
reading as no results (Rule 2)."""

from types import SimpleNamespace

import pytest

from kapso.knowledge_base.search import kg_graph_search as module
from kapso.knowledge_base.search.base import KGSearchFilters
from kapso.knowledge_base.search.factory import KnowledgeSearchFactory

pytest.importorskip("weaviate")


class FailingCollection:
    def __init__(self):
        self.query = self

    def hybrid(self, **kwargs):
        raise RuntimeError("weaviate hybrid query failed")


def test_a_failed_hybrid_query_raises_instead_of_returning_no_results(
    monkeypatch,
):
    cls = module.KGGraphSearch
    monkeypatch.setattr(cls, "_initialize_neo4j", lambda self: None)
    monkeypatch.setattr(cls, "_initialize_weaviate", lambda self: None)
    monkeypatch.setattr(module, "make_embedder", lambda params: None)
    search = KnowledgeSearchFactory.create(
        "kg_graph_search", params={"corpus": "index-id"}
    )
    search._embedder = SimpleNamespace(embed_query=lambda query: [0.0])
    search._weaviate_client = SimpleNamespace(
        collections=SimpleNamespace(get=lambda name: FailingCollection())
    )
    with pytest.raises(RuntimeError, match="hybrid query failed"):
        search.search("q", KGSearchFilters(top_k=3))
    # The stubs have no close() for the destructor to call.
    search._embedder = search._weaviate_client = None
