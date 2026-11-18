"""The rank prior: a page's query-independent standing (`rank_prior`, 0 to
1) blended into the hybrid score. It must reorder a pool wider than top_k
(over top_k alone it could only shuffle what was already chosen), settle a
close call toward the higher prior while a clearly better match keeps its
place, and leave the query untouched when its weight is 0."""

from types import SimpleNamespace

import pytest

from kapso.knowledge_base.search import kg_graph_search as module
from kapso.knowledge_base.search.base import KGSearchFilters
from kapso.knowledge_base.search.factory import KnowledgeSearchFactory

pytest.importorskip("weaviate")


def hit(page_id, score, prior=None):
    properties = {"page_id": page_id, "page_type": "Principle", "overview": "", "content": "", "domains": []}
    if prior is not None:
        properties["rank_prior"] = prior
    return SimpleNamespace(properties=properties, metadata=SimpleNamespace(score=score))


class PoolCollection:
    """Answers the hybrid query with a fixed pool, cut to the requested limit."""

    def __init__(self, hits):
        self.hits = hits
        self.limit = None
        self.query = self

    def hybrid(self, **kwargs):
        self.limit = kwargs["limit"]
        return SimpleNamespace(objects=self.hits[: kwargs["limit"]])


def ranked(monkeypatch, hits, params, top_k):
    """(page ids returned, limit the hybrid query asked for) for one search."""
    cls = module.KGGraphSearch
    monkeypatch.setattr(cls, "_initialize_neo4j", lambda self: None)
    monkeypatch.setattr(cls, "_initialize_weaviate", lambda self: None)
    monkeypatch.setattr(module, "make_embedder", lambda params: None)
    search = KnowledgeSearchFactory.create("kg_graph_search", params={"corpus": "index-id", **params})
    collection = PoolCollection(hits)
    search._embedder = SimpleNamespace(embed_query=lambda query: [0.0])
    search._weaviate_client = SimpleNamespace(collections=SimpleNamespace(get=lambda name: collection))
    results = search._semantic_search("q", KGSearchFilters(top_k=top_k))
    # The stubs have no close() for the destructor to call.
    search._embedder = search._weaviate_client = None
    return [result.id for result in results], collection.limit, [result.score for result in results]


POOL = [hit("tiny_repo_a", 1.0, 0.1), hit("tiny_repo_b", 0.95, 0.1), hit("famous_repo", 0.9, 1.0), hit("unrated", 0.3)]


def test_weight_zero_is_the_plain_hybrid_query(monkeypatch):
    ids, limit, _ = ranked(monkeypatch, POOL, {}, top_k=2)
    assert limit == 2
    assert ids == ["tiny_repo_a", "tiny_repo_b"]


def test_the_prior_reorders_a_pool_wider_than_top_k(monkeypatch):
    ids, limit, scores = ranked(monkeypatch, POOL, {"rank_prior_weight": 0.2, "rank_prior_pool": 4}, top_k=2)
    assert limit == 4
    # 0.9 relevance with a full prior overtakes 0.95 and 1.0 with a weak one
    assert ids == ["famous_repo", "tiny_repo_a"]
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_a_clearly_better_match_keeps_its_place(monkeypatch):
    pool = [hit("exact_match", 1.0, 0.0), hit("famous_but_off_topic", 0.4, 1.0)]
    ids, _, _ = ranked(monkeypatch, pool, {"rank_prior_weight": 0.2, "rank_prior_pool": 4}, top_k=1)
    assert ids == ["exact_match"]
