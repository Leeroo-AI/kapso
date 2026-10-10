"""get_page must return the page whose id equals the request.

Weaviate stores ``page_id`` as a word-tokenized text property, so an ``equal``
filter also matches sibling pages that contain every token of the request
("<id>_Config", "<id>_Test"). The old ``limit=1`` lookup returned whichever
sibling sorted first — about 2% of live get_page calls came back with the
wrong page. These tests drive the method against a fake collection with the
siblings listed first.
"""

from types import SimpleNamespace

import pytest

from kapso.knowledge_base.search import kg_graph_search as module
from kapso.knowledge_base.search.kg_graph_search import KGGraphSearch

pytest.importorskip("weaviate")

REQUESTED = "Principle/Repo_Topic"
SIBLING = "Principle/Repo_Topic_Detail"


def page(page_id):
    return SimpleNamespace(
        uuid=f"uuid-of-{page_id}",
        properties={
            "page_id": page_id, "page_type": "Principle",
            "overview": f"{page_id} overview", "content": f"{page_id} body",
            "description": f"{page_id} description", "domains": ["LLM"],
        },
    )


class FakeQuery:
    """Token-matching candidates come back siblings first, the way the live
    index happened to order them."""

    def __init__(self, candidates):
        self.candidates = candidates
        self.fetch_objects_calls = []
        self.fetched_by_id = []

    def fetch_objects(self, **kwargs):
        self.fetch_objects_calls.append(kwargs)
        return SimpleNamespace(objects=[
            SimpleNamespace(uuid=c.uuid, properties={"page_id": c.properties["page_id"]})
            for c in self.candidates
        ])

    def fetch_object_by_id(self, uuid, include_vector=False):
        self.fetched_by_id.append(uuid)
        return next((c for c in self.candidates if c.uuid == uuid), None)


def search_over(candidates):
    query = FakeQuery(candidates)
    search = KGGraphSearch.__new__(KGGraphSearch)
    search.weaviate_collection = "LeeroopediaKG"
    search._weaviate_client = SimpleNamespace(
        collections=SimpleNamespace(get=lambda name: SimpleNamespace(query=query)),
    )
    return search, query


def test_get_page_returns_the_exact_id_not_the_sibling_that_sorts_first():
    search, query = search_over([page(SIBLING), page(REQUESTED)])

    result = search.get_page(REQUESTED)

    assert result.id == REQUESTED
    assert result.content == f"{REQUESTED} body"
    assert query.fetched_by_id == [f"uuid-of-{REQUESTED}"]
    (call,) = query.fetch_objects_calls
    assert call["filters"].target == "page_id" and call["filters"].value == REQUESTED
    # Candidates are fetched as ids only; the page body is read once, for the match.
    assert call["return_properties"] == ["page_id"]
    assert call["limit"] == module._PAGE_ID_TOKEN_MATCH_LIMIT > 1


def test_get_page_is_none_when_only_siblings_share_the_tokens():
    search, query = search_over([page(SIBLING)])

    assert search.get_page(REQUESTED) is None
    assert query.fetched_by_id == []
