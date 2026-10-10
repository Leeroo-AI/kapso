"""Re-indexing a page must replace its Weaviate object instead of adding a duplicate."""

from types import SimpleNamespace

from kapso.knowledge_base.search.base import WikiPage
from kapso.knowledge_base.search.kg_graph_search import KGGraphSearch


class _FakeData:
    def __init__(self, collection):
        self.collection = collection

    def insert(self, *, properties, vector):
        self.collection.next_id += 1
        self.collection.objects.append(SimpleNamespace(uuid=self.collection.next_id, properties=properties))

    def delete_many(self, *, where):
        self.collection.objects = [
            obj for obj in self.collection.objects if obj.properties.get(where.target) != where.value
        ]


class _FakeCollection:
    def __init__(self):
        self.objects = []
        self.next_id = 0
        self.data = _FakeData(self)


class _FakeWeaviate:
    def __init__(self, collection):
        self.collections = SimpleNamespace(get=lambda name: collection)


def _page(page_id, overview):
    return WikiPage(
        id=page_id,
        page_type="Principle",
        overview=overview,
        content=f"== Overview ==\n{overview}",
        domains=["Deep_Learning"],
    )


def _search(collection):
    search = object.__new__(KGGraphSearch)
    search._weaviate_client = _FakeWeaviate(collection)
    search._neo4j_driver = None
    search.weaviate_collection = "KGWikiPages"
    search._ensure_weaviate_collection = lambda: None
    search._embed_page = lambda page_id, text: [0.1, 0.2, 0.3]
    search._embed = lambda text: [0.1, 0.2, 0.3]
    return search


def test_reindexing_replaces_existing_page_and_keeps_other_pages():
    collection = _FakeCollection()
    search = _search(collection)

    search._index_to_weaviate([_page("Principle/Batch_Size", "old text"), _page("Principle/Learning_Rate", "lr")])
    search._index_to_weaviate([_page("Principle/Batch_Size", "new text")])

    by_id = {}
    for obj in collection.objects:
        by_id.setdefault(obj.properties["page_id"], []).append(obj.properties["overview"])
    assert by_id == {"Principle/Batch_Size": ["new text"], "Principle/Learning_Rate": ["lr"]}
