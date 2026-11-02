"""close() and the destructor must work on an instance __init__ never ran for.

Tests build KGGraphSearch with __new__ to skip the Weaviate/Neo4j/embedder
setup; the destructor then called close(), which read attributes that only
__init__ sets and raised AttributeError at garbage collection.
"""

from kapso.knowledge_base.search.kg_graph_search import KGGraphSearch


def test_close_on_an_uninitialised_instance_is_a_noop():
    search = KGGraphSearch.__new__(KGGraphSearch)

    search.close()

    assert search._neo4j_driver is None
    assert search._weaviate_client is None
    assert search._embedder is None
