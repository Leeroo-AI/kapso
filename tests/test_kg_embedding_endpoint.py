"""Embedding endpoint compatibility without connecting to KG services."""

import pytest

from kapso.knowledge_base.search import kg_graph_search as graph


@pytest.mark.parametrize(
    "base,legacy,expected",
    [
        ("https://new.test/v1", "https://old.test/v1", "https://new.test/v1"),
        (None, "https://old.test/v1", "https://old.test/v1"),
        (None, None, None),
    ],
)
def test_embedding_endpoint_precedence(monkeypatch, base, legacy, expected):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    for name, value in [
        ("OPENAI_BASE_URL", base),
        ("OPENAI_API_BASE", legacy),
    ]:
        monkeypatch.delenv(name, raising=False)
        if value:
            monkeypatch.setenv(name, value)
    monkeypatch.setattr(graph, "HAS_OPENAI", True)
    monkeypatch.setattr(graph, "OpenAI", lambda **options: options)
    search = object.__new__(graph.KGGraphSearch)
    search._neo4j_driver = None
    search._weaviate_client = None
    search._initialize_openai()
    assert search._openai_client.get("base_url") == expected
    assert search._openai_client["api_key"] == "test-key"
