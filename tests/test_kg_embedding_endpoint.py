"""Embedding endpoints come from config even with SDK environment overrides."""

import httpx
import pytest
from openai import OpenAI

from kapso.core.api_endpoint import openai_compatible_options
from kapso.knowledge_base.search import kg_graph_search as graph


@pytest.mark.parametrize("configured", [None, "https://embeddings.test/v1"])
def test_embedding_endpoint_ignores_environment(monkeypatch, configured):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://ambient.test/v1")
    monkeypatch.setenv("OPENAI_API_BASE", "https://legacy.test/v1")
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1]}
                ],
                "model": "embedding-model",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    def client(**options):
        return OpenAI(
            **options,
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        )

    monkeypatch.setattr(graph, "OpenAI", client)
    search = object.__new__(graph.KGGraphSearch)
    search._neo4j_driver = None
    search._weaviate_client = None
    search.params = {"embedding_base_url": configured} if configured else {}
    search._initialize_openai()
    search._openai_client.embeddings.create(
        model="embedding-model", input="test"
    )
    expected = configured or openai_compatible_options()["base_url"]
    assert str(requests[0].url) == expected + "/embeddings"
    assert requests[0].headers["authorization"] == "Bearer test-key"
    search._openai_client.close()


def test_embedding_plaintext_endpoint_is_rejected(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    search = object.__new__(graph.KGGraphSearch)
    search._neo4j_driver = None
    search._weaviate_client = None
    search.params = {"embedding_base_url": "http://remote.test/v1"}
    search._openai_client = None
    with pytest.raises(ValueError, match="HTTPS"):
        search._initialize_openai()
