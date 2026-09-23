"""KG search embedding providers.

Pins the contracts retrieval quality depends on and that fail silently when
broken: the exact request each provider receives, that failures raise instead
of dropping pages or returning "no results", and that an index's refs rebuild
the same embedder at search time (query vectors from the pages' own model).
"""

import json
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI

from kapso.core.api_endpoint import openai_compatible_options
from kapso.knowledge_base.search import embeddings
from kapso.knowledge_base.search import kg_graph_search as graph
from kapso.knowledge_base.search.base import WikiPage
from kapso.knowledge_base.search.factory import KnowledgeSearchFactory

VERTEX_PARAMS = {
    "weaviate_collection": "LeeroopediaKG",
    "embedding_provider": "vertex",
    "embedding_model": "gemini-embedding-2",
    "vertex_project": "proj",
}


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload

    @property
    def text(self):
        return json.dumps(self._payload)


OK = FakeResponse(200, {"embedding": {"values": [0.5, 0.5]}})


class FakeSession:
    """Stands in for google-auth's AuthorizedSession; replays scripted responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.posts = []

    def post(self, url, json, timeout):
        self.posts.append((url, json))
        return self.responses.pop(0)

    def close(self):
        pass


@pytest.fixture
def vertex_session(monkeypatch):
    """Route Vertex calls to a FakeSession; returns a factory for scripted sessions."""
    sleeps = []
    monkeypatch.setattr(embeddings.time, "sleep", sleeps.append)
    monkeypatch.setattr(
        embeddings.google.auth, "default",
        lambda scopes, quota_project_id: ("credentials", quota_project_id),
    )

    def script(responses):
        session = FakeSession(responses)
        session.sleeps = sleeps
        monkeypatch.setattr(embeddings, "AuthorizedSession", lambda credentials: session)
        return session

    return script


def vertex_embedder(**overrides):
    """A VertexEmbedder on the shipped defaults, as the search backend builds it."""
    params = {**KnowledgeSearchFactory.get_defaults("kg_graph_search"), **VERTEX_PARAMS, **overrides}
    return embeddings.make_embedder(params)


@pytest.fixture
def offline_backend(monkeypatch):
    """KGGraphSearch built through the factory without its databases or LLM:
    only the embedder is real, so no test touches a live Weaviate or Neo4j."""
    for name in ("_initialize_neo4j", "_initialize_weaviate", "_initialize_llm"):
        monkeypatch.setattr(graph.KGGraphSearch, name, lambda self: None)


# --------------------------------------------------------------------------
# OpenAI
# --------------------------------------------------------------------------

@pytest.mark.parametrize("configured", [None, "https://embeddings.test/v1"])
def test_openai_endpoint_comes_from_config_not_environment(monkeypatch, configured):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://ambient.test/v1")
    monkeypatch.setenv("OPENAI_API_BASE", "https://legacy.test/v1")
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
            "model": "embedding-model",
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        })

    monkeypatch.setattr(embeddings, "OpenAI", lambda **options: OpenAI(
        **options, http_client=httpx.Client(transport=httpx.MockTransport(respond))))
    params = {"embedding_provider": "openai", "embedding_model": "embedding-model"}
    if configured:
        params["embedding_base_url"] = configured
    embedder = embeddings.make_embedder(params)

    # OpenAI pages are embedded from their text alone, as before providers existed.
    assert embedder.embed_document("Some Page", "page text") == [0.1]
    expected = configured or openai_compatible_options()["base_url"]
    assert str(requests[0].url) == expected + "/embeddings"
    assert requests[0].headers["authorization"] == "Bearer test-key"
    assert json.loads(requests[0].content)["input"] == "page text"
    embedder.close()


def test_openai_plaintext_endpoint_is_rejected(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(ValueError, match="HTTPS"):
        embeddings.make_embedder({
            "embedding_provider": "openai",
            "embedding_model": "embedding-model",
            "embedding_base_url": "http://remote.test/v1",
        })


# --------------------------------------------------------------------------
# Vertex AI
# --------------------------------------------------------------------------

@pytest.mark.parametrize("location, host", [
    ("global", "aiplatform.googleapis.com"),
    ("us", "aiplatform.us.rep.googleapis.com"),
])
def test_vertex_request_contract(vertex_session, location, host):
    session = vertex_session([OK, OK])
    embedder = vertex_embedder(vertex_location=location, embedding_dimensions=1536)

    assert embedder.embed_document("Some Page", "What the page covers.") == [0.5, 0.5]
    embedder.embed_query("gradient checkpointing memory")

    (page_url, page_body), (query_url, query_body) = session.posts
    assert page_url == query_url == (
        f"https://{host}/v1/projects/proj/locations/{location}"
        "/publishers/google/models/gemini-embedding-2:embedContent"
    )
    assert page_body == {
        "content": {"parts": [{"text": "title: Some Page | text: What the page covers."}]},
        "autoTruncate": False,
        "outputDimensionality": 1536,
    }
    assert query_body["content"]["parts"][0]["text"] == (
        "task: search result | query: gradient checkpointing memory"
    )


def test_vertex_retries_rate_limits_and_server_errors(vertex_session):
    session = vertex_session([FakeResponse(429, {}), FakeResponse(503, {}), OK])
    assert vertex_embedder().embed_query("q") == [0.5, 0.5]
    assert len(session.posts) == 3
    assert session.sleeps == [1, 2]


def test_vertex_other_errors_raise_immediately(vertex_session):
    session = vertex_session([FakeResponse(400, {"error": {"message": "input exceeds 8192 tokens"}})])
    with pytest.raises(RuntimeError, match="HTTP 400.*exceeds 8192 tokens"):
        vertex_embedder().embed_document("Title", "text")
    assert len(session.posts) == 1
    assert session.sleeps == []


def test_vertex_gives_up_after_max_attempts(vertex_session):
    session = vertex_session([FakeResponse(503, {})] * 3)
    with pytest.raises(RuntimeError, match="after 3 attempt"):
        vertex_embedder(embedding_max_attempts=3).embed_query("q")
    assert len(session.posts) == 3


def test_vertex_requires_a_project(vertex_session):
    vertex_session([])
    with pytest.raises(ValueError, match="vertex_project"):
        vertex_embedder(vertex_project=None)


# --------------------------------------------------------------------------
# The index round trip
# --------------------------------------------------------------------------

def test_index_refs_rebuild_the_same_embedder(vertex_session, offline_backend):
    session = vertex_session([OK, OK])
    built = KnowledgeSearchFactory.create(
        "kg_graph_search", params={**VERTEX_PARAMS, "embedding_dimensions": 1536})

    refs = built.get_backend_refs()
    assert refs == {
        "weaviate_collection": "LeeroopediaKG",
        "embedding_provider": "vertex",
        "embedding_model": "gemini-embedding-2",
        "embedding_dimensions": 1536,
        "vertex_project": "proj",
        "vertex_location": "global",
    }

    # What the kg gate and get_page do with an index: build a backend from its refs.
    served = KnowledgeSearchFactory.create("kg_graph_search", params=refs)
    built._embed_page("Implementation/Unslothai_Unsloth_Get_Peft_Model", "Adds LoRA adapters.")
    served._embedder.embed_query("add lora adapters")

    (page_url, page_body), (query_url, query_body) = session.posts
    assert page_url == query_url
    assert page_body["outputDimensionality"] == query_body["outputDimensionality"] == 1536
    assert page_body["content"]["parts"][0]["text"] == (
        "title: Unslothai Unsloth Get Peft Model | text: Adds LoRA adapters."
    )


def test_index_without_a_provider_keeps_openai(monkeypatch, offline_backend):
    """Indexes written before providers existed (e.g. production's .index) keep
    searching with OpenAI."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    served = KnowledgeSearchFactory.create("kg_graph_search", params={
        "weaviate_collection": "KapsoKG",
        "embedding_model": "text-embedding-3-large",
    })
    assert isinstance(served._embedder, embeddings.OpenAIEmbedder)
    assert served.get_backend_refs() == {
        "weaviate_collection": "KapsoKG",
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-large",
    }


def test_index_build_stops_on_an_embedding_failure(vertex_session, offline_backend, monkeypatch):
    """A page that cannot be embedded stops the build; it is never silently
    left out (the failure that emptied most of the production index)."""
    vertex_session([OK, FakeResponse(403, {"error": {"message": "permission denied"}})])
    search = KnowledgeSearchFactory.create("kg_graph_search", params=VERTEX_PARAMS)
    inserted = []
    collection = SimpleNamespace(data=SimpleNamespace(
        insert=lambda properties, vector: inserted.append(properties["page_id"])))
    search._weaviate_client = SimpleNamespace(collections=SimpleNamespace(get=lambda name: collection))
    monkeypatch.setattr(search, "_ensure_weaviate_collection", lambda: None)

    pages = [
        WikiPage(id="Principle/A", page_type="Principle", overview="a", content=""),
        WikiPage(id="Principle/No_Text", page_type="Principle", overview="", content=""),
        WikiPage(id="Principle/B", page_type="Principle", overview="b", content=""),
    ]
    with pytest.raises(RuntimeError, match="HTTP 403.*permission denied"):
        search._index_to_weaviate(pages)
    assert inserted == ["Principle/A"]
    search._weaviate_client = None
