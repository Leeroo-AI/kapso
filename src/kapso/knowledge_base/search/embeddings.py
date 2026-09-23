"""Text embedding providers for KG search: OpenAI (the default) and Google Vertex AI.

An index records the provider, model and size that embedded its pages in its
backend_refs, and search builds its embedder from those same refs. A query is
therefore always embedded by the model that embedded the pages it is compared
against; vectors from two different models are never mixed.

Failures raise. A page that cannot be embedded must stop an index build rather
than silently go missing from it, and a query that cannot be embedded must
surface as an error rather than as "no results".
"""

import time
from typing import Any, Dict, List, Mapping, Optional, Union

import google.auth
from google.auth.transport.requests import AuthorizedSession
from openai import OpenAI

from kapso.core.api_endpoint import openai_compatible_options, validate_api_base_url

VERTEX_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

# Rate limiting and transient server errors are retried; any other status
# (bad request, permission, over-long input) is final.
TRANSIENT_STATUSES = frozenset({429, 500, 502, 503, 504})


class OpenAIEmbedder:
    """OpenAI embeddings; the endpoint comes from config, never the environment."""

    def __init__(self, model: str, base_url: str):
        self.model = model
        # Pass the configured URL explicitly so the SDK cannot pick an env URL.
        self._client = OpenAI(base_url=validate_api_base_url(base_url))

    def embed_document(self, title: str, text: str) -> List[float]:
        """Embed a page. OpenAI pages are embedded from their text alone."""
        return self._embed(text)

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        return self._client.embeddings.create(model=self.model, input=text).data[0].embedding

    def close(self) -> None:
        self._client.close()


class VertexEmbedder:
    """Vertex AI embeddings through the embedContent API (gemini-embedding-2).

    gemini-embedding-2 takes no task_type (the API accepts the field and
    ignores it). Google's documented retrieval format carries the task in the
    text instead: pages as "title: ... | text: ...", queries as
    "task: search result | query: ...".
    """

    DOCUMENT_FORMAT = "title: {title} | text: {text}"
    QUERY_FORMAT = "task: search result | query: {text}"

    def __init__(
        self,
        model: str,
        project: Optional[str],
        location: str,
        dimensions: Optional[int],
        max_attempts: int,
        backoff_seconds: float,
        timeout_seconds: float,
    ):
        if not project:
            raise ValueError("vertex_project is required when embedding_provider is 'vertex'")
        if max_attempts < 1:
            raise ValueError(f"embedding_max_attempts must be at least 1, got {max_attempts}")
        self.model = model
        # Application Default Credentials, read by google-auth itself.
        credentials, _ = google.auth.default(scopes=[VERTEX_SCOPE], quota_project_id=project)
        self._session = AuthorizedSession(credentials)
        # gemini-embedding-2 is served from the global endpoint and from the
        # multi-region ("us", "eu") endpoints, not from single regions.
        host = "aiplatform.googleapis.com" if location == "global" else f"aiplatform.{location}.rep.googleapis.com"
        self._url = (
            f"https://{host}/v1/projects/{project}/locations/{location}"
            f"/publishers/google/models/{model}:embedContent"
        )
        self._dimensions = dimensions
        self._max_attempts = max_attempts
        self._backoff_seconds = backoff_seconds
        self._timeout_seconds = timeout_seconds

    def embed_document(self, title: str, text: str) -> List[float]:
        return self._embed(self.DOCUMENT_FORMAT.format(title=title, text=text))

    def embed_query(self, text: str) -> List[float]:
        return self._embed(self.QUERY_FORMAT.format(text=text))

    def _embed(self, text: str) -> List[float]:
        # autoTruncate off: an input over the model's limit is an error, never
        # a silently shortened embedding.
        body: Dict[str, Any] = {"content": {"parts": [{"text": text}]}, "autoTruncate": False}
        if self._dimensions:
            body["outputDimensionality"] = self._dimensions
        for attempt in range(1, self._max_attempts + 1):
            response = self._session.post(self._url, json=body, timeout=self._timeout_seconds)
            if response.status_code == 200:
                return response.json()["embedding"]["values"]
            if response.status_code not in TRANSIENT_STATUSES or attempt == self._max_attempts:
                break
            time.sleep(self._backoff_seconds * 2 ** (attempt - 1))
        raise RuntimeError(
            f"Vertex embedding with {self.model} failed after {attempt} attempt(s): "
            f"HTTP {response.status_code} {response.text}"
        )

    def close(self) -> None:
        self._session.close()


Embedder = Union[OpenAIEmbedder, VertexEmbedder]


def make_embedder(params: Mapping[str, Any]) -> Embedder:
    """Build the embedder the (defaults-merged) search params name."""
    provider = params["embedding_provider"]
    if provider == "openai":
        return OpenAIEmbedder(
            model=params["embedding_model"],
            base_url=params.get("embedding_base_url", openai_compatible_options()["base_url"]),
        )
    if provider == "vertex":
        return VertexEmbedder(
            model=params["embedding_model"],
            project=params.get("vertex_project"),
            location=params["vertex_location"],
            dimensions=params.get("embedding_dimensions"),
            max_attempts=params["embedding_max_attempts"],
            backoff_seconds=params["embedding_backoff_seconds"],
            timeout_seconds=params["embedding_timeout_seconds"],
        )
    raise ValueError(f"Unknown embedding_provider {provider!r}; expected 'openai' or 'vertex'")
