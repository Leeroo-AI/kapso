{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|langchain-ai_langchain|https://github.com/langchain-ai/langchain]]
* [[source::Doc|LangChain Embeddings|https://python.langchain.com/docs/concepts/embedding_models/]]
|-
! Domains
| [[domain::Embeddings]], [[domain::Factory Pattern]], [[domain::Model Selection]]
|-
! Last Updated
| [[last_updated::2025-12-18 12:00 GMT]]
|}

== Overview ==

Factory function `init_embeddings` for creating embedding model instances from provider:model strings.

=== Description ===

The `init_embeddings` function provides a unified interface for instantiating embedding models from various providers (OpenAI, Azure, AWS Bedrock, Cohere, Google VertexAI, HuggingFace, MistralAI, Ollama). It parses a "provider:model" string format to dynamically load the appropriate embedding class and handles package installation validation.

=== Usage ===

Use this function to create embedding models without importing provider-specific classes directly. It simplifies configuration-driven model selection and enables runtime switching between embedding providers.

== Code Reference ==

=== Source Location ===
* '''Repository:''' [https://github.com/langchain-ai/langchain langchain-ai_langchain]
* '''File:''' [https://github.com/langchain-ai/langchain/blob/main/libs/langchain_v1/langchain/embeddings/base.py libs/langchain_v1/langchain/embeddings/base.py]
* '''Lines:''' 123-239

=== Signature ===
<syntaxhighlight lang="python">
def init_embeddings(
    model: str,
    *,
    provider: str | None = None,
    **kwargs: Any,
) -> Embeddings:
    """Initialize an embedding model from a model name and optional provider.

    Args:
        model: Model name, e.g. 'openai:text-embedding-3-small'.
        provider: Provider if not in model string.
        **kwargs: Additional model-specific parameters.

    Returns:
        An Embeddings instance.

    Raises:
        ValueError: If provider is not supported.
        ImportError: If required package is not installed.
    """
</syntaxhighlight>

=== Import ===
<syntaxhighlight lang="python">
from langchain.embeddings import init_embeddings
</syntaxhighlight>

== I/O Contract ==

=== Supported Providers ===
{| class="wikitable"
|-
! Provider !! Package !! Example Model
|-
| openai || langchain-openai || text-embedding-3-small
|-
| azure_openai || langchain-openai || text-embedding-ada-002
|-
| bedrock || langchain-aws || amazon.titan-embed-text-v1
|-
| cohere || langchain-cohere || embed-english-v3.0
|-
| google_vertexai || langchain-google-vertexai || textembedding-gecko
|-
| huggingface || langchain-huggingface || sentence-transformers/all-MiniLM-L6-v2
|-
| mistralai || langchain-mistralai || mistral-embed
|-
| ollama || langchain-ollama || nomic-embed-text
|}

=== Inputs ===
{| class="wikitable"
|-
! Name !! Type !! Required !! Description
|-
| model || str || Yes || Model identifier (provider:model or just model)
|-
| provider || str | None || No || Explicit provider override
|-
| **kwargs || Any || No || Provider-specific parameters (api_key, etc.)
|}

=== Outputs ===
{| class="wikitable"
|-
! Type !! Description
|-
| Embeddings || Configured embedding model instance
|}

== Usage Examples ==

=== Basic Initialization ===
<syntaxhighlight lang="python">
from langchain.embeddings import init_embeddings

# Using provider:model format
embeddings = init_embeddings("openai:text-embedding-3-small")

# Embed text
vector = embeddings.embed_query("Hello, world!")
print(len(vector))  # 1536 dimensions for this model
</syntaxhighlight>

=== Explicit Provider ===
<syntaxhighlight lang="python">
# Specify provider separately
embeddings = init_embeddings(
    model="text-embedding-3-small",
    provider="openai"
)
</syntaxhighlight>

=== With Configuration ===
<syntaxhighlight lang="python">
# Pass API key and other parameters
embeddings = init_embeddings(
    "openai:text-embedding-3-small",
    api_key="sk-...",
    dimensions=512,  # OpenAI supports dimension reduction
)

# Batch embed documents
vectors = embeddings.embed_documents([
    "First document",
    "Second document",
    "Third document",
])
</syntaxhighlight>

=== Multiple Providers ===
<syntaxhighlight lang="python">
# AWS Bedrock
bedrock_embed = init_embeddings("bedrock:amazon.titan-embed-text-v1")

# Cohere
cohere_embed = init_embeddings("cohere:embed-english-v3.0")

# Local with Ollama
ollama_embed = init_embeddings("ollama:nomic-embed-text")

# HuggingFace (local)
hf_embed = init_embeddings(
    "huggingface:sentence-transformers/all-MiniLM-L6-v2"
)
</syntaxhighlight>

=== Configuration-Driven Selection ===
<syntaxhighlight lang="python">
import os

# Select embedding model from environment
model_config = os.environ.get("EMBEDDING_MODEL", "openai:text-embedding-3-small")
embeddings = init_embeddings(model_config)

# Use in RAG pipeline
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Same embeddings interface regardless of provider
vectorstore = FAISS.from_documents(documents, embeddings)
</syntaxhighlight>

== Related Pages ==
* [[implements::Concept:langchain-core_Embeddings_Interface]]
* [[used_by::Implementation:langchain-ai_langchain_create_retrieval_chain]]

