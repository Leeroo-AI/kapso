# File: `libs/langchain_v1/tests/integration_tests/embeddings/test_base.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 44 |
| Functions | `test_init_embedding_model` |
| Imports | importlib, langchain, langchain_core, pytest |

## Understanding

**Status:** ✅ Explored

**Purpose:** Integration tests for init_embeddings function covering multiple embedding providers.

**Mechanism:** Parametrized test across providers (OpenAI, Google Vertex AI, AWS Bedrock, Cohere) with specific models. Tests both colon-separated syntax (provider:model) and explicit parameters. Skips tests when provider packages aren't installed. Validates embedding output format (list of floats) for query embeddings.

**Significance:** Validates the unified embedding initialization system across multiple providers. Ensures consistent API regardless of backend, critical for building provider-agnostic retrieval applications.
