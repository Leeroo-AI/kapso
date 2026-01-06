# File: `libs/langchain_v1/langchain/embeddings/base.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 245 |
| Functions | `init_embeddings` |
| Imports | functools, importlib, langchain_core, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Factory function for initializing embeddings models from 8 supported providers using a unified interface with provider:model format.

**Mechanism:** `init_embeddings` parses model strings in "provider:model" format (e.g., "openai:text-embedding-3-small"), validates provider support, checks package availability, and dynamically imports/instantiates the appropriate embeddings class. Provider-specific parameter names are handled (e.g., model_id for Bedrock, model_name for HuggingFace). Uses LRU caching for package checks.

**Significance:** Core abstraction for provider-agnostic embeddings initialization. Supports OpenAI, Azure OpenAI, Google Vertex AI, AWS Bedrock, Cohere, MistralAI, HuggingFace, and Ollama. Enforces consistent naming convention ("provider:model" format) across all embeddings implementations.
