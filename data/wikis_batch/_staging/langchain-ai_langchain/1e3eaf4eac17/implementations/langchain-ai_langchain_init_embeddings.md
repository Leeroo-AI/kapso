---
title: "init_embeddings Factory Function"
repository: "langchain-ai/langchain"
commit_hash: "1e3eaf4eac17"
file_path: "libs/langchain_v1/langchain/embeddings/base.py"
component_type: "Factory Function"
component_name: "init_embeddings"
layer: "Implementation Layer"
added_version: "0.3.9"
---

# init_embeddings Factory Function

## Overview

The `init_embeddings` factory function provides a unified interface for initializing embedding models across multiple AI providers. It abstracts the complexity of importing and configuring provider-specific embedding classes, allowing developers to instantiate embeddings using a simple string-based model specification.

**Key Features:**
- Unified interface for 8+ embedding providers (OpenAI, Azure OpenAI, Bedrock, Cohere, Google Vertex AI, HuggingFace, MistralAI, Ollama)
- Flexible model specification: `provider:model-name` format or explicit provider parameter
- Automatic package availability checking with helpful error messages
- Provider-specific parameter forwarding via `**kwargs`
- LRU caching for package import checks

**Location:** `/tmp/praxium_repo_wjjl6pl8/libs/langchain_v1/langchain/embeddings/base.py`

## Code Reference

### Main Factory Function

```python
def init_embeddings(
    model: str,
    *,
    provider: str | None = None,
    **kwargs: Any,
) -> Embeddings:
    """Initialize an embedding model from a model name and optional provider.

    Args:
        model: The name of the model, e.g. 'openai:text-embedding-3-small'.
            You can also specify model and model provider in a single argument using
            '{model_provider}:{model}' format.
        provider: The model provider if not specified as part of the model arg.
            Supported providers: openai, azure_openai, bedrock, cohere,
            google_vertexai, huggingface, mistralai, ollama
        **kwargs: Additional model-specific parameters passed to the embedding model.

    Returns:
        An Embeddings instance that can generate embeddings for text.

    Raises:
        ValueError: If the model provider is not supported or cannot be determined
        ImportError: If the required provider package is not installed
    """
```

### Supported Providers Mapping

```python
_SUPPORTED_PROVIDERS = {
    "azure_openai": "langchain_openai",
    "bedrock": "langchain_aws",
    "cohere": "langchain_cohere",
    "google_vertexai": "langchain_google_vertexai",
    "huggingface": "langchain_huggingface",
    "mistralai": "langchain_mistralai",
    "ollama": "langchain_ollama",
    "openai": "langchain_openai",
}
```

### Helper Functions

```python
def _parse_model_string(model_name: str) -> tuple[str, str]:
    """Parse 'provider:model-name' format into components."""

def _infer_model_and_provider(
    model: str,
    *,
    provider: str | None = None,
) -> tuple[str, str]:
    """Determine provider and model name from inputs."""

@functools.lru_cache(maxsize=len(_SUPPORTED_PROVIDERS))
def _check_pkg(pkg: str) -> None:
    """Check if a package is installed (cached)."""
```

## I/O Contract

### Input

**Required Parameter:**
- `model` (str): Model identifier in `"provider:model-name"` format or just model name if provider is specified separately

**Optional Parameters:**
- `provider` (str | None): Explicit provider specification (alternative to `provider:model` format)
- `**kwargs` (Any): Provider-specific configuration parameters (API keys, endpoints, etc.)

### Output

**Returns:** `Embeddings` instance
- Concrete implementation varies by provider (OpenAIEmbeddings, AzureOpenAIEmbeddings, etc.)
- All implement the `langchain_core.embeddings.Embeddings` interface
- Provides `embed_query()` and `embed_documents()` methods

### Exceptions

- `ValueError`: Invalid model format, unsupported provider, empty model name
- `ImportError`: Required provider package not installed

## Usage Examples

### Basic Usage - Model String Format

```python
from langchain.embeddings import init_embeddings

# Using provider:model format
embeddings = init_embeddings("openai:text-embedding-3-small")
vector = embeddings.embed_query("Hello, world!")
```

### Explicit Provider Parameter

```python
# Separate provider and model
embeddings = init_embeddings(
    model="text-embedding-3-small",
    provider="openai"
)
vectors = embeddings.embed_documents([
    "Hello, world!",
    "Goodbye, world!"
])
```

### With Provider-Specific Parameters

```python
# Pass API key and other parameters
embeddings = init_embeddings(
    "openai:text-embedding-3-small",
    api_key="sk-...",
    model_kwargs={"encoding_format": "float"}
)
```

### Multiple Providers

```python
# OpenAI
openai_emb = init_embeddings("openai:text-embedding-3-small")

# Cohere
cohere_emb = init_embeddings("cohere:embed-english-v3.0")

# Bedrock
bedrock_emb = init_embeddings("bedrock:amazon.titan-embed-text-v1")

# HuggingFace
hf_emb = init_embeddings(
    "huggingface:sentence-transformers/all-mpnet-base-v2"
)
```

### Error Handling

```python
try:
    embeddings = init_embeddings("invalid_format")
except ValueError as e:
    print(f"Invalid format: {e}")
    # Error message includes examples and supported providers

try:
    embeddings = init_embeddings("openai:gpt-4")
except ImportError as e:
    print(f"Missing package: {e}")
    # Error message includes pip install command
```

## Implementation Details

### Provider Instantiation Logic

The function uses a straightforward if-elif chain to instantiate the appropriate provider class:

```python
if provider == "openai":
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=model_name, **kwargs)

if provider == "azure_openai":
    from langchain_openai import AzureOpenAIEmbeddings
    return AzureOpenAIEmbeddings(model=model_name, **kwargs)

# ... similar blocks for other providers
```

### Package Import Checking

The `_check_pkg()` function is cached using `@functools.lru_cache` to avoid repeated import checks:

```python
@functools.lru_cache(maxsize=len(_SUPPORTED_PROVIDERS))
def _check_pkg(pkg: str) -> None:
    if not util.find_spec(pkg):
        msg = f"Could not import {pkg} python package. Please install it with `pip install {pkg}`"
        raise ImportError(msg)
```

### Model String Parsing

The parsing logic handles two formats:
1. `"provider:model-name"` - Extracts both provider and model
2. `"model-name"` with separate `provider` parameter

Validation includes:
- Checking for colon separator when provider not specified
- Trimming whitespace from components
- Verifying provider is in supported list
- Ensuring model name is not empty

## Design Patterns

### Factory Pattern
The function implements the Factory pattern, creating instances of different classes based on runtime parameters without exposing instantiation logic.

### Strategy Pattern
Each provider represents a different strategy for generating embeddings, with the factory selecting the appropriate strategy.

### Lazy Imports
Provider-specific packages are imported only when needed, reducing startup time and avoiding import errors for unused providers.

## Related Components

### Base Classes
- `langchain_core.embeddings.Embeddings` - Abstract base class for all embeddings

### Provider Implementations
- `langchain_openai.OpenAIEmbeddings` - OpenAI embeddings
- `langchain_openai.AzureOpenAIEmbeddings` - Azure OpenAI embeddings
- `langchain_aws.BedrockEmbeddings` - AWS Bedrock embeddings
- `langchain_cohere.CohereEmbeddings` - Cohere embeddings
- `langchain_google_vertexai.VertexAIEmbeddings` - Google Vertex AI embeddings
- `langchain_huggingface.HuggingFaceEmbeddings` - HuggingFace embeddings
- `langchain_mistralai.MistralAIEmbeddings` - Mistral AI embeddings
- `langchain_ollama.OllamaEmbeddings` - Ollama embeddings

### Related Functions
- `init_chat_model` - Similar factory for chat models (libs/langchain_v1/langchain/chat_models/base.py)

## Integration Points

### Package Dependencies
The function requires provider-specific packages to be installed:
- `langchain-openai` (OpenAI, Azure OpenAI)
- `langchain-aws` (Bedrock)
- `langchain-cohere` (Cohere)
- `langchain-google-vertexai` (Google Vertex AI)
- `langchain-huggingface` (HuggingFace)
- `langchain-mistralai` (Mistral AI)
- `langchain-ollama` (Ollama)

### Configuration
Provider-specific configuration (API keys, endpoints) should be passed via `**kwargs` or environment variables as supported by each provider.

## Testing Considerations

### Unit Tests
- Test model string parsing with valid/invalid formats
- Test provider inference logic
- Test error handling for missing packages
- Test error messages include helpful information

### Integration Tests
- Test actual embedding generation with mocked providers
- Test parameter forwarding to provider classes
- Test each supported provider can be instantiated

### Mock Strategies
```python
from unittest.mock import patch

# Mock package availability
with patch('langchain.embeddings.base.util.find_spec') as mock_spec:
    mock_spec.return_value = None
    # Test ImportError handling
```

## Version History

- **v0.3.9** (2024): Initial release of `init_embeddings` factory function
- Provides unified interface for 8 embedding providers

## Best Practices

1. **Use model string format** for cleaner code: `"provider:model-name"`
2. **Handle ImportError** to guide users to install required packages
3. **Pass sensitive data** (API keys) via kwargs or environment variables
4. **Check provider documentation** for supported kwargs parameters
5. **Cache embeddings instances** when generating multiple embeddings with same configuration

## Common Pitfalls

1. **Wrong parameter names**: Different providers use different parameter names (e.g., `model` vs `model_id`)
2. **Missing packages**: Forgetting to install provider-specific packages
3. **Invalid model names**: Using model names not supported by the provider
4. **Incorrect format**: Forgetting the colon in `provider:model` format
