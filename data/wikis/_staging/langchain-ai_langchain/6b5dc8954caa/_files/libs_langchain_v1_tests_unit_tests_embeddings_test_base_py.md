# File: `libs/langchain_v1/tests/unit_tests/embeddings/test_base.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 111 |
| Functions | `test_parse_model_string`, `test_parse_model_string_errors`, `test_infer_model_and_provider`, `test_infer_model_and_provider_errors`, `test_supported_providers_package_names`, `test_is_sorted` |
| Imports | langchain, pytest |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests embeddings model string parsing and provider inference for the init_embeddings factory function.

**Mechanism:** Tests _parse_model_string (parses "provider:model" format), _infer_model_and_provider (infers from various input formats), error cases (missing provider, empty model, invalid provider), and validates that _SUPPORTED_PROVIDERS contains valid package names (langchain_* format, lowercase, no hyphens, sorted alphabetically).

**Significance:** Validates the universal embeddings factory's ability to correctly parse and infer provider/model combinations from strings like "openai:text-embedding-3-small" or "bedrock:amazon.titan-embed-text-v1".
