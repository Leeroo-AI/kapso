# File: `libs/langchain_v1/tests/unit_tests/embeddings/test_base.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 111 |
| Functions | `test_parse_model_string`, `test_parse_model_string_errors`, `test_infer_model_and_provider`, `test_infer_model_and_provider_errors`, `test_supported_providers_package_names`, `test_is_sorted` |
| Imports | langchain, pytest |

## Understanding

**Status:** ✅ Explored

**Purpose:** Unit tests for embeddings base module functions covering model string parsing, provider inference, and provider registry validation.

**Mechanism:** Tests verify: (1) parsing of colon-separated model strings like "openai:text-embedding-3-small" into provider and model components, (2) error handling for malformed strings (empty, missing provider, invalid format), (3) inference logic that handles both explicit provider parameters and prefixed model strings, (4) fine-tuned model support (e.g., "ft:text-embedding-3-small"), (5) supported providers registry has valid package names (lowercase, starts with "langchain_"), and (6) provider list is alphabetically sorted.

**Significance:** Critical for ensuring the embeddings initialization API correctly parses various model string formats and maintains a clean provider registry. Tests guarantee consistent error messages that help users identify supported providers and enforce naming conventions for integration packages.
