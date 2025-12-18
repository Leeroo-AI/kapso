# File: `tests/test_pooling_params.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 156 |
| Classes | `MockModelConfig` |
| Functions | `test_task`, `test_embed`, `test_embed_dimensions`, `test_classify`, `test_token_embed`, `test_token_classify` |
| Imports | dataclasses, pytest, tests, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** PoolingParams validation tests

**Mechanism:** Tests PoolingParams validation for different tasks (score, embed, classify, token_embed, token_classify) including: task verification, parameter validation (normalize, use_activation, dimensions), Matryoshka embedding dimension support, and pooling type compatibility (ALL, STEP, CLS). Ensures invalid parameters for each task type raise appropriate errors.

**Significance:** Validates the PoolingParams API for embedding and classification tasks, ensuring proper parameter validation and compatibility with different model architectures and pooling strategies.
