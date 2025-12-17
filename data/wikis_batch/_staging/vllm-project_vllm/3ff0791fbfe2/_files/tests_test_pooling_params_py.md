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

**Purpose:** PoolingParams validation testing

**Mechanism:** Tests PoolingParams.verify() for different tasks (embed, score, classify, token_embed, token_classify). Validates parameter compatibility with pooling types (CLS, ALL, STEP), dimension constraints for matryoshka models, and invalid parameter rejection.

**Significance:** Ensures pooling models receive valid parameters for their specific tasks, preventing misconfigurations that could produce incorrect embeddings or classifications.
