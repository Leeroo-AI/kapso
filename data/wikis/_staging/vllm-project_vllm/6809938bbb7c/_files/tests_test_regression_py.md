# File: `tests/test_regression.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 79 |
| Functions | `test_duplicated_ignored_sequence_group`, `test_max_tokens_none`, `test_gc`, `test_model_from_modelscope` |
| Imports | gc, pytest, torch, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Regression tests for reported issues

**Mechanism:** Contains tests for specific user-reported bugs including: duplicated sequence group handling (issue #1655, currently skipped in V1), max_tokens=None handling, garbage collection and memory cleanup verification (ensures CUDA memory is released), and ModelScope model loading support (alternative to HuggingFace).

**Significance:** Prevents regressions of previously fixed issues. Documents historical bugs and validates fixes remain effective across code changes. Critical for maintaining stability and user confidence.
