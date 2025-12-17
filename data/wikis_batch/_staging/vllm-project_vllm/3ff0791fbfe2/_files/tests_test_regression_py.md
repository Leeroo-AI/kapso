# File: `tests/test_regression.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 79 |
| Functions | `test_duplicated_ignored_sequence_group`, `test_max_tokens_none`, `test_gc`, `test_model_from_modelscope` |
| Imports | gc, pytest, torch, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Regression tests for user-reported bugs

**Mechanism:** Contains tests for specific issues: duplicated ignored sequence groups (#1655), max_tokens=None handling, garbage collection memory release verification, and ModelScope model loading. Each test prevents previously fixed bugs from reoccurring.

**Significance:** Maintains system stability by ensuring reported bugs stay fixed across code changes, providing confidence in production deployments.
