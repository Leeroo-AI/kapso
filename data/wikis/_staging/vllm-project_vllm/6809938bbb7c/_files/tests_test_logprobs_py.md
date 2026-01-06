# File: `tests/test_logprobs.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 210 |
| Functions | `test_create_logprobs_non_flat`, `test_create_logprobs_flat`, `test_append_logprobs_for_next_position_none_flat`, `test_append_logprobs_for_next_position_flat`, `test_flat_logprobs_append`, `test_flat_logprobs_extend`, `test_flat_logprobs_access` |
| Imports | vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Logprobs data structure tests

**Mechanism:** Tests both flat and nested logprobs representations including: creation (prompt and sample logprobs), appending logprobs for next positions, FlatLogprobs operations (append, extend with list or another FlatLogprobs), and access patterns (__len__, __iter__, __getitem__ with single index or slice).

**Significance:** Validates the FlatLogprobs optimization that provides memory-efficient storage for logprobs while maintaining compatibility with the traditional nested dict representation. Critical for API correctness and performance.
