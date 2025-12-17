# File: `tests/test_logprobs.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 210 |
| Functions | `test_create_logprobs_non_flat`, `test_create_logprobs_flat`, `test_append_logprobs_for_next_position_none_flat`, `test_append_logprobs_for_next_position_flat`, `test_flat_logprobs_append`, `test_flat_logprobs_extend`, `test_flat_logprobs_access` |
| Imports | vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Logprobs data structure testing

**Mechanism:** Tests both flat and non-flat logprobs formats. Validates create_prompt_logprobs, create_sample_logprobs, append_logprobs_for_next_position, FlatLogprobs append/extend/access operations, and slice indexing. Tests ensure proper structure, rank calculation, and data storage.

**Significance:** Validates the logprobs output format that users rely on for token probability inspection, ensuring consistent API behavior and efficient storage for both streaming and non-streaming scenarios.
