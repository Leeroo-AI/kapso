# File: `vllm/logprobs.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 206 |
| Classes | `Logprob`, `FlatLogprobs` |
| Functions | `create_prompt_logprobs`, `create_sample_logprobs`, `append_logprobs_for_next_position` |
| Imports | collections, dataclasses, itertools, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Data structures for storing and managing token log probabilities.

**Mechanism:** `Logprob` dataclass stores per-token information: logprob value, vocab rank, and decoded token string. `FlatLogprobs` is a memory-efficient alternative to `list[dict[int, Logprob]]` that flattens data into primitive arrays (token_ids, logprobs, ranks, decoded_tokens) with position indices. This reduces GC overhead significantly for long sequences. Implements `MutableSequence` interface for backward compatibility. Helper functions create containers for prompt vs sample logprobs and append new position data. Type aliases define `PromptLogprobs` and `SampleLogprobs` for clarity.

**Significance:** Critical for OpenAI API compatibility which requires detailed logprob information. The `FlatLogprobs` optimization is important for performance at scale - thousands of tokens with top-k logprobs create many objects. The flattened design reduces memory pressure and GC pauses. This demonstrates vLLM's attention to performance even in seemingly simple data structures.
