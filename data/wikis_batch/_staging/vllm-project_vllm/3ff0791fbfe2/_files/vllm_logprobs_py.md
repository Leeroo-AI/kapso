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

**Purpose:** Log probability data structures

**Mechanism:** Defines data structures for storing and managing log probabilities during generation. The Logprob dataclass stores (logprob, rank, decoded_token) tuples. FlatLogprobs provides a memory-efficient alternative to list[dict] by flattening logprob data into primitive-type lists with index tracking, significantly reducing GC overhead. Implements MutableSequence interface for backward compatibility. Helper functions create_prompt_logprobs, create_sample_logprobs, and append_logprobs_for_next_position manage logprob containers.

**Significance:** Critical for OpenAI API compatibility which requires returning log probabilities for generated tokens. The FlatLogprobs optimization is important for high-throughput serving where GC pressure from nested dictionaries can impact performance. Essential for applications requiring confidence scores, uncertainty estimation, or token-level analysis of model outputs.
