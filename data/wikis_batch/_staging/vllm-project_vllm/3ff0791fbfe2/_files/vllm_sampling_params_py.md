# File: `vllm/sampling_params.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 597 |
| Classes | `SamplingType`, `StructuredOutputsParams`, `RequestOutputKind`, `SamplingParams`, `BeamSearchParams` |
| Imports | copy, dataclasses, enum, functools, msgspec, pydantic, typing, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Text generation parameters

**Mechanism:** Defines SamplingParams class with extensive generation controls: temperature, top_p, top_k, min_p (sampling), presence_penalty, frequency_penalty, repetition_penalty (penalties), max_tokens, min_tokens (length), stop strings/tokens, logprobs configuration, and structured outputs. Includes validation, default handling, and integration with tokenizers for bad words. StructuredOutputsParams enables constrained generation (JSON, regex, grammar, choice). BeamSearchParams for beam search decoding. SamplingType enum distinguishes greedy/random/seeded sampling.

**Significance:** Core user-facing API that controls all aspects of text generation. One of the most important public interfaces in vLLM. Follows OpenAI API conventions for compatibility. Extensive validation ensures parameter combinations are valid. The clone() method enables safe parameter reuse across requests. Structured outputs support is critical for tool calling and constrained generation use cases.
