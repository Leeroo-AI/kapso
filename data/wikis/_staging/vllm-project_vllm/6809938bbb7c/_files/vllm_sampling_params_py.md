# File: `vllm/sampling_params.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 597 |
| Classes | `SamplingType`, `StructuredOutputsParams`, `RequestOutputKind`, `SamplingParams`, `BeamSearchParams` |
| Imports | copy, dataclasses, enum, functools, msgspec, pydantic, typing, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive sampling configuration for text generation.

**Mechanism:** `SamplingParams` is a msgspec Struct with 30+ parameters controlling generation: temperature, top-p, top-k, min-p for randomness; presence/frequency/repetition penalties; max/min tokens; stop strings/tokens; logprobs settings; and structured outputs. `StructuredOutputsParams` enables JSON schema, regex, grammar, and choice constraints. Extensive validation in `_verify_args()` ensures parameter consistency. Supports greedy (temp=0), random, and seeded sampling via `SamplingType` enum. `BeamSearchParams` provides separate config for beam search. Includes helper methods like `update_from_generation_config()` for HuggingFace compatibility and `clone()` for copying.

**Significance:** This is the primary user-facing API for controlling generation behavior. The parameter set mirrors OpenAI's API while adding vLLM-specific features like structured outputs and advanced sampling techniques. The validation prevents common mistakes and provides clear error messages. This class demonstrates vLLM's goal of being a production-ready, user-friendly inference engine with extensive control over generation quality vs speed tradeoffs.
