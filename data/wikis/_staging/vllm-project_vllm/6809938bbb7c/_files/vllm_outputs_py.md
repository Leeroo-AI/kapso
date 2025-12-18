# File: `vllm/outputs.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 345 |
| Classes | `CompletionOutput`, `PoolingOutput`, `RequestOutput`, `PoolingRequestOutput`, `EmbeddingOutput`, `EmbeddingRequestOutput`, `ClassificationOutput`, `ClassificationRequestOutput`, `ScoringOutput`, `ScoringRequestOutput` |
| Imports | collections, dataclasses, torch, typing, typing_extensions, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Output data structures for generation and pooling tasks.

**Mechanism:** Defines the complete output hierarchy for vLLM. `CompletionOutput` represents a single completion with text, token IDs, cumulative logprob, and finish reason. `RequestOutput` aggregates completions for a request with prompt info, logprobs, and metrics. Supports incremental updates via `add()` method. For pooling tasks: `PoolingOutput` holds tensor data, with specialized subclasses `EmbeddingOutput` (list of floats), `ClassificationOutput` (probability vector), and `ScoringOutput` (scalar score). Corresponding `*RequestOutput` classes wrap these with metadata. All classes are dataclasses for clean serialization.

**Significance:** Defines the public output API that users receive from vLLM. These structures must be stable and well-documented since they're part of the user-facing interface. The hierarchy cleanly separates generation (text) from pooling (embeddings/classification) tasks. The incremental update mechanism enables streaming responses. These classes bridge the internal inference engine with external APIs (OpenAI, custom).
