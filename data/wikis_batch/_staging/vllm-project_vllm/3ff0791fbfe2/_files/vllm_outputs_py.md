# File: `vllm/outputs.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 345 |
| Classes | `CompletionOutput`, `PoolingOutput`, `RequestOutput`, `PoolingRequestOutput`, `EmbeddingOutput`, `EmbeddingRequestOutput`, `ClassificationOutput`, `ClassificationRequestOutput`, `ScoringOutput`, `ScoringRequestOutput` |
| Imports | collections, dataclasses, torch, typing, typing_extensions, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Output data structures

**Mechanism:** Defines the output formats returned by vLLM inference. RequestOutput represents completion results with prompt, generated text, token IDs, logprobs, and metrics. CompletionOutput handles individual sequences within a request. For pooling models: PoolingOutput (base), EmbeddingOutput (vector embeddings), ClassificationOutput (probability distributions), and ScoringOutput (scalar scores). Each has corresponding RequestOutput wrappers. Supports streaming with output aggregation via the add() method.

**Significance:** Core public API that defines what users receive from vLLM. These structures must maintain backward compatibility as they're part of the stable interface. Different output types support diverse model tasks: text generation, embeddings, classification, and scoring. Critical for both synchronous and streaming API patterns.
