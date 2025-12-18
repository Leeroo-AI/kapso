# File: `vllm/forward_context.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 358 |
| Classes | `BatchDescriptor`, `DPMetadata`, `ForwardContext` |
| Functions | `get_forward_context`, `is_forward_context_available`, `create_forward_context`, `override_forward_context`, `set_forward_context` |
| Imports | collections, contextlib, dataclasses, time, torch, typing, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Context management for model forward passes with metadata tracking.

**Mechanism:** Provides thread-local context for passing metadata during model forward execution. `BatchDescriptor` describes batch characteristics (tokens, requests, uniformity, LoRA) for CUDA graph dispatching. `DPMetadata` tracks data-parallel metadata including token counts across DP ranks and chunking information. `ForwardContext` holds attention metadata, virtual engine ID, CUDA graph mode, and DP metadata. Context managers `set_forward_context()` and `override_forward_context()` manage the global context state. Includes batch size tracking and logging for performance analysis. Supports chunked execution for MoE models with SP/EP parallelism.

**Significance:** Essential infrastructure for passing execution context through the model forward stack without explicit parameter passing. Enables CUDA graph optimization by providing batch descriptors for graph selection. Critical for data-parallel execution coordination and performance tracking. The context pattern keeps model code clean while providing necessary runtime information.
