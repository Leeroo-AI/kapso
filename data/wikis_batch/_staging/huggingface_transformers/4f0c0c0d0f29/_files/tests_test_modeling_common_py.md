# File: `tests/test_modeling_common.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 4372 |
| Classes | `ModelTesterMixin`, `CopyClass` |
| Functions | `sdpa_kernel`, `compare_state_dicts`, `ids_tensor`, `random_attention_mask`, `floats_tensor` |
| Imports | collections, contextlib, copy, generation, inspect, math, numpy, os, packaging, parameterized, ... +8 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Massive common test suite for all transformer model architectures with 4300+ lines of tests.

**Mechanism:** ModelTesterMixin provides comprehensive tests including model initialization, forward pass, parameter tying, head pruning, gradient checkpointing, attention implementations (eager vs SDPA vs Flash Attention), mixed precision, quantization (bitsandbytes), device mapping, state dict loading, safetensors support, and model parallelism. Includes helper functions (ids_tensor, floats_tensor, random_attention_mask) and sdpa_kernel context manager for testing different attention backends.

**Significance:** Foundational testing infrastructure that every model in transformers inherits from, ensuring consistent behavior across 100+ model architectures including proper CUDA/CPU handling, quantization support, distributed training compatibility, and attention mechanism correctness.
