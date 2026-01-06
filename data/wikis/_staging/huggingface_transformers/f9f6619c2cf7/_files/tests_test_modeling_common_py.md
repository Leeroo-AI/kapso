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

**Purpose:** Core test infrastructure providing the foundational `ModelTesterMixin` class that validates model implementations across all Transformers architectures.

**Mechanism:** Massive test suite (4372 lines) that validates model behavior through comprehensive tests including: forward pass correctness, save/load mechanisms (PyTorch, safetensors), attention implementations (SDPA vs eager, Flash Attention equivalence), quantization (BitsAndBytes, GPTQ), device placement, tensor parallelism, gradient checkpointing, initialization, head masking, hidden states, training mode, batch processing, determinism, parameter tying, torchscript compatibility, and model pruning. Includes helper functions like `ids_tensor`, `floats_tensor`, `random_attention_mask` for test data generation and `sdpa_kernel` context manager for attention backend control. The `_test_eager_matches_sdpa_inference` function with detailed tolerance specifications ensures SDPA produces equivalent outputs to eager attention.

**Significance:** The most critical testing file in the repository - ensures every model meets quality standards and behaves consistently. Provides a contract that all models must fulfill, enabling confident addition of new architectures. Tests cover production-critical aspects like numerical stability, serialization, optimization compatibility, and edge cases. Without this infrastructure, maintaining quality across 100+ model architectures would be impossible.
