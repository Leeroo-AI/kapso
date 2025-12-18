# File: `vllm/_custom_ops.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 3116 |
| Classes | `CPUDNNLGEMMHandler` |
| Functions | `paged_attention_v1`, `paged_attention_v2`, `paged_attention_rocm`, `mla_decode_kvcache_cpu`, `merge_attn_states`, `convert_vertical_slash_indexes`, `convert_vertical_slash_indexes_mergehead`, `rotary_embedding`, `... +121 more` |
| Imports | torch, typing, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Central registry of custom CUDA/ROCm operations for high-performance inference.

**Mechanism:** Defines over 120 custom operations implemented in C++/CUDA/ROCm for performance-critical tasks. Key operation categories include: (1) Attention mechanisms (paged attention v1/v2, flash attention, MLA), (2) Quantization operations (FP8, INT4, INT8, AWQ, GPTQ), (3) MoE (Mixture of Experts) operations, (4) Rotary embeddings and RoPE, (5) Matrix operations (GEMM, scaled_mm), (6) Normalization (RMSNorm), (7) Activation functions (GELU, SiLU, Swiglu), (8) KV cache operations. The `CPUDNNLGEMMHandler` class provides CPU-based implementations. All operations are registered with PyTorch's custom op system.

**Significance:** Core performance layer of vLLM. These custom operations are what make vLLM fast - they implement highly optimized CUDA kernels for transformer inference. This file acts as the Python interface to C++ implementations, bridging Python-land with optimized GPU code. Critical for achieving state-of-the-art inference performance.
