# File: `unsloth/kernels/rope_embedding.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 465 |
| Classes | `Fast_RoPE_Embedding`, `Fast_RoPE_Embedding_QK`, `Slow_RoPE_Embedding` |
| Functions | `fast_rope_embedding`, `inplace_rope_embedding` |
| Imports | device_type, torch, triton, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Optimized Rotary Position Embedding (RoPE) implementation for encoding positional information in transformers.

**Mechanism:** Implements RoPE via rotation in complex space: splits head dimension in half, applies rotation matrix [cos, -sin; sin, cos] element-wise. The formula is: Q * cos + rotate_half(Q) * sin, where rotate_half swaps and negates the second half. Provides three implementations: Fast_RoPE_Embedding (standard, processes in groups of 4 heads), Fast_RoPE_Embedding_QK (simultaneous Q/K rotation with optional indices for non-contiguous positions), and Slow_RoPE_Embedding (fallback using torch ops). Backward pass inverts rotation by negating sin. Handles both training (with autograd) and inference modes. Supports multi-GPU synchronization.

**Significance:** RoPE is the dominant positional encoding method in modern LLMs (LLaMA, GPT-NeoX, PaLM), superior to absolute or learned embeddings for length extrapolation. This optimized implementation is crucial as RoPE is applied to every attention operation. The fused kernel reduces memory bandwidth by computing rotations in-place. The support for position indices enables advanced use cases like sparse attention patterns and positional interpolation. The grouping strategy (ROPE_GROUP_SIZE=4) balances parallelism and memory access patterns.
