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

**Purpose:** Optimized Rotary Position Embedding for transformer attention

**Mechanism:** Implements RoPE via Triton kernels: splits head_dim in half, applies rotation Q' = Q*cos - rotate_half(Q)*sin. Provides three implementations: (1) Fast_RoPE_Embedding for separate Q/K processing, (2) Fast_RoPE_Embedding_QK for joint Q/K rotation with optional position indices, (3) Slow_RoPE_Embedding fallback using pure PyTorch. Groups multiple attention heads (ROPE_GROUP_SIZE=4) for better parallelism. Backward pass simply negates sin for inverse rotation. Handles multi-GPU synchronization when needed.

**Significance:** RoPE is the dominant position encoding in modern LLMs (Llama, Mistral, etc). Called at every attention layer during forward and backward passes. Fusing the rotation operations and processing multiple heads together reduces kernel launch overhead. The kernel accounts for 5-10% of total training time, making optimization worthwhile.

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
