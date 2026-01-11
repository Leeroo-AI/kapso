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

**Purpose:** Rotary position embedding kernel (RoPE) with multiple implementation strategies

**Mechanism:** Provides three autograd functions: 1) Fast_RoPE_Embedding - groups heads (ROPE_GROUP_SIZE=4) for better parallelism, applies rotation Q*cos+rotate_half(Q)*sin to each group, 2) Fast_RoPE_Embedding_QK - fuses Q and K rotation in single kernel with optional position_ids, handles grouped query attention by processing only n_heads_K for K, 3) Slow_RoPE_Embedding - fallback using PyTorch ops. Backward pass negates sin for inverse rotation. Supports both regular sequence indexing and custom rope_embedding_indices (for TRL)

**Significance:** Critical for positional encoding in Llama/GPT-style models. Triton implementation significantly faster than PyTorch by fusing operations and exploiting parallelism across heads. Handles edge cases like Gemma (requires float32 RoPE despite bfloat16 model) and multi-GPU synchronization. Marked with torch.compiler.disable due to compilation issues. 10% faster kernel from HuyNguyen-hust optimization
