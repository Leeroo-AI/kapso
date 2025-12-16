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

**Purpose:** Implements optimized Triton kernels for RoPE (Rotary Position Embedding) used in transformer attention mechanisms to encode positional information. Provides both fast Triton implementations and slower fallback versions.

**Mechanism:** Three autograd functions: (1) `Fast_RoPE_Embedding` - standalone RoPE for single tensor using `_rope_embedding` kernel that applies rotation `Q_new = [Q0*cos - Q1*sin, Q1*cos + Q0*sin]` to each head in groups of 4 for parallelism; (2) `Fast_RoPE_Embedding_QK` - simultaneously applies RoPE to both Q and K tensors using `_rope_embedding_QK` kernel, more efficient for attention where both need rotation. Supports optional position indices for sequence packing. Backward passes negate sin to reverse rotation. (3) `Slow_RoPE_Embedding` - torch.compile fallback using concatenation for older PyTorch versions or debugging. Multi-GPU synchronization when DEVICE_COUNT > 1.

**Significance:** RoPE is applied to Q and K in every attention layer of modern LLMs (Llama, Mistral, GPT-NeoX), making it very frequently called. The rotation operation involves element-wise multiplication with cos/sin values and rearranging tensor halves. Custom kernels fuse these operations avoiding intermediate tensors. The QK-fused version eliminates redundant cos/sin loads. Group processing (4 heads per kernel launch) balances parallelism and memory access. Essential for efficient attention computation. Compiler disabled due to issues with torch.compile.
