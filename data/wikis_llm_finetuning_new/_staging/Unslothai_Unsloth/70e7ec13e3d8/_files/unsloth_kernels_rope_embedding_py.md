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

**Purpose:** Implements fast Triton kernels for Rotary Position Embedding (RoPE) computation, a key component of position encoding in modern transformer architectures.

**Mechanism:** Provides multiple RoPE implementations: (1) _rope_embedding applies rotation to Q tensors: Q_new[0:d/2] = Q[0:d/2]*cos - Q[d/2:d]*sin, Q_new[d/2:d] = Q[d/2:d]*cos + Q[0:d/2]*sin. Uses ROPE_GROUP_SIZE=4 for coalesced memory access across heads. (2) _rope_embedding_QK fuses Q and K rotation in single kernel, handles GQA (fewer K heads than Q heads). Supports custom rope_embedding_indices for non-sequential position patterns (used by TRL). Backward pass: negates sin for inverse rotation. Fast_RoPE_Embedding and Fast_RoPE_Embedding_QK wrap as autograd.Functions. Slow_RoPE_Embedding provides PyTorch fallback using torch.cat for rotate_half. fast_rope_embedding() selects appropriate implementation. Handles multi-GPU synchronization via torch_device_stream().synchronize().

**Significance:** RoPE is used by Llama, Mistral, Qwen, and most modern LLMs for position encoding. The Triton implementation avoids creating intermediate tensors for rotate_half operation, reducing memory and improving speed significantly.
