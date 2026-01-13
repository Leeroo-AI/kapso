# File: `unsloth/kernels/flex_attention.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 187 |
| Functions | `slow_inference_attention_softcapping` |
| Imports | functools, os, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides attention implementations with logit softcapping support, using PyTorch's flex_attention when available (torch >= 2.5) or falling back to a manual implementation.

**Mechanism:** When flex_attention is available (HAS_FLEX_ATTENTION=True), uses torch.nn.attention.flex_attention with custom score_mod functions for tanh softcapping and block masks for causal/sliding window patterns. The generate_tanh_softcap() creates a score modifier implementing t * tanh(x/t). Block masks are created via create_block_mask() with configurable causal_masker or sliding_window_masker. When flex_attention is unavailable, slow_attention_softcapping() implements manual grouped query attention: expands K,V for GQA, computes Q*K^T with proper scaling (using query_pre_attn_scalar), applies softcapping, adds causal mask, softmax, then V multiplication. Also provides slow_inference_attention_softcapping() as a non-compiled version for inference.

**Significance:** Required for Gemma 2 model support which uses attention logit softcapping. Flex attention provides optimized attention patterns when available, falling back gracefully for older PyTorch versions. Enables proper GQA handling across different attention head configurations.
