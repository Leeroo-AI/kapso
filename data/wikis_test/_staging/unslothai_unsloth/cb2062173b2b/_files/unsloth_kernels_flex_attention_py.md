# File: `unsloth/kernels/flex_attention.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 187 |
| Functions | `slow_inference_attention_softcapping` |
| Imports | functools, os, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides attention mechanisms with logit softcapping support for models like Gemma 2. Conditionally uses PyTorch 2.5+ flex_attention when available, otherwise falls back to torch.compile implementations. Enables efficient attention with custom score modifications and masking.

**Mechanism:** Detects PyTorch 2.5+ flex_attention support. If available, creates wrapper functions using `flex_attention` with `tanh_softcap` score modifier and block masks for causal/sliding window attention via `create_flex_attention_causal_mask` and `create_flex_attention_sliding_window_mask`. If unavailable, provides `slow_attention_softcapping` compiled with torch.compile that manually implements grouped query attention expansion, applies softcapping `t * tanh(A/t)`, adds causal mask, and computes softmax. Also provides inference-specific version `slow_inference_attention_softcapping` without compilation overhead.

**Significance:** Essential for supporting Gemma 2's logit softcapping feature, which stabilizes attention computation. The flex_attention path leverages PyTorch's latest optimizations when available, while the fallback ensures compatibility with older PyTorch versions. Softcapping prevents attention scores from exploding, improving training stability. The block mask functions enable efficient causal and sliding window attention patterns.
