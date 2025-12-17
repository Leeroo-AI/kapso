# File: `unsloth/utils/attention_dispatch.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 283 |
| Classes | `AttentionConfig`, `AttentionContext` |
| Functions | `select_attention_backend`, `run_attention` |
| Imports | __future__, dataclasses, models, torch, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Dynamically selects and executes optimal attention implementation based on hardware availability and input characteristics.

**Mechanism:** Implements AttentionConfig and AttentionContext dataclasses; select_attention_backend() prioritizes FlashAttention > xFormers > SDPA based on availability; run_attention() dispatches to appropriate backend with format conversions and masking logic. Handles varlen (packed) vs dense batches, supports grouped query attention (GQA), and applies causal/sliding-window masking per backend.

**Significance:** Critical optimization layer enabling efficient attention computation across diverse hardware (CUDA with various kernels, CPU), handling complexity of packed sequence metadata and gradient computation modes.
