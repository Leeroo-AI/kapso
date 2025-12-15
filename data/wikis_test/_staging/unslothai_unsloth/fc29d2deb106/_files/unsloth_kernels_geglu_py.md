# File: `unsloth/kernels/geglu.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 290 |
| Functions | `geglu_exact_forward_kernel`, `geglu_exact_backward_kernel`, `geglu_approx_forward_kernel`, `geglu_approx_backward_kernel` |
| Imports | torch, triton, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Optimized GEGLU activation for transformer FFN layers

**Mechanism:** Provides two GEGLU implementations in Triton: (1) Exact version using erf for GELU computation: h = 0.5 * e * (1 + erf(e/√2)) * g, (2) Approximate version using tanh approximation for faster computation. Forward kernels fuse the gating and activation into single pass. Backward kernels compute gradients with respect to both gate and up projections, reusing intermediate values. Handles large tensors via int64 indexing when needed.

**Significance:** GEGLU (Gated GELU) is used in T5, PaLM, and other architectures as an alternative to SwiGLU. Fusing the gating operation with activation reduces memory traffic. The approximate version trades slight accuracy for speed, useful for inference or where precision is less critical.

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
