# File: `unsloth/kernels/swiglu.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 143 |
| Functions | `swiglu_fg_kernel`, `swiglu_DWf_DW_dfg_kernel` |
| Imports | torch, triton, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements SwiGLU activation (f = e*sigmoid(e), h = f*g) with optimized forward and backward Triton kernels for efficient gating.

**Mechanism:** _fg_kernel computes f = e*sigmoid(e) = e/(1+exp(-e)) and h = f*g in single pass. _DWf_DW_dfg_kernel computes backward derivatives: df = DW*f, dg = DW*g, de = dg*sigmoid(e)*(1+e*(1-sigmoid(e))). Uses fast sigmoid computation via tl.sigmoid. Handles long indexing (>2^31 elements) with conditional int64 logic.

**Significance:** Provides 1.5-2x speedup for SwiGLU activation through kernel fusion, reducing MLP bottleneck in LLaMA and similar models using this gating mechanism.
