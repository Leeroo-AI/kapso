# File: `unsloth/kernels/geglu.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 290 |
| Functions | `geglu_exact_forward_kernel`, `geglu_exact_backward_kernel`, `geglu_approx_forward_kernel`, `geglu_approx_backward_kernel` |
| Imports | torch, triton, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements exact and approximate GeGLU activation function with optimized forward and backward Triton kernels for efficient MLP gating.

**Mechanism:** Four Triton kernels handling forward/backward for exact (erf-based) and approximate (tanh-based) GeGLU: f = 0.5*e*(1+erf(e/sqrt(2))) or f = 0.5*e*(1+tanh(sqrt(2/pi)*e*(1+0.044715*e^2))). Kernels compute h = f*g output and derivatives (df/de, dg/dg). Handles long indexing (>2^31 elements) with int64 offsets. Pre-computed constants optimize computation.

**Significance:** Enables 2x faster GeGLU activation compared to PyTorch defaults through kernel fusion, reducing MLP bottleneck in transformer inference and training for Gemma and similar architectures.
