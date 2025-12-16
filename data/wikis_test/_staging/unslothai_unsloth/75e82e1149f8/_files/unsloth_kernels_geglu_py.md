# File: `unsloth/kernels/geglu.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 290 |
| Functions | `geglu_exact_forward_kernel`, `geglu_exact_backward_kernel`, `geglu_approx_forward_kernel`, `geglu_approx_backward_kernel` |
| Imports | torch, triton, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Optimized GEGLU (Gaussian Error Gated Linear Unit) activation function implementations.

**Mechanism:** Provides two variants: exact (using erf function) and approximate (using tanh). The exact version computes f = 0.5 * e * (1 + erf(e/√2)) following the GELU definition precisely. The approximate version uses f = 0.5 * e * (1 + tanh(√(2/π) * e * (1 + 0.044715 * e²))) for faster computation. Both implement custom backward passes that compute derivatives efficiently: exact uses df/de = 0.5(1 + erf(...)) + 1/√(2π) * e * exp(-0.5e²), while approximate reuses intermediate tanh values. Handles long indexing (>2³¹ elements) via int64 offsets.

**Significance:** GEGLU is used in transformer MLP layers for models like GLM and some variants of GPT. The gating mechanism (element-wise multiplication with activated gate projection) improves model capacity. This optimized implementation fuses the activation and gating operations, reducing memory bandwidth requirements. The availability of both exact and approximate versions allows trading off between accuracy and speed. The careful handling of numerical stability and large tensor support makes it production-ready.
