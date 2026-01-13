# File: `unsloth/kernels/geglu.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 290 |
| Functions | `geglu_exact_forward_kernel`, `geglu_exact_backward_kernel`, `geglu_approx_forward_kernel`, `geglu_approx_backward_kernel` |
| Imports | torch, triton, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements Triton kernels for GEGLU (Gaussian Error Gated Linear Unit) activation with both exact (erf-based) and approximate (tanh-based) variants for forward and backward passes.

**Mechanism:** Provides four kernel pairs: (1) _exact_forward_kernel computes f = 0.5 * e * (1 + erf(e/sqrt(2))), h = f * g where e is gate and g is up projection. (2) _exact_backward_kernel computes derivatives: df/de = 0.5 * (1 + erf(e/sqrt(2))) + e * exp(-0.5*e^2) / sqrt(2*pi), then stores h, df=DW*f, de=DW*g*df/de in the input buffers for memory reuse. (3) _approx_forward_kernel uses the tanh approximation: f = 0.5 * e * (1 + tanh(sqrt(2/pi) * e * (1 + 0.044715*e^2))). (4) _approx_backward_kernel computes the approximation's derivative using sech^2 = 1 - tanh^2 identity. All kernels handle large tensors via LONG_INDEXING flag for int64 offsets when element count exceeds int32 safety buffer.

**Significance:** MLP optimization for models using GELU-gated activations (like GPT-J, PaLM). Fusing GEGLU into a single kernel avoids multiple memory round-trips. The exact/approx variants match different model implementations (some use tanh approximation for speed).
