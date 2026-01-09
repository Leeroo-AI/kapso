# File: `unsloth/kernels/geglu.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 290 |
| Functions | `geglu_exact_forward_kernel`, `geglu_exact_backward_kernel`, `geglu_approx_forward_kernel`, `geglu_approx_backward_kernel` |
| Imports | torch, triton, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** GELU gated linear unit kernels (exact and approximate variants)

**Mechanism:** Provides four Triton kernels for GeGLU activation: exact forward (f=0.5*e*(1+erf(e/sqrt(2)))*g), exact backward (uses derivative with erf and exp), approximate forward (f=0.5*e*(1+tanh(sqrt(2/pi)*e*(1+0.044715*e^2)))*g), and approximate backward (uses tanh derivative sech^2=1-tanh^2). Handles large tensors via LONG_INDEXING for >2^31 elements. Returns fused gate*up result in single kernel call

**Significance:** Essential for models using GeGLU activations (some transformer variants). Exact version matches HuggingFace precision, approximate is faster. Fusing gate and up projections reduces memory bandwidth. Used by fast_lora.py for GeGLU-based MLPs via apply_lora_mlp_geglu_exact/approx
