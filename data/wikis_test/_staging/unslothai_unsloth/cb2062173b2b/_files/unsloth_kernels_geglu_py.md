# File: `unsloth/kernels/geglu.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 290 |
| Functions | `geglu_exact_forward_kernel`, `geglu_exact_backward_kernel`, `geglu_approx_forward_kernel`, `geglu_approx_backward_kernel` |
| Imports | torch, triton, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides optimized Triton kernels for GEGLU (Gated GLU with GELU activation) used in transformer MLPs. Implements both exact GELU (using erf) and approximate GELU (using tanh) versions with custom forward and backward passes.

**Mechanism:** Four Triton kernels: (1) `_exact_forward_kernel` - computes `h = f * g` where `f = 0.5 * e * (1 + erf(e/sqrt(2)))` is exact GELU; (2) `_exact_backward_kernel` - computes gradients using `df/de = 0.5 * (1 + erf(e/sqrt(2))) + (1/sqrt(2*pi)) * e * exp(-0.5*e^2)`; (3) `_approx_forward_kernel` - uses faster tanh approximation `f = 0.5 * e * (1 + tanh(sqrt(2/pi) * e * (1 + 0.044715*e^2)))`; (4) `_approx_backward_kernel` - corresponding gradient with sech^2 = 1-tanh^2 reuse. All kernels support long indexing for tensors >2^31 elements and process elements in blocks of 1024.

**Significance:** GEGLU is a key component of modern transformer architectures (used in models like Google's T5, PaLM). The gating mechanism `f * g` allows the network to control information flow. Implementing both exact and approximate versions provides speed-accuracy tradeoffs. The custom kernels fuse the activation and gating operations, avoiding intermediate tensor materialization and providing substantial speedups over PyTorch's native implementation. Used by `fast_lora.py` for fused LoRA+GEGLU MLP operations.
