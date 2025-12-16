# File: `unsloth/kernels/geglu.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 290 |
| Functions | `geglu_exact_forward_kernel`, `geglu_exact_backward_kernel`, `geglu_approx_forward_kernel`, `geglu_approx_backward_kernel` |
| Imports | torch, triton, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** GeGLU activation function kernels

**Mechanism:** Implements exact and approximate GeGLU kernels using Triton. Exact version uses error function (erf) for precise GELU approximation. Approximate version uses tanh-based approximation with coefficient 0.044715. Each includes forward and backward passes with element-wise fusion. Automatically handles large tensors with 64-bit indexing when needed (>2B elements).

**Significance:** Optimizes GeGLU activation computation compared to naive PyTorch implementations. Fusing forward and backward passes reduces memory bandwidth and improves cache locality. Essential for models using GeGLU (common in modern transformers) to maintain Unsloth's 2x speedup goal.
