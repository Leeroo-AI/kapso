# File: `unsloth/kernels/swiglu.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 143 |
| Functions | `swiglu_fg_kernel`, `swiglu_DWf_DW_dfg_kernel` |
| Imports | torch, triton, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** SwiGLU activation function kernels

**Mechanism:** Implements SwiGLU (Swish-Gated Linear Unit) using Triton: f(x) = x * sigmoid(x), then h = f(gate) * up. Forward kernel fuses the gating and multiplication. Backward kernel (_DWf_DW_dfg_kernel) efficiently computes derivatives using sigmoid properties. Automatically selects between 32-bit and 64-bit indexing for large tensors.

**Significance:** SwiGLU is more effective than standard ReLU/GELU in large models, showing consistent improvements in language modeling. Optimized kernel implementation maintains Unsloth's 2x speedup goal. Fused forward/backward reduces memory traffic and improves cache efficiency.
