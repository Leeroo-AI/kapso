# File: `unsloth/kernels/swiglu.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 143 |
| Functions | `swiglu_fg_kernel`, `swiglu_DWf_DW_dfg_kernel` |
| Imports | torch, triton, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** SwiGLU activation forward/backward kernels (Swish-Gated Linear Unit)

**Mechanism:** Implements two Triton kernels: 1) _fg_kernel (forward) - computes f=e*sigmoid(e), then h=f*g (fused gate*up), 2) _DWf_DW_dfg_kernel (backward) - computes derivatives: df=DW*f, dg=DW*g, de=dg*sigmoid(e)*(1+e*(1-sigmoid(e))), stores results back in input buffers (h, df, de) for memory efficiency. Handles large tensors via LONG_INDEXING for >2^31 elements

**Significance:** Essential for Llama and many modern transformer MLPs which use SwiGLU (f=silu(gate)*up) instead of simple ReLU. Fusing operations in Triton reduces memory bandwidth and improves speed. Used by fast_lora.py's apply_lora_mlp_swiglu for efficient LoRA+SwiGLU computation. Part of Unsloth's core performance optimizations for popular model architectures
