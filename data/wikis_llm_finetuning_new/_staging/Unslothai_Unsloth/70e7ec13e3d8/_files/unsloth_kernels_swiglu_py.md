# File: `unsloth/kernels/swiglu.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 143 |
| Functions | `swiglu_fg_kernel`, `swiglu_DWf_DW_dfg_kernel` |
| Imports | torch, triton, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements Triton kernels for SwiGLU (Swish-Gated Linear Unit) activation function used in Llama, Mistral, and most modern LLM MLPs.

**Mechanism:** Provides two Triton JIT kernels: (1) _fg_kernel (forward): computes f = e * sigmoid(e) (the Swish function), then h = f * g where e is gate projection output and g is up projection output. Handles large tensors with LONG_INDEXING for int64 offsets. (2) _DWf_DW_dfg_kernel (backward): computes the derivative se = sigmoid(e), f = se * e, then de = dg * se * (1 + e * (1 - se)) following the SwiGLU derivative formula. Reuses input buffers for output (DW -> h, e -> df, g -> de) to minimize memory allocation. The wrapper functions swiglu_fg_kernel() and swiglu_DWf_DW_dfg_kernel() handle tensor reshaping and kernel launch configuration.

**Significance:** Core MLP activation optimization for Llama-family models. SwiGLU computes gate*sigmoid(gate)*up which requires multiple operations - fusing into one kernel reduces memory bandwidth by avoiding intermediate tensor materialization. Used by fast_lora.py for efficient MLP forward/backward.
