# File: `unsloth/kernels/swiglu.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 143 |
| Functions | `swiglu_fg_kernel`, `swiglu_DWf_DW_dfg_kernel` |
| Imports | torch, triton, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides optimized Triton kernels for SwiGLU (Swish-Gated Linear Unit) activation function used in transformer MLP layers. Implements both forward and backward passes with element-wise fusion to minimize memory traffic.

**Mechanism:** Two Triton kernels: (1) `_fg_kernel` wrapped as `swiglu_fg_kernel` - computes forward pass `h = f * g` where `f = e * sigmoid(e)` is Swish activation, processing elements in blocks of 1024; (2) `_DWf_DW_dfg_kernel` wrapped as `swiglu_DWf_DW_dfg_kernel` - computes backward pass derivatives: `h = f * g`, `df = dW * f`, `de = dg * sigmoid(e) * (1 + e * (1 - sigmoid(e)))` all in single pass. Both kernels support long indexing for tensors exceeding 2^31 elements and reuse sigmoid computation to avoid redundant exp operations.

**Significance:** SwiGLU is the standard activation in modern LLMs (Llama, Mistral, PaLM), used in every MLP layer (typically 2/3 of model parameters). The gating mechanism `f * g` allows dynamic control of information flow. Custom kernels fuse activation and gating operations, avoiding materialization of intermediate tensors which would double memory usage. The backward kernel's triple-output design (h, df, de) enables further fusion in `fast_lora.py` where it's combined with LoRA operations. Essential for efficient MLP computation in training and inference.
