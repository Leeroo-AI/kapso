# File: `unsloth/kernels/swiglu.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 143 |
| Functions | `swiglu_fg_kernel`, `swiglu_DWf_DW_dfg_kernel` |
| Imports | torch, triton, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Fused SwiGLU activation for Llama/Mistral FFN layers

**Mechanism:** Two Triton kernels: (1) Forward kernel computes h = (e * sigmoid(e)) * g, where sigmoid is computed as 1/(1 + exp(-e)). (2) Backward kernel computes gradients for both gate and up projections: df/de = sigmoid(e) * (1 + e * (1 - sigmoid(e))). Operates on flattened tensors for efficiency. Handles large tensor sizes via int64 indexing when elements exceed 2^31.

**Significance:** SwiGLU (Swish-Gated Linear Unit) is the standard activation in Llama, Mistral, and most modern LLMs. It replaces the three separate operations (swish, multiply, matmul) with a single fused kernel. This is called twice per transformer layer (forward and backward) and provides measurable speedup by reducing memory traffic. Essential component of Unsloth's fast_lora MLP operations.

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
