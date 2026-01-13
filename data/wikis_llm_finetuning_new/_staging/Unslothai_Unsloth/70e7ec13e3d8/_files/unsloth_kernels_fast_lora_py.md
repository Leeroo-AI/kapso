# File: `unsloth/kernels/fast_lora.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 730 |
| Classes | `LoRA_MLP`, `LoRA_QKV`, `LoRA_W` |
| Functions | `apply_lora_mlp_swiglu`, `apply_lora_mlp_geglu_exact`, `apply_lora_mlp_geglu_approx`, `apply_lora_qkv`, `apply_lora_o`, `fast_lora_forward` |
| Imports | geglu, swiglu, torch, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements fused forward and backward passes for LoRA (Low-Rank Adaptation) layers in MLP and attention modules, minimizing memory usage and maximizing GPU efficiency.

**Mechanism:** Defines three main autograd.Function classes: (1) LoRA_MLP handles gate/up/down projections with fused SwiGLU/GeGLU activations. Forward: computes e=X@G, g=X@U, h=activation(e,g), i=h@W with LoRA additions. Backward: computes all 6 LoRA gradients (dA, dB for each projection) plus dX using efficient matmul chains. (2) LoRA_QKV handles fused Q/K/V projections with 6 LoRA weight gradients. (3) LoRA_W handles single projection (like output projection). Each uses matmul_lora() for quantized base weights + LoRA computation. Helper functions (apply_lora_mlp_swiglu, apply_lora_mlp_geglu_*, apply_lora_qkv, apply_lora_o) extract LoRA parameters from peft modules and invoke the appropriate autograd function. Uses inplace operations where possible to reduce memory allocations.

**Significance:** Core LoRA training component. Fusing all LoRA computations into single autograd functions avoids materializing intermediate activation tensors, dramatically reducing GPU memory during backpropagation - the key to Unsloth's memory efficiency claims.
