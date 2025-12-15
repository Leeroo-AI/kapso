# File: `unsloth/kernels/fast_lora.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 717 |
| Classes | `LoRA_MLP`, `LoRA_QKV`, `LoRA_W` |
| Functions | `apply_lora_mlp_swiglu`, `apply_lora_mlp_geglu_exact`, `apply_lora_mlp_geglu_approx`, `apply_lora_qkv`, `apply_lora_o`, `fast_lora_forward` |
| Imports | geglu, swiglu, torch, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Fused LoRA operations for efficient parameter-efficient fine-tuning

**Mechanism:** Implements three custom autograd functions (LoRA_MLP, LoRA_QKV, LoRA_W) that fuse base weight operations with LoRA adapter computations in single kernels. Handles MLP blocks with SwiGLU/GEGLU activations by computing gate, up, and down projections together with LoRA weights. Supports quantized (4-bit/FP8) base weights via dequantization helpers. Forward pass computes Y = X @ (W + A @ B), backward computes gradients for both LoRA adapters and inputs.

**Significance:** Core component for Unsloth's fast LoRA training. By fusing base layer and LoRA operations, it eliminates redundant memory access and kernel launches. This is essential for making LoRA fine-tuning 2x faster while supporting quantized models.

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
