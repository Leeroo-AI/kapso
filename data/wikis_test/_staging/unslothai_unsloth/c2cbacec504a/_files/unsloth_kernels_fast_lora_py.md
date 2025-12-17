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

**Purpose:** Implements efficient LoRA fine-tuning for MLP blocks (SwiGLU, GeGLU) and attention projections (Q,K,V,O) with fused forward and backward passes supporting quantized weights.

**Mechanism:** Specialized PyTorch autograd Functions (LoRA_MLP, LoRA_QKV, LoRA_W) that fuse multiple matrix multiplications and gating operations into single backward passes. Reuses intermediate activations (e,g,h) to compute gradients efficiently, minimizing memory. Supports quantized weights via fast_dequantize and matmul_lora utilities. Backward pass computes LoRA adapter gradients via efficient matrix operations.

**Significance:** Enables practical LoRA training with 2-3x memory savings and computation speedup through operator fusion. Critical for fitting large models in memory while maintaining training speed through in-place gradient computations where safe.
