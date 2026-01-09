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

**Purpose:** LoRA forward/backward autograd functions with fused QKV and MLP operations

**Mechanism:** Implements three custom autograd functions (LoRA_MLP, LoRA_QKV, LoRA_W) that fuse base weight matmuls with LoRA adapter computations. For MLP: computes gate/up/down projections with LoRA adapters in single pass, supporting SwiGLU and GeGLU activations. For attention: fuses Q/K/V projections with their respective LoRA adapters. Uses matmul_lora for efficient quantized weight handling

**Significance:** Core LoRA optimization that significantly reduces memory and improves speed by fusing operations. Eliminates redundant memory reads/writes. Supports 4-bit/8-bit quantization via fast_dequantize. Essential for efficient LoRA fine-tuning with Unsloth's 2x speedup claims
