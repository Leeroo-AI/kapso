# File: `unsloth/kernels/__init__.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 73 |
| Imports | cross_entropy_loss, fast_lora, flex_attention, fp8, geglu, layernorm, os, rms_layernorm, rope_embedding, swiglu, ... +1 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Package initializer that exports all optimized GPU kernel functions and utilities from the kernels subpackage.

**Mechanism:** Imports and re-exports functions from submodules including cross-entropy loss (fast_cross_entropy_loss, patch_loss_functions), RMS/Layer normalization (fast_rms_layernorm, fast_layernorm with patching utilities), RoPE embeddings (fast_rope_embedding, inplace_rope_embedding), activation functions (swiglu/geglu kernels), LoRA operations (apply_lora_mlp_*, apply_lora_qkv, apply_lora_o), FP8 quantization support, flex attention with softcapping, and utility functions (fast_dequantize, fast_gemv, matmul_lora). Also prints a startup message indicating Unsloth will enable 2x faster finetuning.

**Significance:** Central entry point for all custom Triton/CUDA kernels. This is a core component that exposes all performance-critical GPU operations used throughout Unsloth's model patching system.
