# File: `unsloth/kernels/__init__.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 73 |
| Imports | cross_entropy_loss, fast_lora, flex_attention, fp8, geglu, layernorm, os, rms_layernorm, rope_embedding, swiglu, ... +1 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Central module initialization that exposes all optimized Triton kernel implementations for fast transformer operations.

**Mechanism:** Imports and exports key functions from specialized kernel modules including cross-entropy loss, normalization layers, activation functions, LoRA operations, RoPE embeddings, and FP8 quantization. Displays a startup message informing users about the performance patching capability.

**Significance:** This is the main entry point for Unsloth's kernel-level optimizations. It consolidates all custom Triton kernels and patches that enable 2x faster finetuning by replacing standard PyTorch/Transformers operations with optimized GPU kernels. The module ensures proper initialization order, especially for FP8 operations that must patch forward functions before model creation.
