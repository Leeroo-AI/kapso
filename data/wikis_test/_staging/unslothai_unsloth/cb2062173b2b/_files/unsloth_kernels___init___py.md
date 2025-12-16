# File: `unsloth/kernels/__init__.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 73 |
| Imports | cross_entropy_loss, fast_lora, flex_attention, fp8, geglu, layernorm, os, rms_layernorm, rope_embedding, swiglu, ... +1 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Central initialization module for all custom Triton kernels that enable Unsloth's performance optimizations. Exports optimized implementations for core transformer operations and prints a startup message about enabling faster finetuning.

**Mechanism:** Imports and re-exports key functions from specialized kernel modules including cross-entropy loss, RMS/LayerNorm, RoPE embeddings, SwiGLU/GEGLU activations, LoRA operations, FP8 quantization, and flex attention. The FP8 module is imported with wildcard to patch FP8Linear forward functions before model creation. Displays user-facing message "Will patch your computer to enable 2x faster free finetuning" on startup.

**Significance:** This is the main entry point for Unsloth's performance-critical kernel implementations. It provides a unified API surface for all the custom Triton kernels that replace standard PyTorch/Transformers operations with faster GPU-optimized versions. Essential for making the optimized operations available throughout the codebase.
