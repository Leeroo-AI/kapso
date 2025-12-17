# File: `unsloth/kernels/__init__.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 73 |
| Imports | cross_entropy_loss, fast_lora, flex_attention, fp8, geglu, layernorm, os, rms_layernorm, rope_embedding, swiglu, ... +1 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Package initialization that exports optimized kernel functions for loss computation, layer normalization, embeddings, and LoRA operations.

**Mechanism:** Central hub importing and re-exporting specialized Triton kernels (cross-entropy, RMS-LayerNorm, RoPE, SwiGLU, GeGLU, LoRA, FP8, and flex attention). Includes conditional imports from unsloth_zoo for loss functions and optional patching setup.

**Significance:** Critical export point that enables all downstream performance optimizations by making kernels available to the patching system and model initialization pipelines.
