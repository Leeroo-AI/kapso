# File: `unsloth/kernels/__init__.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 73 |
| Imports | cross_entropy_loss, fast_lora, flex_attention, fp8, geglu, layernorm, os, rms_layernorm, rope_embedding, swiglu, ... +1 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Central export hub for all Unsloth kernel optimizations

**Mechanism:** Imports and re-exports fast kernel implementations from submodules (cross_entropy_loss, rms_layernorm, layernorm, rope_embedding, swiglu, geglu, fast_lora, fp8, flex_attention, utils). Prints initialization message about "2x faster free finetuning" when imported.

**Significance:** Core entry point for Unsloth's optimized Triton kernels that accelerate transformer training and inference operations. This file makes all kernel functions accessible through a single import point and establishes the kernel module namespace.

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
