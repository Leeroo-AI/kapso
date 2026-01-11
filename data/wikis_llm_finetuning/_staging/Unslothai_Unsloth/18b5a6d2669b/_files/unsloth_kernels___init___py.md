# File: `unsloth/kernels/__init__.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 73 |
| Imports | cross_entropy_loss, fast_lora, flex_attention, fp8, geglu, layernorm, os, rms_layernorm, rope_embedding, swiglu, ... +1 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Kernel import hub - exports all kernel functions for centralized access

**Mechanism:** Imports and re-exports kernel functions from specialized modules (cross_entropy_loss, rms_layernorm, layernorm, rope_embedding, swiglu, geglu, fast_lora, fp8, utils, flex_attention). Patches FP8 linear layers on import and displays initialization message

**Significance:** Central entry point for Unsloth's optimized Triton kernels. Provides clean API and ensures FP8 patches are applied before model creation. Essential for organizing performance-critical operations
