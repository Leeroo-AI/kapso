# File: `unsloth/kernels/__init__.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 73 |
| Imports | cross_entropy_loss, fast_lora, flex_attention, fp8, geglu, layernorm, os, rms_layernorm, rope_embedding, swiglu, ... +1 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Module exports and initialization point

**Mechanism:** Imports and re-exports all kernel functions and classes from submodules. Serves as the entry point for the entire kernels package, making optimized operations available when imported. Includes a startup message about Unsloth's patching capabilities.

**Significance:** Provides a clean public API for accessing all optimized kernel implementations. Centralizes kernel functionality under a single import namespace, simplifying integration with the broader Unsloth framework.
