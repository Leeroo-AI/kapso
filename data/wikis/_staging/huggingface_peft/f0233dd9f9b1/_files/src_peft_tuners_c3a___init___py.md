# File: `src/peft/tuners/c3a/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 23 |
| Imports | config, layer, model, peft |

## Understanding

**Status:** ✅ Explored

**Purpose:** C3A module initialization

**Mechanism:** Exports C3AConfig, C3ALayer, C3ALinear, C3AModel classes and registers "c3a" as PEFT method for circular convolution-based adaptation.

**Significance:** Entry point for C3A (Circular Convolution Adapter) tuning method from paper (https://huggingface.co/papers/2407.19342). Enables efficient parameter updates via block circular convolution operations.
