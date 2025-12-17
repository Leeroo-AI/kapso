# File: `src/peft/tuners/vblora/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 24 |
| Imports | config, layer, model, peft |

## Understanding

**Status:** ✅ Explored

**Purpose:** Module initialization and registration for VBLoRA (Vector Bank LoRA) tuning method.

**Mechanism:** Exports VBLoRA components (VBLoRAConfig, VBLoRALayer, Linear, VBLoRAModel) and registers "vblora" as a PEFT method using register_peft_method.

**Significance:** Entry point for VBLoRA, a parameter-efficient method that uses a shared vector bank with top-k selection to construct LoRA matrices. Enables weight sharing across adapter parameters for extreme compression.
