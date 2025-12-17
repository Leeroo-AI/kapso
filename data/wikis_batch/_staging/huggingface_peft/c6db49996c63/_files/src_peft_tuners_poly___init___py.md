# File: `src/peft/tuners/poly/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 24 |
| Imports | config, layer, model, peft |

## Understanding

**Status:** ✅ Explored

**Purpose:** Module initialization and registration for Poly (Polytropon) tuning method.

**Mechanism:** Exports Poly components (PolyConfig, PolyLayer, Linear, PolyModel) and registers "poly" as a PEFT method using register_peft_method.

**Significance:** Entry point for Poly, a multi-task learning method that maintains multiple LoRA "skills" per layer and learns to mix them via task-specific routing. Enables efficient multi-task adaptation with learned skill composition.
