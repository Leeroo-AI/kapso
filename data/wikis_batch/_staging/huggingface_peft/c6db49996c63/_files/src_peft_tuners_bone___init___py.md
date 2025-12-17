# File: `src/peft/tuners/bone/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 24 |
| Imports | config, layer, model, peft |

## Understanding

**Status:** ✅ Explored

**Purpose:** Module initialization and registration for BONE (Block Affine) method

**Mechanism:** Imports and exposes BoneConfig, BoneLayer, BoneLinear, and BoneModel classes, then registers "bone" as a PEFT method using register_peft_method

**Significance:** Entry point for the BONE tuner, enabling block-wise affine transformations as a PEFT technique (deprecated in v0.19.0, replaced by MISS)
