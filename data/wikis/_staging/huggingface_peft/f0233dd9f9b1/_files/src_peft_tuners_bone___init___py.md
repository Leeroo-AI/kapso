# File: `src/peft/tuners/bone/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 24 |
| Imports | config, layer, model, peft |

## Understanding

**Status:** ✅ Explored

**Purpose:** Bone module initialization

**Mechanism:** Exports BoneConfig, BoneLayer, BoneLinear, BoneModel classes and registers "bone" as PEFT method for Householder reflection-based adaptation.

**Significance:** Entry point for Bone tuning method from paper (https://huggingface.co/papers/2409.15371). Note: Deprecated in v0.19.0, replaced by MissConfig. Enables parameter-efficient adaptation via block-structured updates.
