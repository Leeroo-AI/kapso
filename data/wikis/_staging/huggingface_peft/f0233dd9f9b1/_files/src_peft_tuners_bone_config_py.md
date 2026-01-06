# File: `src/peft/tuners/bone/config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 129 |
| Classes | `BoneConfig` |
| Imports | __future__, dataclasses, peft, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Bone configuration dataclass

**Mechanism:** Defines BoneConfig with r (rank, preferably even), target_modules, init_weights (bool/"bat" for Bone/Bat variants), layers_to_transform, and bias type. Validates layers_pattern/layers_to_transform compatibility. Issues deprecation warning for v0.19.0 removal.

**Significance:** Core configuration for Bone/Bat adaptation methods. init_weights selector enables Bone (True) vs. Bat (block-structured) variants. Deprecated in favor of MissConfig.
