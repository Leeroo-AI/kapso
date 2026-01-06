# File: `src/peft/tuners/bone/layer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 352 |
| Classes | `BoneLayer`, `BoneLinear` |
| Imports | math, peft, torch, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Bone layer implementations

**Mechanism:** BoneLayer stores bone_block parameter (r x out_features for Bone, out_features//r x r x r for Bat). BoneLinear implements get_delta_weight_bone() (adds bone_block to reshaped weights) and get_delta_weight() for Bat (block-wise @ operations with inverse). Forward pass adds bone_block contribution via sum/reshape operations.

**Significance:** Core layer implementing Householder reflection-based adaptation. Bone uses simple addition, Bat uses block-wise multiplication with inverse. Essential for parameter-efficient weight updates on Linear layers.
