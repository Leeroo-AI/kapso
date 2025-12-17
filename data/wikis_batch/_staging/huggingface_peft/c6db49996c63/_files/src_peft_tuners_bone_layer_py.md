# File: `src/peft/tuners/bone/layer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 352 |
| Classes | `BoneLayer`, `BoneLinear` |
| Imports | math, peft, torch, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implementation of BONE layer adaptation for linear layers

**Mechanism:** Stores rank-based block parameters (bone_block), applies block-wise transformations by reshaping inputs and summing across block dimensions, supports both standard BONE (adding blocks to reshaped input) and BAT variant (matrix multiplication with blocks)

**Significance:** Core layer implementation enabling parameter-efficient adaptation through block-structured transformations with configurable rank
