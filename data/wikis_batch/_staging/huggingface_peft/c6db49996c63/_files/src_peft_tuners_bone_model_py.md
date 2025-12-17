# File: `src/peft/tuners/bone/model.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 126 |
| Classes | `BoneModel` |
| Imports | layer, peft, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Top-level model wrapper that applies BONE adapters to pretrained models

**Mechanism:** Extends BaseTuner to create and replace linear layers with BoneLinear layers, managing adapter injection and lifecycle

**Significance:** Main interface for applying BONE/BAT-style block affine transformations to pretrained models, supporting only torch.nn.Linear layers
