# File: `src/peft/tuners/boft/model.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 131 |
| Classes | `BOFTModel` |
| Imports | layer, peft, torch, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Top-level model wrapper that applies BOFT adapters to pretrained models

**Mechanism:** Extends BaseTuner to create and replace target modules with BOFT layers (Linear or Conv2d), dispatching based on layer type and managing adapter lifecycle

**Significance:** Main interface for applying BOFT to pretrained models, handling the orchestration of adapter injection across model architecture
