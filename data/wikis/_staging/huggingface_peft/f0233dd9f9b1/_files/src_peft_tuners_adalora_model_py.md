# File: `src/peft/tuners/adalora/model.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 346 |
| Classes | `AdaLoraModel` |
| Imports | gptq, layer, peft, torch, transformers, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** AdaLora model implementation

**Mechanism:** Extends LoraModel with SVD-based adaptation. Manages RankAllocator for dynamic rank adjustment, implements orthogonal regularization in forward pass, handles adapter creation for Linear/Conv1D/BNB/GPTQ layers, and provides update_and_allocate() method for budget scheduling. Supports resize_modules_by_rank_pattern() for rank pruning.

**Significance:** Core tuner class implementing AdaLora paper (https://openreview.net/forum?id=lq62uWRJjiY). Enables parameter-efficient training with adaptive rank allocation across layers. Central to AdaLora functionality.
