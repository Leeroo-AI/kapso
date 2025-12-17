# File: `src/peft/tuners/adalora/model.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 346 |
| Classes | `AdaLoraModel` |
| Imports | gptq, layer, peft, torch, transformers, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Main AdaLoRA model class coordinating adaptive rank training

**Mechanism:** AdaLoraModel extends LoraModel, creates SVD layers (SVDLinear/SVDQuantLinear/SVDLinear8bitLt/SVDLinear4bit), maintains single trainable RankAllocator. Forward pass adds orthogonal regularization loss (||A@A.T - I||). update_and_allocate() method calls RankAllocator to update importance scores and adjust ranks per training phase. Supports resize_modules_by_rank_pattern for pruning

**Significance:** Orchestrates the complete AdaLoRA workflow - manages adapter injection, enforces single-trainable-adapter constraint, coordinates importance tracking and rank adjustment, making adaptive parameter-efficient fine-tuning practical for large language models
