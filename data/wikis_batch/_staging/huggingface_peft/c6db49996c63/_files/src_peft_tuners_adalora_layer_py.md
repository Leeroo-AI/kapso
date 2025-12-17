# File: `src/peft/tuners/adalora/layer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 360 |
| Classes | `AdaLoraLayer`, `SVDLinear`, `RankAllocator` |
| Imports | packaging, peft, torch, transformers, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Core AdaLoRA layer implementations and rank allocation logic

**Mechanism:** AdaLoraLayer maintains lora_A (right singular vectors), lora_E (singular values), lora_B (left singular vectors), and ranknum parameters. SVDLinear applies these in forward passes with merge/unmerge support. RankAllocator computes importance scores using gradient-based sensitivity (EMA smoothing with beta1/beta2), implements cubic budget scheduling, and masks low-importance triplets using k-th value thresholding

**Significance:** Implements the mathematical foundation of AdaLoRA - SVD-based parameterization enables fine-grained pruning of singular values, while RankAllocator provides dynamic rank adjustment based on parameter importance throughout training phases
