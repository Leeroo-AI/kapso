# File: `src/peft/tuners/adalora/layer.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 360 |
| Classes | `AdaLoraLayer`, `SVDLinear`, `RankAllocator` |
| Imports | packaging, peft, torch, transformers, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** AdaLora layer implementations

**Mechanism:** AdaLoraLayer extends LoraLayer with SVD decomposition (lora_E, lora_A, lora_B, ranknum parameters). SVDLinear implements forward/merge/unmerge with delta weight computation via (lora_B @ (lora_A * lora_E)). RankAllocator manages importance scoring (sensitivity × uncertainty), budget scheduling (cubic decay), and mask-based rank pruning.

**Significance:** Core layer implementation for AdaLora. RankAllocator is critical for adaptive rank allocation using importance-aware budget distribution. Essential for dynamic parameter efficiency.
