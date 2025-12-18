# File: `src/peft/tuners/lora/variants.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 722 |
| Classes | `DoraLinearVariant`, `ALoraLinearVariant`, `QALoraLinearVariant`, `ArrowLinearVariant` |
| Imports | __future__, arrow, math, peft, torch, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Advanced LoRA variant implementations

**Mechanism:** Implements specialized LoRA variants as extensions to standard LoRA. DoraLinearVariant adds weight normalization with magnitude vector for improved stability. ALoraLinearVariant implements Adaptive LoRA with learnable offset parameters for position-aware adaptation. QALoraLinearVariant quantizes LoRA weights for memory efficiency using group-wise quantization. ArrowLinearVariant implements mixture-of-experts routing across multiple LoRA adapters using prototype-based token routing with cosine similarity and top-k selection, including general knowledge subtraction for task purification.

**Significance:** Extends LoRA's capabilities beyond basic low-rank adaptation. DoRA improves training stability, ALoRA adds fine-grained control, QALoRA reduces memory, and Arrow enables multi-task learning with dynamic adapter routing. These variants make LoRA more versatile and effective for diverse fine-tuning scenarios.
