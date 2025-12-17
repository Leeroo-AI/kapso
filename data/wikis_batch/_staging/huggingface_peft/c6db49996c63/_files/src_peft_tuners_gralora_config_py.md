# File: `src/peft/tuners/gralora/config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 182 |
| Classes | `GraloraConfig` |
| Imports | dataclasses, peft, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines configuration parameters for GraLoRA (Gradient Low-Rank Adaptation) block-structured fine-tuning.

**Mechanism:** GraloraConfig extends PeftConfig with GraLoRA parameters: r (rank, must be divisible by gralora_k), gralora_k (number of subblocks, typically 2 for r<=32, 4 for r>=64), hybrid_r (vanilla LoRA rank for hybrid mode), alpha (scaling factor for alpha/(r+hybrid_r)), dropout, and standard PEFT options. Validates r % gralora_k == 0 in __post_init__.

**Significance:** Core configuration for block-wise low-rank adaptation. The gralora_k parameter increases expressivity by multiplying effective rank while preserving parameter count. Hybrid mode (hybrid_r > 0) combines GraLoRA's structured adaptation with vanilla LoRA's general updates for better flexibility.
