# File: `src/peft/tuners/adalora/config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 108 |
| Classes | `AdaLoraConfig` |
| Imports | dataclasses, peft, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Configuration class for AdaLoRA with three training phases (warmup, reduction, fine-tuning)

**Mechanism:** AdaLoraConfig extends LoraConfig with adaptive parameters: target_r (target rank), init_r (initial rank), tinit/tfinal (phase boundaries), deltaT (allocation interval), beta1/beta2 (EMA smoothing), orth_reg_weight (orthogonality regularization). Validates phase scheduling ensures tinit < total_step - tfinal

**Significance:** Defines the adaptive rank allocation schedule that distinguishes AdaLoRA from standard LoRA - critical for dynamic parameter budget management across training phases with importance-based pruning
