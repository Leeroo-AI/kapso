# File: `src/peft/tuners/adalora/config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 108 |
| Classes | `AdaLoraConfig` |
| Imports | dataclasses, peft, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** AdaLora configuration dataclass definition

**Mechanism:** Extends LoraConfig with AdaLora-specific parameters including target_r (target rank), init_r (initial rank), tinit/tfinal (warmup phases), deltaT (budget allocation interval), beta1/beta2 (EMA hyperparameters), orth_reg_weight (orthogonal regularization), and total_step (training duration). Validates scheduling parameters and disables DoRA/LOFTQ compatibility.

**Significance:** Core configuration defining AdaLora's three-phase training strategy (initial warmup, rank reduction, final fine-tuning) and budget allocation mechanism. Essential for controlling adaptive rank behavior.
