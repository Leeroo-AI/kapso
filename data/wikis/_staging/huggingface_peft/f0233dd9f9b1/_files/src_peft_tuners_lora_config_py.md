# File: `src/peft/tuners/lora/config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 879 |
| Classes | `LoraRuntimeConfig`, `LoftQConfig`, `ArrowConfig`, `BdLoraConfig`, `EvaConfig`, `CordaConfig`, `LoraConfig` |
| Imports | __future__, dataclasses, peft, torch, typing, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** LoRA configuration classes

**Mechanism:** Defines configuration dataclasses for LoRA and its variants. LoraConfig is the main configuration class inheriting from PeftConfig, specifying parameters like rank (r), alpha, dropout, target modules, and initialization methods. Includes specialized configs: LoftQConfig for quantization-aware initialization, ArrowConfig for mixture-of-experts routing, BdLoraConfig for budget-aware decomposition, EvaConfig for eigenvalue-based adaptation, and CordaConfig for covariance-based initialization. Each config uses dataclasses with field validation and default values.

**Significance:** Core configuration infrastructure that controls LoRA behavior across the entire PEFT library. These configs enable users to customize adapter parameters, choose initialization strategies (random, PiSSA, OLoRA, LoftQ, CorDA), specify target modules, and configure advanced variants. Critical for reproducibility and flexibility in fine-tuning workflows.
