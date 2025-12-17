# File: `src/peft/tuners/lora/config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 879 |
| Classes | `LoraRuntimeConfig`, `LoftQConfig`, `ArrowConfig`, `BdLoraConfig`, `EvaConfig`, `CordaConfig`, `LoraConfig` |
| Imports | __future__, dataclasses, peft, torch, typing, warnings |

## Understanding

**Status:** ✅ Documented

**Purpose:** Configuration dataclasses for all LoRA variants and initialization methods

**Mechanism:** Defines LoraConfig (main configuration with rank, alpha, dropout, target modules) plus specialized configs: LoftQConfig (quantization-aware init), ArrowConfig (mixture-of-experts routing), BdLoraConfig (block-diagonal structure), EvaConfig (eigenvalue-based init), CordaConfig (correlation-based decomposition), and LoraRuntimeConfig (runtime optimization settings).

**Significance:** Central configuration hub that controls LoRA behavior across all variants. Provides extensive hyperparameter control including per-layer rank patterns, initialization strategies (random, PiSSA, OLoRA, LoftQ, etc.), bias handling, and integration with various quantization schemes. Essential for reproducibility and experimentation.
