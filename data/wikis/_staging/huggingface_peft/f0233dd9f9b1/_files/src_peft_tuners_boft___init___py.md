# File: `src/peft/tuners/boft/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 24 |
| Imports | config, layer, model, peft |

## Understanding

**Status:** ✅ Explored

**Purpose:** BOFT module initialization

**Mechanism:** Exports BOFTConfig, BOFTLayer, BOFTModel classes and registers "boft" as PEFT method for butterfly factorization-based orthogonal fine-tuning.

**Significance:** Entry point for BOFT tuning method from ICLR 2024 paper (https://huggingface.co/papers/2311.06243). Enables parameter-efficient orthogonal weight updates via butterfly factorization.
