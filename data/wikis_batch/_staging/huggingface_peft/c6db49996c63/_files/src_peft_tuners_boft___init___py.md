# File: `src/peft/tuners/boft/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 24 |
| Imports | config, layer, model, peft |

## Understanding

**Status:** ✅ Explored

**Purpose:** Module initialization and registration for BOFT (Butterfly Orthogonal Finetuning) method

**Mechanism:** Imports and exposes BOFTConfig, BOFTLayer, and BOFTModel classes, then registers "boft" as a PEFT method using register_peft_method

**Significance:** Entry point for the BOFT tuner, enabling it to be discovered and used as a PEFT adaptation technique
