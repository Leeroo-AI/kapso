# File: `src/peft/tuners/mixed/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 18 |
| Imports | model |

## Understanding

**Status:** ✅ Documented

**Purpose:** Package initialization file that exports COMPATIBLE_TUNER_TYPES and MixedModel for using multiple different adapter types simultaneously in a single model.

**Mechanism:** Simply imports and re-exports COMPATIBLE_TUNER_TYPES (LoRA, LoHa, LoKr, AdaLoRA, OFT, Shira) and MixedModel from the model module. Does not register a PEFT method since mixed adapters work by composing existing registered methods.

**Significance:** Entry point for the Mixed adapter functionality, which allows combining different adapter types (e.g., LoRA on some layers, LoHa on others) within the same model. This provides maximum flexibility for experimentation and optimization.
