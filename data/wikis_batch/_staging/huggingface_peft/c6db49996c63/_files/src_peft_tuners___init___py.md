# File: `src/peft/tuners/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 135 |
| Imports | adalora, adaption_prompt, boft, bone, c3a, cpt, delora, fourierft, gralora, hra, ... +22 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Central tuners module aggregator exposing all PEFT tuning method implementations.

**Mechanism:** Imports and re-exports configs and models from 30+ tuner implementations including LoRA, AdaLoRA, IA3, Prefix Tuning, Prompt Tuning, P-Tuning, BOFT, LoHa, LoKr, and many others, providing unified access to all PEFT methods.

**Significance:** Core public API for the tuners subsystem, serving as the single entry point for accessing any PEFT method's configuration and model classes, essential for the factory pattern used in PeftModel creation.
