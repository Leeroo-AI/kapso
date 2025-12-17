# File: `src/peft/tuners/cpt/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 24 |
| Imports | config, model, peft |

## Understanding

**Status:** ✅ Explored

**Purpose:** Registers CPT (Context-aware Prompt Tuning) as a PEFT method and exposes its public API.

**Mechanism:** Imports CPT components (CPTConfig, CPTEmbedding) and registers the method with PEFT's method registry using register_peft_method, making it available as a prompt learning technique for causal language models.

**Significance:** Entry point for CPT, a prompt-learning method that extends standard prompt tuning with context-aware features including token type masks, weighted loss, and projection settings for improved prompt optimization.
