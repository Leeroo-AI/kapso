# File: `src/peft/tuners/cpt/__init__.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 24 |
| Imports | config, model, peft |

## Understanding

**Status:** ✅ Explored

**Purpose:** Package initialization file for the CPT (Context-aware Prompt Tuning) PEFT method that exports key components and registers CPT as a PEFT method in the library.

**Mechanism:** The file imports CPTConfig and CPTEmbedding from their respective modules, exposes them via __all__, and calls register_peft_method() to register CPT with the PEFT framework, mapping the name "cpt" to the CPTConfig configuration class and CPTEmbedding model class. This registration enables users to instantiate CPT adapters using the standard PEFT API.

**Significance:** This is a core initialization file that makes CPT available as a first-class PEFT method. CPT is a prompt learning technique that uses context-aware soft prompts with constrained optimization (epsilon-based projection) and optional loss weighting for few-shot learning tasks. The registration here integrates it into PEFT's adapter ecosystem alongside methods like LoRA, IA3, etc.
