# File: `src/peft/utils/peft_types.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 183 |
| Classes | `PeftType`, `TaskType` |
| Functions | `register_peft_method` |
| Imports | enum, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines enumerations for PEFT types and task types, plus registration system for adding custom PEFT methods to the library.

**Mechanism:** PeftType enum lists all supported adapter methods (LORA, ADALORA, BOFT, PROMPT_TUNING, IA3, etc. - 25+ types). TaskType enum defines ML task categories (SEQ_CLS, CAUSAL_LM, TOKEN_CLS, etc.). register_peft_method() allows plugins to register custom methods by adding to global mappings (PEFT_TYPE_TO_CONFIG_MAPPING, PEFT_TYPE_TO_TUNER_MAPPING, PEFT_TYPE_TO_PREFIX_MAPPING).

**Significance:** Central type registry that makes PEFT extensible. Acts as the single source of truth for supported PEFT methods and provides a standardized way to add new methods without modifying core library code. Registration validates uniqueness and consistency of method names, prefixes, and mixed model compatibility.
