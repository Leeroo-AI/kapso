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

**Purpose:** Defines core type enumerations and registration system for PEFT methods and task types.

**Mechanism:** Provides PeftType enum (LORA, PREFIX_TUNING, PROMPT_TUNING, P_TUNING, IA3, etc.) and TaskType enum (SEQ_CLS, SEQ_2_SEQ_LM, CAUSAL_LM, TOKEN_CLS, etc.) along with register_peft_method() function for dynamic PEFT method registration.

**Significance:** Foundational type system that enables type-safe PEFT method identification, configuration validation, and dynamic extension of PEFT with custom methods, serving as the registry backbone for all supported fine-tuning techniques.
