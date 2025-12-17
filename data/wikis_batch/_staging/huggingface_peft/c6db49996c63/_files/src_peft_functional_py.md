# File: `src/peft/functional.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 34 |
| Imports | peft |

## Understanding

**Status:** ✅ Explored

**Purpose:** Exposes functional API for PEFT operations that work with non-PeftModel models.

**Mechanism:** Re-exports key functions from internal modules: cast_adapter_dtype, delete_adapter, get_peft_model_state_dict, inject_adapter_in_model, set_adapter, set_peft_model_state_dict, and set_requires_grad. These functions can be used directly on models that have PEFT adapters injected without wrapping them in PeftModel.

**Significance:** Provides a functional programming interface for PEFT operations, useful for integration with packages like transformers and diffusers that manage their own model wrappers. Allows low-level manipulation of PEFT adapters without the full PeftModel abstraction. Important for advanced use cases and library integrations.
