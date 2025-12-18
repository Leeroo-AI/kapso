# File: `src/peft/functional.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 34 |
| Imports | peft |

## Understanding

**Status:** ✅ Explored

**Purpose:** Public functional API for non-PeftModel integrations.

**Mechanism:** Re-exports key functions as stable public API for external packages (transformers, diffusers) that integrate PEFT without using PeftModel wrapper: `inject_adapter_in_model`, `cast_adapter_dtype`, `delete_adapter`, `set_adapter`, `set_requires_grad`, `get_peft_model_state_dict`, `set_peft_model_state_dict`. Provides explicit `__all__` defining the stable interface contract.

**Significance:** Stable API surface for integrations. Functions here are "safe" for external packages to use without breaking changes.
