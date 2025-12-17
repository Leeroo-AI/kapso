# File: `tests/test_mapping.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 55 |
| Classes | `TestGetPeftModel` |
| Imports | peft, pytest, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for PEFT model loading and reloading behavior

**Mechanism:** Validates that `get_peft_model` warns when reloading an already adapted model, verifies the suggested fix (unload before reloading), and tests repeated invocation of `get_peft_model` on existing PEFT models with warnings

**Significance:** Ensures proper user feedback when attempting to apply adapters to already-adapted models, preventing accidental double-wrapping or configuration conflicts
