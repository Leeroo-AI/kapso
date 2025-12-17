# File: `tests/test_helpers.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 473 |
| Classes | `TestCheckIsPeftModel`, `TestScalingAdapters`, `TestDisableInputDtypeCasting`, `MLP` |
| Imports | diffusers, peft, pytest, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for PEFT helper utilities including model detection, adapter scaling, and dtype casting.

**Mechanism:** Contains three test classes - TestCheckIsPeftModel validates the check_if_peft_model helper with Hub models, local models, and edge cases. TestScalingAdapters tests the rescale_adapter_scale context manager for temporarily adjusting LoRA scaling factors across various scenarios (single model, pipelines, merged adapters, multi-adapters). TestDisableInputDtypeCasting validates the disable_input_dtype_casting context manager that prevents automatic input dtype conversion during mixed precision scenarios.

**Significance:** Validates critical utility functions that support advanced PEFT workflows. The scaling feature enables dynamic adapter strength adjustment without retraining, dtype casting control prevents precision issues with custom training loops, and model detection helps users identify PEFT models programmatically.
