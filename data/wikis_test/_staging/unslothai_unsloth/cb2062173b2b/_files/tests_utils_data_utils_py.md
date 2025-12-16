# File: `tests/utils/data_utils.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 153 |
| Functions | `create_instruction_dataset`, `create_dataset`, `describe_param`, `format_summary`, `get_peft_weights`, `describe_peft_weights`, `check_responses` |
| Imports | datasets, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Test data utilities providing dataset creation, model weight analysis, and response validation helpers for fine-tuning tests.

**Mechanism:** Implements seven utility functions organized in three categories: (1) Dataset creation - create_instruction_dataset wraps messages in HF Dataset format, create_dataset applies chat templates and optionally repeats/truncates to specified size using default messages (user asks "What day was I born?", assistant answers "January 1, 2058"), (2) Weight analysis - describe_param computes statistical summary of tensor (shape, mean, std, min, max, percentiles, optional L1/L2/infinity norms), format_summary renders stats dictionary as formatted string with configurable precision, get_peft_weights filters model parameters for lora_A/lora_B weights, describe_peft_weights generates (name, stats) tuples for all LoRA weights, (3) Response validation - check_responses verifies each response contains expected answer string and prints success/failure with response content. Uses bfloat16 as default dtype.

**Significance:** Reusable test infrastructure reducing boilerplate in fine-tuning tests. The dataset creation functions provide consistent test data across test suites. Weight analysis functions enable validation that LoRA adapters are properly initialized and trained (checking for non-zero weights, reasonable distributions). Response checking automates output validation for instruction-following tests. Together these utilities make tests more maintainable and readable.
