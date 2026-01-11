# File: `tests/utils/data_utils.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 153 |
| Functions | `create_instruction_dataset`, `create_dataset`, `describe_param`, `format_summary`, `get_peft_weights`, `describe_peft_weights`, `check_responses` |
| Imports | datasets, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Test data generation and model inspection utilities. Provides functions for creating instruction datasets, describing tensor statistics, analyzing PEFT/LoRA weights, and validating model responses.

**Mechanism:** create_dataset() generates test datasets from message templates, applies chat templates, and optionally repeats to reach desired size. describe_param() computes comprehensive tensor statistics (mean, std, percentiles, optional L1/L2/infinity norms) with formatted output. get_peft_weights() filters model parameters for LoRA weights (lora_A/lora_B). check_responses() validates that model outputs contain expected answers. Uses default test data (birthday question/answer) for consistent testing.

**Significance:** Provides standardized test data creation and model inspection tools used across multiple test files. The tensor statistics functions are valuable for debugging training issues and verifying that LoRA adapters are learning properly. The response checking helps validate that fine-tuned models produce correct outputs. Reduces code duplication and ensures consistent test data formats.
