# File: `tests/utils/data_utils.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 153 |
| Functions | `create_instruction_dataset`, `create_dataset`, `describe_param`, `format_summary`, `get_peft_weights`, `describe_peft_weights`, `check_responses` |
| Imports | datasets, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides helper functions and test fixtures for creating datasets and analyzing model parameters during testing.

**Mechanism:** Defines default test message structures (QUESTION/ANSWER pairs), implements `create_instruction_dataset()` to generate HuggingFace Dataset objects from message dictionaries, `create_dataset()` to apply chat templates and optionally repeat/truncate examples, `describe_param()` for statistical tensor summaries (mean, std, min, max, percentiles, optional L1/L2/infinity norms), `format_summary()` for pretty-printing statistics, `get_peft_weights()` to filter LoRA adapter weights, `describe_peft_weights()` as a generator yielding weight statistics, and `check_responses()` to verify if expected answers appear in model responses.

**Significance:** Core test utility module that standardizes dataset creation and model weight inspection across the test suite, enabling consistent validation of fine-tuning behavior and LoRA adapter updates.
