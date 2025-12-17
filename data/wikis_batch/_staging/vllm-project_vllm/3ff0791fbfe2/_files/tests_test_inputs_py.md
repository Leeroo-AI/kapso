# File: `tests/test_inputs.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 125 |
| Functions | `test_invalid_input_raise_type_error`, `test_parse_raw_single_batch_empty`, `test_parse_raw_single_batch_string_consistent`, `test_parse_raw_single_batch_token_consistent`, `test_parse_raw_single_batch_string_slice`, `test_zip_enc_dec_prompts`, `test_preprocessor_always_mm_code_path` |
| Imports | pytest, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Input parsing and preprocessing tests

**Mechanism:** Tests parse_raw_prompts for string/token input handling, batch consistency, and slice operations. Tests zip_enc_dec_prompts for encoder-decoder models with multimodal processor kwargs. Includes skipped test for multimodal preprocessing performance.

**Significance:** Validates input transformation pipeline handles various prompt formats correctly before they reach the model, ensuring consistent behavior across text, token, and multimodal inputs.
