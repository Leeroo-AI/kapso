# File: `tests/test_inputs.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 125 |
| Functions | `test_invalid_input_raise_type_error`, `test_parse_raw_single_batch_empty`, `test_parse_raw_single_batch_string_consistent`, `test_parse_raw_single_batch_token_consistent`, `test_parse_raw_single_batch_string_slice`, `test_zip_enc_dec_prompts`, `test_preprocessor_always_mm_code_path` |
| Imports | pytest, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Input parsing and validation tests

**Mechanism:** Tests input preprocessing including: mixed-type validation (raises TypeError), empty prompt handling, single vs batched input consistency for strings and token IDs, slicing behavior, encoder-decoder prompt zipping with mm_processor_kwargs, and multimodal model preprocessing (currently skipped due to performance concerns).

**Significance:** Ensures robust input handling for various prompt formats (strings, token IDs, embeddings, multimodal) and validates encoder-decoder model support. Critical for API compatibility and correctness.
