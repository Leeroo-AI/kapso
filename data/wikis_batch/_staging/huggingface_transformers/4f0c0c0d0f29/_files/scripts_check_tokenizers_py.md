# File: `scripts/check_tokenizers.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 179 |
| Functions | `check_diff`, `check_LTR_mark`, `check_details`, `test_string`, `test_tokenizer` |
| Imports | collections, datasets, transformers |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Validates and compares encoding consistency between slow (Python-based) and fast (Rust-based) tokenizer implementations across multiple languages and tokenizer types.

**Mechanism:** Loads XNLI multilingual dataset and encodes text with both slow and fast tokenizer versions. Implements sophisticated difference checking including handling reversed token splits, equivalent decodings, re-encoding validation, and special character handling (LTR marks). Categorizes mismatches as perfect, imperfect (acceptable differences), or wrong, recursively subdividing complex differences to isolate problematic segments. Reports accuracy statistics per tokenizer checkpoint.

**Significance:** Critical quality assurance tool ensuring fast tokenizers maintain parity with their slow counterparts after conversion. Helps identify edge cases, encoding inconsistencies, and potential bugs in tokenizer implementations. Essential for maintaining backward compatibility and reliability when migrating tokenizers to faster Rust-based implementations, particularly important given tokenizers are foundational to all model input processing.
