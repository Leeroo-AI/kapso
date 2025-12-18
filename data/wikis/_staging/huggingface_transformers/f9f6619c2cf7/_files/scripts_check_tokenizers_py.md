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

**Purpose:** Validates that fast tokenizer implementations produce equivalent results to their slow (Python-based) counterparts by comparing tokenization outputs across multilingual text samples.

**Mechanism:** Iterates through all tokenizer classes that have fast implementations (from SLOW_TO_FAST_CONVERTERS), loads both slow and fast versions from each checkpoint, and tests them against the XNLI dataset (multilingual test and validation splits). For each text sample, it compares the encoded token IDs and uses sophisticated heuristics in check_details() to determine if differences are acceptable (e.g., reversible tokenization like "AAA" -> "AA+A" vs "A+AA", equivalent decodings, handling of LTR marks, re-encoding consistency). Tracks statistics for perfect matches, imperfect but acceptable matches, and wrong tokenizations, printing accuracy percentages for each tokenizer checkpoint.

**Significance:** Essential quality assurance tool for the transformers library, ensuring that the Rust-based fast tokenizers maintain compatibility with the original slow implementations. This is critical because fast tokenizers are preferred for production use due to performance, but they must produce identical or equivalently valid results. The script helps catch tokenization discrepancies that could lead to model degradation or incorrect behavior when users switch between slow and fast tokenizers.
