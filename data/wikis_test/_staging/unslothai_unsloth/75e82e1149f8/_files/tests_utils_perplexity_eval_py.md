# File: `tests/utils/perplexity_eval.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 81 |
| Functions | `ppl_model`, `add_to_comparison`, `print_model_comparison` |
| Imports | pandas, torch, tqdm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Perplexity evaluation utility

**Mechanism:** Computes model perplexity on datasets using sliding window approach for long sequences, compares multiple models

**Significance:** Standard metric for evaluating language model quality
