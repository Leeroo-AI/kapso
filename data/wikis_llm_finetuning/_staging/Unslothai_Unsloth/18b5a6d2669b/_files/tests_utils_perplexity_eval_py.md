# File: `tests/utils/perplexity_eval.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 81 |
| Functions | `ppl_model`, `add_to_comparison`, `print_model_comparison` |
| Imports | pandas, torch, tqdm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Computes perplexity scores for language models on text datasets and provides comparison reporting utilities for evaluating multiple models.

**Mechanism:** Uses a sliding window approach (max_length=2048, stride=512) to process sequences longer than the context window, computing negative log-likelihood on each example individually. Maintains a global comparison tracker (model_comparison_results) and generates comparison tables using pandas to display perplexity metrics across different models.

**Significance:** Critical evaluation utility for measuring and comparing model quality during testing. Perplexity is a standard metric for language model performance, with lower scores indicating better predictive capability. The sliding window approach ensures accurate evaluation even on long sequences.
