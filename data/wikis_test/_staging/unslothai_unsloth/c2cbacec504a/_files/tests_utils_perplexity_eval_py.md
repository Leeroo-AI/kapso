# File: `tests/utils/perplexity_eval.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 81 |
| Functions | `ppl_model`, `add_to_comparison`, `print_model_comparison` |
| Imports | pandas, torch, tqdm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides perplexity calculation utilities for language model evaluation, computing cross-entropy loss as a measure of model quality and tracking results across multiple models.

**Mechanism:** Computes perplexity by running models on validation datasets with tqdm progress tracking, calculates negative log-likelihood of sequences, maintains comparison data structure with pandas, and generates formatted comparison tables showing relative performance.

**Significance:** Core evaluation utility for quantitatively measuring language model quality, enabling objective comparison between base models and fine-tuned versions to validate that training improves or maintains model performance.
