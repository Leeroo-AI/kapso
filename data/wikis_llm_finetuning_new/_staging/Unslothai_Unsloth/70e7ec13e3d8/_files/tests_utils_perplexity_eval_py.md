# File: `tests/utils/perplexity_eval.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 81 |
| Functions | `ppl_model`, `add_to_comparison`, `print_model_comparison` |
| Imports | pandas, torch, tqdm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Evaluates language model quality by computing perplexity on datasets and enabling multi-model comparison.

**Mechanism:** The `ppl_model()` function computes perplexity using a sliding window approach (max_length=2048, stride=512) to handle long sequences, iterating through dataset examples, tokenizing each, creating attention masks based on pad token presence, computing negative log-likelihood losses via model forward pass with labels set to -100 for context tokens, and returning exp(mean(nlls)). A global `model_comparison_results` dictionary stores perplexity scores per model name via `add_to_comparison()`. The `print_model_comparison()` function creates a pandas DataFrame and prints formatted comparison tables, handling tensor-to-float conversion.

**Significance:** Core evaluation utility for benchmarking fine-tuned model quality, enabling systematic comparison of different training configurations and checkpoint selections based on the standard perplexity metric.
