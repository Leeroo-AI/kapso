# File: `tests/utils/perplexity_eval.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 81 |
| Functions | `ppl_model`, `add_to_comparison`, `print_model_comparison` |
| Imports | pandas, torch, tqdm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Calculates and compares perplexity scores for language models on text datasets. Perplexity is a key metric for evaluating how well a language model predicts sequences, with lower perplexity indicating better performance.

**Mechanism:** The core evaluation happens in `ppl_model()`:
- Processes each text sample in the dataset with a sliding window approach for sequences longer than 512 tokens
- Uses a max context length of 2048 tokens with 512-token stride between windows
- For each window, computes cross-entropy loss between model predictions and target tokens
- Creates proper attention masks based on pad token IDs
- Aggregates negative log-likelihoods across all windows and samples
- Calculates final perplexity as `exp(mean(neg_log_likelihoods))`

The helper functions `add_to_comparison()` and `print_model_comparison()` maintain a global `model_comparison_results` dictionary to track and display perplexity scores across multiple models, with pandas DataFrame formatting for clear comparison tables.

**Significance:** Important for benchmarking language model quality after fine-tuning or quantization. Perplexity evaluation helps verify that unsloth's optimizations don't degrade model performance. The comparison features enable systematic testing of multiple model configurations.
