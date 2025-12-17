# File: `tests/saving/language_models/test_merged_model_perplexity_qwen_2.5.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 311 |
| Functions | `formatting_prompts_func`, `load_and_compute_8bit_ppl` |
| Imports | datasets, gc, multiprocessing, pandas, pathlib, sys, tests, torch, tqdm, transformers, ... +2 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Measures perplexity for Qwen 2.5 models through the complete training and merging workflow to validate Unsloth's support for Qwen's architecture and multilingual capabilities.

**Mechanism:** Trains Qwen 2.5 models with QLoRA on formatted instruction data, merges trained adapters, computes perplexity by evaluating language modeling loss, and generates detailed performance comparisons using pandas for statistical analysis.

**Significance:** Validates Unsloth's compatibility with Qwen models, which are important for Chinese and multilingual NLP tasks, ensuring the training pipeline preserves model quality across different language model architectures and tokenization schemes.
