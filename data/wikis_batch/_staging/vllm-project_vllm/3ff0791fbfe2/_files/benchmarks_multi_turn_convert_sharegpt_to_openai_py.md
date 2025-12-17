# File: `benchmarks/multi_turn/convert_sharegpt_to_openai.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 354 |
| Functions | `has_non_english_chars`, `content_is_valid`, `print_stats`, `convert_sharegpt_to_openai`, `main` |
| Imports | argparse, json, pandas, random, statistics, tqdm, transformers, typing |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Converts ShareGPT dataset to OpenAI format

**Mechanism:** Data conversion tool for transforming ShareGPT conversation format to OpenAI chat completion format. Validates conversation content (checks for non-English chars, content validity). Filters conversations by length, turn count, and content quality. Uses transformers tokenizer for token counting. Prints statistics about filtered conversations (token distributions, turn counts). Supports random sampling for dataset size control. Outputs cleaned, standardized JSON dataset suitable for multi-turn benchmarking.

**Significance:** Essential preprocessing tool for using public conversation datasets (ShareGPT) with vLLM multi-turn benchmarks. ShareGPT format is common but differs from OpenAI standard. Conversion enables leveraging existing conversation datasets for realistic benchmark workloads. Filtering ensures high-quality benchmark data by removing invalid/low-quality conversations. Critical for reproducible multi-turn benchmarking with standardized datasets.
