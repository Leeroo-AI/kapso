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

**Purpose:** Converts ShareGPT conversation format to OpenAI API format for benchmarking.

**Mechanism:** Transforms ShareGPT conversation datasets into OpenAI-compatible format used by vLLM's API server for standardized benchmarking.

**Significance:** Enables using popular ShareGPT datasets for multi-turn benchmarks. Important for reproducible evaluation on common datasets.
