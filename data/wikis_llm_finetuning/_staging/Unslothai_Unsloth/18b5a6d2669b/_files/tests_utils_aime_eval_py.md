# File: `tests/utils/aime_eval.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 545 |
| Functions | `download_and_combine_aime_datasets`, `load_aime_dataset`, `extract_aime_answer`, `get_num_tokens`, `evaluate_model_aime`, `compare_aime_results` |
| Imports | json, logging, os, re, requests, tqdm, typing, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive evaluation module for testing language models on AIME (American Invitational Mathematics Examination) datasets. Provides dataset downloading, model evaluation with multiple sampling, answer extraction, and performance comparison functionality.

**Mechanism:** Downloads and combines AIME datasets (test2024, test2025-I, test2025-II) from GitHub, formats problems as chat messages, uses vLLM for efficient model inference with configurable sampling parameters (temperature=0.3, n=8 samples, max_tokens=32768). Extracts numerical answers (0-999) from model responses using regex patterns, calculates Pass@k metrics, tracks performance by source dataset, and generates detailed JSON reports with token statistics. Includes model comparison functionality for benchmarking multiple models.

**Significance:** Enables rigorous mathematical reasoning evaluation on challenging AIME problems (requiring high-school mathematics competition level). Critical for validating that fine-tuned models maintain or improve mathematical problem-solving capabilities. The multi-sampling approach with Pass@k metrics provides robust evaluation even when models produce inconsistent outputs. Essential for research and development of reasoning-focused models.
