# File: `tests/utils/aime_eval.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 545 |
| Functions | `download_and_combine_aime_datasets`, `load_aime_dataset`, `extract_aime_answer`, `get_num_tokens`, `evaluate_model_aime`, `compare_aime_results` |
| Imports | json, logging, os, re, requests, tqdm, typing, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides evaluation utilities for testing language models on AIME (American Invitational Mathematics Examination) problems to measure mathematical reasoning capabilities.

**Mechanism:** Downloads AIME math problem datasets, loads and formats problems for model inference, extracts numerical answers from model responses using regex, evaluates models using vLLM for efficient inference, computes accuracy metrics, and compares results across different model checkpoints.

**Significance:** Enables rigorous mathematical reasoning benchmarking for Unsloth-trained models, providing quantitative metrics for advanced reasoning capabilities essential for validating model quality on complex STEM problems.
