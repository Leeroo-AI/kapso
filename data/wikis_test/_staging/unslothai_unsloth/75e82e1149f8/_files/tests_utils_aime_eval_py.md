# File: `tests/utils/aime_eval.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 545 |
| Functions | `download_and_combine_aime_datasets`, `load_aime_dataset`, `extract_aime_answer`, `get_num_tokens`, `evaluate_model_aime`, `compare_aime_results` |
| Imports | json, logging, os, re, requests, tqdm, typing, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** AIME evaluation framework

**Mechanism:** Downloads AIME datasets (2024, 2025-I, 2025-II), evaluates models on mathematical problems, extracts numerical answers, calculates accuracy and Pass@K metrics

**Significance:** Critical for benchmarking models on advanced mathematics problems
