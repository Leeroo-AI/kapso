# File: `tests/utils/aime_eval.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 545 |
| Functions | `download_and_combine_aime_datasets`, `load_aime_dataset`, `extract_aime_answer`, `get_num_tokens`, `evaluate_model_aime`, `compare_aime_results` |
| Imports | json, logging, os, re, requests, tqdm, typing, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive evaluation framework for testing language models on AIME (American Invitational Mathematics Examination) dataset with Pass@k sampling and detailed accuracy tracking.

**Mechanism:** Implements complete AIME evaluation pipeline: (1) download_and_combine_aime_datasets fetches three JSONL files (test2024, test2025-I, test2025-II) from GAIR-NLP GitHub repo, assigns global IDs, and combines into single dataset with source tracking, (2) load_aime_dataset formats problems as chat messages with system/user/assistant structure, (3) extract_aime_answer uses regex patterns to find numerical answers (0-999 range) in model responses including boxed notation and natural language patterns, (4) evaluate_model_aime runs vLLM inference with n_sampling (default 8) responses per problem using SamplingParams, calculates Pass@k (probability at least one sample is correct), tracks per-source accuracy, and saves detailed records with token statistics, (5) compare_aime_results generates comparison tables across multiple model runs with improvement analysis. Suppresses verbose vLLM logging during evaluation.

**Significance:** Specialized benchmark for mathematical reasoning capability of language models. AIME problems require advanced problem-solving skills, making this a challenging evaluation for testing model improvements from fine-tuning. The Pass@k metric with multiple samples (n=8) is more robust than single-shot accuracy for difficult problems. Per-source tracking allows analysis across different AIME competition years. Critical for validating that Unsloth-optimized models maintain reasoning abilities after fine-tuning.
