# File: `tests/utils/aime_eval.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 545 |
| Functions | `download_and_combine_aime_datasets`, `load_aime_dataset`, `extract_aime_answer`, `get_num_tokens`, `evaluate_model_aime`, `compare_aime_results` |
| Imports | json, logging, os, re, requests, tqdm, typing, vllm |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive evaluation module for benchmarking language models on the AIME (American Invitational Mathematics Examination) dataset, supporting combined evaluation across test2024, test2025-I, and test2025-II subsets.

**Mechanism:** The module provides a complete evaluation pipeline: (1) download_and_combine_aime_datasets fetches JSONL files from GAIR-NLP's GitHub repository, adds source tracking metadata (source_dataset, global_id), and saves a combined aime.jsonl file, (2) load_aime_dataset formats problems into chat-style prompts with system and user roles, (3) extract_aime_answer uses regex patterns to parse numerical answers (0-999 range) from model responses including boxed LaTeX format, (4) evaluate_model_aime runs batch inference using vLLM's SamplingParams with configurable temperature, n_sampling (for pass@k), and max_tokens, tracking per-problem correctness and source-level statistics, (5) compare_aime_results generates comparison tables across model configurations with improvement analysis. The evaluation suppresses verbose vLLM logging during inference and calculates metrics including accuracy, pass@k, and token statistics.

**Significance:** AIME is a challenging mathematical reasoning benchmark that tests advanced problem-solving capabilities. This evaluation module provides rigorous assessment of model performance on competition-level mathematics, with source-level breakdowns enabling analysis across different exam years. The pass@k metric with multiple samples per problem helps measure model consistency, and the tiered performance assessment (Exceptional 50%+, Excellent 30%+, etc.) provides interpretable quality benchmarks.
