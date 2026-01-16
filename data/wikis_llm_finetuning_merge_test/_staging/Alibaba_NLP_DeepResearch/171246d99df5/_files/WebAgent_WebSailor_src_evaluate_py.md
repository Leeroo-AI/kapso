# File: `WebAgent/WebSailor/src/evaluate.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 329 |
| Functions | `extract_correct_judgement`, `call_llm_judge`, `process_single_round`, `aggregate_statistics`, `single_round_statistics`, `aggregate_results`, `calculate_pass_at_k`, `calculate_best_pass_at_1`, `... +2 more` |
| Imports | argparse, concurrent, json, openai, os, prompt, re, tiktoken, tqdm, traceback |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Evaluates WebSailor agent predictions against ground-truth answers across multiple evaluation rounds, computing accuracy metrics like Pass@k and aggregating statistics about tool usage and response quality.

**Mechanism:** The file loads prediction results from three iteration files (iter1.jsonl, iter2.jsonl, iter3.jsonl) and uses an LLM judge (Qwen2.5-72B) via a local vLLM server to determine if predictions match ground-truth answers. It supports multiple datasets (GAIA, BrowseComp, SimpleQA, TimeQA, etc.) with dataset-specific judge prompts. Key functions include `call_llm_judge()` for LLM-based evaluation, `single_round_statistics()` for analyzing tool usage patterns (search/visit actions), answer lengths, and token counts, and `calculate_pass_at_k()` / `calculate_avg_pass_at_3()` / `calculate_best_pass_at_1()` for computing different accuracy metrics. Results are aggregated using concurrent execution with ThreadPoolExecutor.

**Significance:** Core evaluation component for the WebSailor system that provides standardized benchmarking across multiple datasets and rounds, enabling comparison of model performance and analysis of agent behavior patterns.
