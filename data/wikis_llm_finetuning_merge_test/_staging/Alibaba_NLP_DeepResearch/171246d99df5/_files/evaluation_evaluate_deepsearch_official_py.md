# File: `evaluation/evaluate_deepsearch_official.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 584 |
| Functions | `get_client`, `is_correct_judgement`, `call_llm_judge`, `process_single_round`, `get_termination_value`, `count_tokens_with_tokenizer`, `aggregate_statistics`, `single_round_statistics`, `... +6 more` |
| Imports | argparse, concurrent, json, litellm, openai, os, prompt, pydantic, re, threading, ... +6 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Evaluates deep search model predictions across multiple datasets (GAIA, BrowseComp, WebWalker, XBench-DeepSearch) by using LLM judges to assess answer correctness and computing pass@k metrics across three evaluation rounds.

**Mechanism:** The script processes three rounds of predictions from JSONL files (iter1.jsonl, iter2.jsonl, iter3.jsonl). For each prediction, it calls an LLM judge (GPT-4o, Qwen2.5-72B, or Gemini-2.0-Flash depending on dataset) using dataset-specific prompts from `prompt.py` to determine if the model's answer matches the correct answer. It computes multiple metrics including: (1) Pass@1/Pass@3 accuracy rates, (2) average tool usage statistics (search/visit actions), (3) token consumption analysis, (4) termination frequency analysis (answered vs max_turns_reached vs max_tokens_reached), and (5) enhanced statistics for correctly solved questions. Results are written to scored JSONL files and a summary report.

**Significance:** Core evaluation component for benchmarking deep search systems. It implements the official evaluation methodology referenced in the DeepResearch paper, enabling reproducible comparison across different models and datasets. The multi-round evaluation with pass@k metrics accounts for the stochastic nature of LLM-based search agents.
