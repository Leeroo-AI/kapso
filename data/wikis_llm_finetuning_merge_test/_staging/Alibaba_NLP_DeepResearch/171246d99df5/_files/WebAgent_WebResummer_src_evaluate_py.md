# File: `WebAgent/WebResummer/src/evaluate.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 309 |
| Functions | `call_llm_judge`, `single_round_statistics`, `process_one_prediction` |
| Imports | argparse, collections, concurrent, dashscope, glob, json, judge_prompt, os, time, tqdm, ... +1 more |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Evaluates agent predictions against ground-truth answers using LLM-based judging, computing accuracy metrics (Pass@1, Pass@k) and trajectory statistics for web research benchmarks.

**Mechanism:** Uses DashScope API with Qwen2.5-72B-Instruct to judge whether predicted answers match ground-truth by applying dataset-specific judge prompts (GAIA, BrowseComp English/Chinese). The `call_llm_judge` function formats prompts and interprets LLM responses. `single_round_statistics` calculates metrics like average answer length, trajectory length, and tool invocation counts using tokenizers. `process_one_prediction` handles scoring of prediction files with caching (scored files), supporting parallel processing via ThreadPoolExecutor. The main script aggregates results across multiple rollout iterations to compute average/best Pass@1 and Pass@k scores.

**Significance:** Core evaluation component for the WebResummer system. Enables benchmarking of web research agents on standard datasets, providing quantitative metrics for model performance assessment and comparison.
